#!/usr/bin/env python3
"""
LLaMA模型在海光DCU上的训练示例
支持分布式训练、混合精度、梯度累积等高级功能
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from datasets import Dataset, load_dataset
import wandb

@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "预训练模型名称或路径"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "模型缓存目录"}
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "是否使用Flash Attention"}
    )

@dataclass
class DataArguments:
    """数据相关参数"""
    dataset_name: Optional[str] = field(
        default="wikitext",
        metadata={"help": "数据集名称"}
    )
    dataset_config: Optional[str] = field(
        default="wikitext-2-raw-v1",
        metadata={"help": "数据集配置"}
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "最大序列长度"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "数据预处理进程数"}
    )

@dataclass
class TrainingArguments:
    """训练相关参数"""
    output_dir: str = field(
        default="./llama_output",
        metadata={"help": "输出目录"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "训练轮数"}
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "每个设备的训练批次大小"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "梯度累积步数"}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "学习率"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "权重衰减"}
    )
    warmup_steps: int = field(
        default=500,
        metadata={"help": "预热步数"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "日志记录间隔"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "模型保存间隔"}
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "评估间隔"}
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "是否使用FP16混合精度"}
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "数据加载器进程数"}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "是否移除未使用的列"}
    )
    report_to: Optional[str] = field(
        default="wandb",
        metadata={"help": "实验跟踪工具"}
    )

class DCUTrainer:
    """DCU训练器"""
    
    def __init__(self, model_args, data_args, training_args):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        
        # 设置分布式训练
        self.setup_distributed()
        
        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        
    def setup_distributed(self):
        """设置分布式训练环境"""
        if 'WORLD_SIZE' in os.environ:
            self.distributed = True
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
            
            # 初始化分布式
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.local_rank)
            
            print(f"初始化分布式训练: rank={self.rank}, local_rank={self.local_rank}, world_size={self.world_size}")
        else:
            self.distributed = False
            self.local_rank = 0
            self.world_size = 1
            self.rank = 0
    
    def load_tokenizer_and_model(self):
        """加载分词器和模型"""
        print("🔄 加载分词器和模型...")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            use_fast=True
        )
        
        # 设置padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            torch_dtype=torch.float16 if self.training_args.fp16 else torch.float32,
            device_map=None,  # 手动管理设备
        )
        
        # 移动模型到当前设备
        self.model = self.model.cuda(self.local_rank)
        
        # 分布式包装
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        
        print(f"✅ 模型加载完成: {self.model_args.model_name_or_path}")
        
        # 打印模型信息
        if self.rank == 0:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"📊 模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    
    def prepare_dataset(self):
        """准备训练数据集"""
        print("🔄 准备数据集...")
        
        # 加载数据集
        if self.data_args.dataset_name == "wikitext":
            dataset = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config,
                cache_dir=self.model_args.cache_dir
            )
        else:
            # 支持本地数据集
            dataset = load_dataset(
                "text",
                data_files=self.data_args.dataset_name,
                cache_dir=self.model_args.cache_dir
            )
        
        # 数据预处理函数
        def tokenize_function(examples):
            # 对文本进行分词
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.data_args.max_seq_length,
                return_special_tokens_mask=True,
            )
            return tokenized
        
        # 分组函数，将短文本合并到最大长度
        def group_texts(examples):
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            
            # 丢弃不足一个完整块的数据
            total_length = (total_length // self.data_args.max_seq_length) * self.data_args.max_seq_length
            
            result = {
                k: [t[i:i + self.data_args.max_seq_length] 
                    for i in range(0, total_length, self.data_args.max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            
            # 添加标签（语言建模任务中标签就是input_ids）
            result["labels"] = result["input_ids"].copy()
            return result
        
        # 应用分词
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing texts",
        )
        
        # 分组文本
        grouped_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            desc="Grouping texts",
        )
        
        self.train_dataset = grouped_datasets["train"]
        self.eval_dataset = grouped_datasets["validation"] if "validation" in grouped_datasets else None
        
        print(f"✅ 数据集准备完成: 训练样本 {len(self.train_dataset)}")
        if self.eval_dataset:
            print(f"   验证样本: {len(self.eval_dataset)}")
    
    def create_optimizer_and_scheduler(self):
        """创建优化器和学习率调度器"""
        # 创建优化器
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.training_args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.training_args.learning_rate
        )
        
        # 计算总训练步数
        num_training_steps = (
            len(self.train_dataset) // 
            (self.training_args.per_device_train_batch_size * self.world_size * self.training_args.gradient_accumulation_steps)
        ) * self.training_args.num_train_epochs
        
        # 创建学习率调度器
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.training_args.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # 混合精度
        if self.training_args.fp16:
            self.scaler = GradScaler()
        
        print(f"✅ 优化器和调度器创建完成")
        print(f"   总训练步数: {num_training_steps}")
        print(f"   预热步数: {self.training_args.warmup_steps}")
    
    def train(self):
        """开始训练"""
        print("🚀 开始训练...")
        
        # 创建数据加载器
        train_sampler = DistributedSampler(self.train_dataset) if self.distributed else None
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.training_args.per_device_train_batch_size,
            sampler=train_sampler,
            num_workers=self.training_args.dataloader_num_workers,
            pin_memory=True,
            collate_fn=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, 
                mlm=False
            )
        )
        
        # 初始化日志
        if self.rank == 0 and self.training_args.report_to == "wandb":
            wandb.init(
                project="dcu-llama-training",
                config={
                    **vars(self.model_args),
                    **vars(self.data_args),
                    **vars(self.training_args)
                }
            )
        
        # 训练循环
        global_step = 0
        self.model.train()
        
        for epoch in range(self.training_args.num_train_epochs):
            if self.distributed:
                train_sampler.set_epoch(epoch)
            
            epoch_loss = 0
            epoch_start_time = time.time()
            
            for step, batch in enumerate(train_dataloader):
                # 移动数据到设备
                batch = {k: v.cuda(self.local_rank, non_blocking=True) 
                        for k, v in batch.items()}
                
                # 前向传播
                if self.training_args.fp16:
                    with autocast():
                        outputs = self.model(**batch)
                        loss = outputs.loss / self.training_args.gradient_accumulation_steps
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.training_args.gradient_accumulation_steps
                
                # 反向传播
                if self.training_args.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                epoch_loss += loss.item()
                
                # 梯度累积
                if (step + 1) % self.training_args.gradient_accumulation_steps == 0:
                    if self.training_args.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    # 日志记录
                    if global_step % self.training_args.logging_steps == 0:
                        lr = self.lr_scheduler.get_last_lr()[0]
                        
                        if self.rank == 0:
                            print(f"Epoch {epoch+1}, Step {global_step}, Loss: {loss.item():.4f}, LR: {lr:.2e}")
                            
                            if self.training_args.report_to == "wandb":
                                wandb.log({
                                    "train/loss": loss.item(),
                                    "train/learning_rate": lr,
                                    "train/epoch": epoch + 1,
                                    "train/global_step": global_step
                                })
                    
                    # 保存模型
                    if global_step % self.training_args.save_steps == 0:
                        self.save_model(global_step)
            
            # 每个epoch结束时的日志
            epoch_time = time.time() - epoch_start_time
            avg_loss = epoch_loss / len(train_dataloader)
            
            if self.rank == 0:
                print(f"Epoch {epoch+1} 完成")
                print(f"  平均损失: {avg_loss:.4f}")
                print(f"  用时: {epoch_time:.2f}s")
                
                if self.training_args.report_to == "wandb":
                    wandb.log({
                        "train/epoch_loss": avg_loss,
                        "train/epoch_time": epoch_time,
                        "train/epoch": epoch + 1
                    })
        
        # 保存最终模型
        self.save_model("final")
        
        if self.rank == 0:
            print("🎉 训练完成！")
    
    def save_model(self, step):
        """保存模型"""
        if self.rank == 0:
            output_dir = Path(self.training_args.output_dir) / f"checkpoint-{step}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存模型和分词器
            if self.distributed:
                self.model.module.save_pretrained(output_dir)
            else:
                self.model.save_pretrained(output_dir)
            
            self.tokenizer.save_pretrained(output_dir)
            
            # 保存训练参数
            with open(output_dir / "training_args.json", "w") as f:
                json.dump(vars(self.training_args), f, indent=2)
            
            print(f"💾 模型已保存到: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="LLaMA模型DCU训练")
    
    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--cache_dir", type=str, default=None)
    
    # 数据参数
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    
    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./llama_output")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--report_to", type=str, default="none")
    
    args = parser.parse_args()
    
    # 创建参数对象
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        cache_dir=args.cache_dir
    )
    
    data_args = DataArguments(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        max_seq_length=args.max_seq_length
    )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=args.fp16,
        report_to=args.report_to
    )
    
    # 创建训练器
    trainer = DCUTrainer(model_args, data_args, training_args)
    
    # 加载模型和数据
    trainer.load_tokenizer_and_model()
    trainer.prepare_dataset()
    trainer.create_optimizer_and_scheduler()
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main() 