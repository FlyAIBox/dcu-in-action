#!/usr/bin/env python3
"""
基于LoRA的大模型微调示例
支持海光DCU，使用PEFT库实现高效参数微调
"""

import os
import json
import torch
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset, load_dataset
import wandb

@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: str = field(
        default="baichuan-inc/Baichuan2-7B-Chat",
        metadata={"help": "预训练模型名称或路径"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "模型缓存目录"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "是否信任远程代码"}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "是否使用认证令牌"}
    )

@dataclass
class DataArguments:
    """数据相关参数"""
    dataset_path: str = field(
        default="./data/alpaca_data.json",
        metadata={"help": "训练数据路径"}
    )
    validation_split_percentage: int = field(
        default=5,
        metadata={"help": "验证集分割比例"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "最大训练样本数"}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "最大评估样本数"}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "是否覆盖缓存"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "数据预处理进程数"}
    )

@dataclass
class LoRAArguments:
    """LoRA相关参数"""
    lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={"help": "LoRA目标模块"}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "LoRA bias类型"}
    )
    use_rslora: bool = field(
        default=False,
        metadata={"help": "是否使用RSLoRA"}
    )

@dataclass
class QuantizationArguments:
    """量化相关参数"""
    use_4bit: bool = field(
        default=True,
        metadata={"help": "是否使用4bit量化"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="float16",
        metadata={"help": "4bit计算数据类型"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "4bit量化类型"}
    )
    bnb_4bit_use_double_quant: bool = field(
        default=False,
        metadata={"help": "是否使用双重量化"}
    )

class LoRATrainer:
    """LoRA微调训练器"""
    
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
        lora_args: LoRAArguments,
        quant_args: QuantizationArguments
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.lora_args = lora_args
        self.quant_args = quant_args
        
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
    
    def load_tokenizer(self):
        """加载分词器"""
        print("🔄 加载分词器...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            use_fast=True,
            trust_remote_code=self.model_args.trust_remote_code,
            use_auth_token=self.model_args.use_auth_token
        )
        
        # 设置特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✅ 分词器加载完成: {len(self.tokenizer)} tokens")
    
    def load_model(self):
        """加载模型"""
        print("🔄 加载模型...")
        
        # 量化配置
        if self.quant_args.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.quant_args.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=self.quant_args.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.quant_args.bnb_4bit_quant_type
            )
        else:
            quantization_config = None
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=self.model_args.trust_remote_code,
            use_auth_token=self.model_args.use_auth_token,
            torch_dtype=torch.float16 if not quantization_config else None
        )
        
        # 准备量化训练
        if quantization_config:
            self.model = prepare_model_for_kbit_training(self.model)
        
        print(f"✅ 模型加载完成: {self.model_args.model_name_or_path}")
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 模型总参数量: {total_params:,}")
    
    def setup_lora(self):
        """设置LoRA配置"""
        print("🔄 配置LoRA...")
        
        # LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_args.lora_rank,
            lora_alpha=self.lora_args.lora_alpha,
            lora_dropout=self.lora_args.lora_dropout,
            target_modules=self.lora_args.lora_target_modules.split(","),
            bias=self.lora_args.lora_bias,
            use_rslora=self.lora_args.use_rslora
        )
        
        # 应用LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # 打印可训练参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"✅ LoRA配置完成")
        print(f"📊 可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"📊 总参数量: {total_params:,}")
    
    def load_dataset(self):
        """加载和预处理数据集"""
        print("🔄 加载数据集...")
        
        # 加载数据
        if self.data_args.dataset_path.endswith('.json'):
            with open(self.data_args.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = load_dataset(self.data_args.dataset_path)
            data = data['train']
        
        # 转换为Dataset格式
        if isinstance(data, list):
            dataset = Dataset.from_list(data)
        else:
            dataset = data
        
        # 数据预处理函数
        def preprocess_function(examples):
            model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
            
            for i in range(len(examples['instruction'])):
                instruction = examples['instruction'][i]
                input_text = examples.get('input', [''])[i]
                output_text = examples['output'][i]
                
                # 构建完整的输入
                if input_text:
                    full_prompt = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:\n{output_text}"
                else:
                    full_prompt = f"### 指令:\n{instruction}\n\n### 回答:\n{output_text}"
                
                # 分词
                tokenized = self.tokenizer(
                    full_prompt,
                    truncation=True,
                    padding=False,
                    max_length=2048,
                    return_tensors=None
                )
                
                model_inputs["input_ids"].append(tokenized["input_ids"])
                model_inputs["attention_mask"].append(tokenized["attention_mask"])
                model_inputs["labels"].append(tokenized["input_ids"].copy())
            
            return model_inputs
        
        # 应用预处理
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=dataset.column_names,
            desc="Preprocessing dataset"
        )
        
        # 分割训练集和验证集
        if self.data_args.validation_split_percentage > 0:
            split_dataset = processed_dataset.train_test_split(
                test_size=self.data_args.validation_split_percentage / 100
            )
            self.train_dataset = split_dataset['train']
            self.eval_dataset = split_dataset['test']
        else:
            self.train_dataset = processed_dataset
            self.eval_dataset = None
        
        # 限制样本数量
        if self.data_args.max_train_samples and len(self.train_dataset) > self.data_args.max_train_samples:
            self.train_dataset = self.train_dataset.select(range(self.data_args.max_train_samples))
        
        if self.eval_dataset and self.data_args.max_eval_samples and len(self.eval_dataset) > self.data_args.max_eval_samples:
            self.eval_dataset = self.eval_dataset.select(range(self.data_args.max_eval_samples))
        
        print(f"✅ 数据集加载完成")
        print(f"📊 训练样本: {len(self.train_dataset)}")
        if self.eval_dataset:
            print(f"📊 验证样本: {len(self.eval_dataset)}")
    
    def train(self):
        """开始训练"""
        print("🚀 开始LoRA微调...")
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # 训练前回调
        if self.training_args.do_train:
            # 开始训练
            train_result = trainer.train()
            
            # 保存模型
            trainer.save_model()
            
            # 保存训练结果
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()
        
        # 评估
        if self.training_args.do_eval and self.eval_dataset:
            print("📊 开始评估...")
            eval_result = trainer.evaluate()
            trainer.log_metrics("eval", eval_result)
            trainer.save_metrics("eval", eval_result)
        
        print("🎉 训练完成！")
    
    def save_lora_model(self, output_dir: str):
        """保存LoRA模型"""
        print(f"💾 保存LoRA模型到: {output_dir}")
        
        # 保存LoRA权重
        self.model.save_pretrained(output_dir)
        
        # 保存分词器
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存配置信息
        config = {
            "base_model": self.model_args.model_name_or_path,
            "lora_rank": self.lora_args.lora_rank,
            "lora_alpha": self.lora_args.lora_alpha,
            "lora_dropout": self.lora_args.lora_dropout,
            "target_modules": self.lora_args.lora_target_modules,
            "task_type": "CAUSAL_LM"
        }
        
        with open(f"{output_dir}/lora_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("✅ 模型保存完成")

def create_sample_data():
    """创建示例训练数据"""
    sample_data = [
        {
            "instruction": "解释什么是机器学习",
            "input": "",
            "output": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。通过算法和统计模型，机器学习系统可以从数据中识别模式，并基于这些模式做出预测或决策。"
        },
        {
            "instruction": "计算两个数的和",
            "input": "25 + 37",
            "output": "25 + 37 = 62"
        },
        {
            "instruction": "写一首关于春天的诗",
            "input": "",
            "output": "春风轻抚绿柳条，\n花开满树鸟儿叫。\n蝴蝶飞舞草青青，\n生机盎然春意闹。"
        }
    ]
    
    os.makedirs("./data", exist_ok=True)
    with open("./data/sample_data.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print("📄 示例数据已创建: ./data/sample_data.json")

def main():
    parser = argparse.ArgumentParser(description="LoRA微调脚本")
    
    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, default="baichuan-inc/Baichuan2-7B-Chat")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    
    # 数据参数
    parser.add_argument("--dataset_path", type=str, default="./data/sample_data.json")
    parser.add_argument("--validation_split_percentage", type=int, default=5)
    parser.add_argument("--max_train_samples", type=int, default=None)
    
    # LoRA参数
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj")
    
    # 量化参数
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16")
    
    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./lora_output")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--do_train", action="store_true", default=True)
    parser.add_argument("--do_eval", action="store_true", default=True)
    parser.add_argument("--create_sample_data", action="store_true", help="创建示例数据")
    
    args = parser.parse_args()
    
    # 创建示例数据
    if args.create_sample_data:
        create_sample_data()
        return
    
    # 创建参数对象
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code
    )
    
    data_args = DataArguments(
        dataset_path=args.dataset_path,
        validation_split_percentage=args.validation_split_percentage,
        max_train_samples=args.max_train_samples
    )
    
    lora_args = LoRAArguments(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules
    )
    
    quant_args = QuantizationArguments(
        use_4bit=args.use_4bit,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype
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
        eval_steps=args.eval_steps,
        fp16=args.fp16,
        do_train=args.do_train,
        do_eval=args.do_eval,
        evaluation_strategy="steps" if args.do_eval else "no",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none"
    )
    
    # 创建训练器
    trainer = LoRATrainer(model_args, data_args, training_args, lora_args, quant_args)
    
    # 加载组件
    trainer.load_tokenizer()
    trainer.load_model()
    trainer.setup_lora()
    trainer.load_dataset()
    
    # 开始训练
    trainer.train()
    
    # 保存LoRA模型
    trainer.save_lora_model(args.output_dir)

if __name__ == "__main__":
    main() 