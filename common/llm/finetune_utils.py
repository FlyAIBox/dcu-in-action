"""
大模型微调工具库
提供LoRA、QLoRA、指令微调、RLHF、参数高效微调等功能
"""

import os
import json
import time
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import numpy as np
from collections import defaultdict
import math
import shutil

try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoConfig,
        AutoModelForCausalLM, 
        TrainingArguments, Trainer,
        DataCollatorForSeq2Seq,
        DataCollatorForLanguageModeling,
        get_scheduler,
        BitsAndBytesConfig
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import (
        LoraConfig, get_peft_model, get_peft_model_state_dict,
        prepare_model_for_kbit_training, TaskType,
        PeftModel, PeftConfig
    )
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    from datasets import Dataset, DatasetDict, load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

try:
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    HAS_TRL = True
except ImportError:
    HAS_TRL = False

from ..utils.logger import get_logger, performance_monitor
from ..utils.monitor import SystemMonitor
from ..dcu.device_manager import DCUManager

logger = get_logger(__name__)


@dataclass
class LoRAConfig:
    """LoRA配置类"""
    # 基础配置
    r: int = 16  # LoRA秩
    lora_alpha: int = 32  # LoRA alpha参数
    target_modules: Optional[List[str]] = None  # 目标模块
    lora_dropout: float = 0.1  # Dropout概率
    bias: str = "none"  # 偏置设置 ("none", "all", "lora_only")
    
    # 任务类型
    task_type: str = "CAUSAL_LM"  # 任务类型
    
    # 高级配置
    fan_in_fan_out: bool = False
    enable_lora: List[bool] = None
    merge_weights: bool = False
    
    def to_peft_config(self) -> "LoraConfig":
        """转换为PEFT LoraConfig"""
        if not HAS_PEFT:
            raise ImportError("需要安装peft库: pip install peft")
        
        from peft import LoraConfig as PeftLoraConfig, TaskType
        
        # 转换任务类型
        task_type = getattr(TaskType, self.task_type)
        
        return PeftLoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=task_type,
            fan_in_fan_out=self.fan_in_fan_out,
        )


@dataclass
class FinetuneConfig:
    """微调配置类"""
    # 模型配置
    model_name_or_path: str
    tokenizer_path: Optional[str] = None
    trust_remote_code: bool = True
    
    # LoRA配置
    use_lora: bool = True
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)
    
    # 量化配置
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    
    # 训练配置
    output_dir: str = "./output"
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: float = 3.0
    max_steps: int = -1
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    
    # 优化器配置
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # 混合精度
    fp16: bool = False
    bf16: bool = True
    
    # 保存和日志
    save_strategy: str = "epoch"
    save_steps: int = 500
    save_total_limit: int = 3
    logging_strategy: str = "steps"
    logging_steps: int = 10
    eval_strategy: str = "epoch"
    eval_steps: int = 500
    
    # 数据处理
    max_seq_length: int = 2048
    preprocessing_num_workers: int = 4
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    
    # 特殊配置
    gradient_checkpointing: bool = True
    dataloader_pin_memory: bool = True
    group_by_length: bool = True
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    # SFT配置
    max_prompt_length: int = 512
    max_completion_length: int = 1536
    
    def to_training_arguments(self) -> "TrainingArguments":
        """转换为TrainingArguments"""
        if not HAS_TRANSFORMERS:
            raise ImportError("需要安装transformers库")
        
        return TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            optim=self.optim,
            lr_scheduler_type=self.lr_scheduler_type,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            adam_epsilon=self.adam_epsilon,
            max_grad_norm=self.max_grad_norm,
            fp16=self.fp16,
            bf16=self.bf16,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            logging_strategy=self.logging_strategy,
            logging_steps=self.logging_steps,
            evaluation_strategy=self.eval_strategy,
            eval_steps=self.eval_steps,
            gradient_checkpointing=self.gradient_checkpointing,
            dataloader_pin_memory=self.dataloader_pin_memory,
            group_by_length=self.group_by_length,
            report_to=self.report_to,
            remove_unused_columns=self.remove_unused_columns,
            dataloader_num_workers=self.dataloader_num_workers,
        )


class ModelPreparer:
    """模型准备器"""
    
    def __init__(self, config: FinetuneConfig):
        self.config = config
        self.dcu_manager = DCUManager()
        self.monitor = SystemMonitor()
    
    @performance_monitor
    def prepare_model_and_tokenizer(self) -> Tuple[nn.Module, Any]:
        """准备模型和分词器"""
        logger.info(f"开始准备模型: {self.config.model_name_or_path}")
        
        # 加载分词器
        tokenizer_path = self.config.tokenizer_path or self.config.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right"  # 训练时使用右侧填充
        )
        
        # 设置特殊token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 量化配置
        quantization_config = None
        if self.config.load_in_4bit or self.config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            )
        
        # 加载模型
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "quantization_config": quantization_config,
        }
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            **model_kwargs
        )
        
        # 准备量化模型
        if quantization_config is not None:
            if not HAS_PEFT:
                raise ImportError("量化训练需要安装peft库: pip install peft")
            model = prepare_model_for_kbit_training(model)
        
        # 应用LoRA
        if self.config.use_lora:
            if not HAS_PEFT:
                raise ImportError("LoRA训练需要安装peft库: pip install peft")
            
            peft_config = self.config.lora_config.to_peft_config()
            
            # 自动确定目标模块
            if peft_config.target_modules is None:
                target_modules = self._find_target_modules(model)
                peft_config.target_modules = target_modules
                logger.info(f"自动检测到的目标模块: {target_modules}")
            
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        
        # 启用梯度检查点
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        logger.info(f"模型准备完成: {type(model).__name__}")
        return model, tokenizer
    
    def _find_target_modules(self, model: nn.Module) -> List[str]:
        """自动查找目标模块"""
        target_modules = set()
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # 排除输出层
                if "lm_head" in name or "output" in name:
                    continue
                
                # 提取模块名称的最后一部分
                module_name = name.split('.')[-1]
                target_modules.add(module_name)
        
        # 常见的目标模块
        common_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        target_modules = [m for m in common_targets if m in target_modules]
        
        if not target_modules:
            # 如果没有找到常见模块，返回所有线性层
            target_modules = list(set([name.split('.')[-1] for name, module in model.named_modules() 
                                     if isinstance(module, nn.Linear) and "lm_head" not in name]))
        
        return target_modules


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, tokenizer, config: FinetuneConfig):
        self.tokenizer = tokenizer
        self.config = config
    
    def process_instruction_dataset(self, 
                                   dataset: Union[Dataset, List[Dict]], 
                                   instruction_template: str = "### 指令:\n{instruction}\n\n### 回答:\n{output}") -> Dataset:
        """处理指令微调数据集"""
        if not HAS_DATASETS:
            raise ImportError("需要安装datasets库: pip install datasets")
        
        if isinstance(dataset, list):
            dataset = Dataset.from_list(dataset)
        
        def format_instruction(example):
            """格式化指令"""
            if "instruction" in example and "output" in example:
                text = instruction_template.format(
                    instruction=example["instruction"],
                    output=example["output"]
                )
            elif "prompt" in example and "response" in example:
                text = instruction_template.format(
                    instruction=example["prompt"],
                    output=example["response"]
                )
            elif "question" in example and "answer" in example:
                text = instruction_template.format(
                    instruction=example["question"],
                    output=example["answer"]
                )
            else:
                raise ValueError("数据集必须包含指令和回答字段")
            
            return {"text": text}
        
        # 格式化数据
        formatted_dataset = dataset.map(
            format_instruction,
            num_proc=self.config.preprocessing_num_workers,
            remove_columns=dataset.column_names
        )
        
        return formatted_dataset
    
    def process_chat_dataset(self, 
                            dataset: Union[Dataset, List[Dict]],
                            system_message: str = "你是一个有用的AI助手。") -> Dataset:
        """处理对话数据集"""
        if not HAS_DATASETS:
            raise ImportError("需要安装datasets库: pip install datasets")
        
        if isinstance(dataset, list):
            dataset = Dataset.from_list(dataset)
        
        def format_chat(example):
            """格式化对话"""
            messages = example.get("messages", example.get("conversations", []))
            
            if not messages:
                raise ValueError("数据集必须包含messages或conversations字段")
            
            text = f"System: {system_message}\n"
            
            for message in messages:
                role = message.get("role", message.get("from", "user"))
                content = message.get("content", message.get("value", ""))
                
                if role in ["user", "human"]:
                    text += f"User: {content}\n"
                elif role in ["assistant", "gpt", "bot"]:
                    text += f"Assistant: {content}\n"
            
            return {"text": text}
        
        # 格式化数据
        formatted_dataset = dataset.map(
            format_chat,
            num_proc=self.config.preprocessing_num_workers,
            remove_columns=dataset.column_names
        )
        
        return formatted_dataset
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """分词数据集"""
        def tokenize_function(examples):
            # 分词
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.config.max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )
            
            # 设置labels
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # 应用分词
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.config.preprocessing_num_workers,
            remove_columns=dataset.column_names,
        )
        
        return tokenized_dataset
    
    def create_sft_dataset(self, 
                          dataset: Union[Dataset, List[Dict]],
                          prompt_template: str = "### 指令:\n{instruction}\n\n### 回答:\n") -> Dataset:
        """创建SFT（监督微调）数据集"""
        if not HAS_DATASETS:
            raise ImportError("需要安装datasets库: pip install datasets")
        
        if isinstance(dataset, list):
            dataset = Dataset.from_list(dataset)
        
        def format_sft(example):
            """格式化SFT数据"""
            instruction = example.get("instruction", example.get("prompt", example.get("question", "")))
            output = example.get("output", example.get("response", example.get("answer", "")))
            
            prompt = prompt_template.format(instruction=instruction)
            
            return {
                "prompt": prompt,
                "completion": output,
                "text": prompt + output
            }
        
        # 格式化数据
        formatted_dataset = dataset.map(
            format_sft,
            num_proc=self.config.preprocessing_num_workers,
        )
        
        return formatted_dataset


class FineTuner:
    """微调器"""
    
    def __init__(self, config: FinetuneConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dcu_manager = DCUManager()
        self.monitor = SystemMonitor()
    
    def setup(self):
        """设置微调环境"""
        # 准备模型和分词器
        preparer = ModelPreparer(self.config)
        self.model, self.tokenizer = preparer.prepare_model_and_tokenizer()
        
        logger.info("微调器设置完成")
    
    @performance_monitor
    def finetune_with_trainer(self, 
                             train_dataset: Dataset,
                             eval_dataset: Optional[Dataset] = None,
                             data_collator: Optional[Callable] = None) -> None:
        """使用Trainer进行微调"""
        if not HAS_TRANSFORMERS:
            raise ImportError("需要安装transformers库")
        
        # 创建训练参数
        training_args = self.config.to_training_arguments()
        
        # 创建数据整理器
        if data_collator is None:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # 因果语言模型不使用MLM
                pad_to_multiple_of=8,
            )
        
        # 创建训练器
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # 开始训练
        logger.info("开始微调训练...")
        train_result = self.trainer.train()
        
        # 保存模型
        self.trainer.save_model()
        self.trainer.save_state()
        
        # 保存训练结果
        with open(os.path.join(self.config.output_dir, "train_results.json"), "w") as f:
            json.dump(train_result.metrics, f, indent=2)
        
        logger.info(f"微调完成，模型已保存到: {self.config.output_dir}")
        
        return train_result
    
    @performance_monitor
    def finetune_with_sft(self, 
                         train_dataset: Dataset,
                         eval_dataset: Optional[Dataset] = None) -> None:
        """使用SFTTrainer进行监督微调"""
        if not HAS_TRL:
            raise ImportError("需要安装trl库: pip install trl")
        
        # 创建训练参数
        training_args = self.config.to_training_arguments()
        
        # 创建SFT训练器
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
            dataset_text_field="text",
            packing=False,
        )
        
        # 开始训练
        logger.info("开始SFT训练...")
        train_result = self.trainer.train()
        
        # 保存模型
        self.trainer.save_model()
        self.trainer.save_state()
        
        # 保存训练结果
        with open(os.path.join(self.config.output_dir, "train_results.json"), "w") as f:
            json.dump(train_result.metrics, f, indent=2)
        
        logger.info(f"SFT训练完成，模型已保存到: {self.config.output_dir}")
        
        return train_result
    
    def save_lora_weights(self, output_dir: str):
        """保存LoRA权重"""
        if not self.config.use_lora:
            logger.warning("当前配置未使用LoRA，无法保存LoRA权重")
            return
        
        if not HAS_PEFT:
            raise ImportError("需要安装peft库: pip install peft")
        
        # 保存LoRA权重
        lora_state_dict = get_peft_model_state_dict(self.model)
        torch.save(lora_state_dict, os.path.join(output_dir, "lora_weights.bin"))
        
        # 保存配置
        self.model.peft_config["default"].save_pretrained(output_dir)
        
        logger.info(f"LoRA权重已保存到: {output_dir}")
    
    def merge_and_save_model(self, output_dir: str):
        """合并LoRA权重并保存完整模型"""
        if not self.config.use_lora:
            logger.warning("当前配置未使用LoRA，直接保存模型")
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            return
        
        if not HAS_PEFT:
            raise ImportError("需要安装peft库: pip install peft")
        
        # 合并LoRA权重
        merged_model = self.model.merge_and_unload()
        
        # 保存合并后的模型
        merged_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"合并后的模型已保存到: {output_dir}")
    
    def evaluate_model(self, eval_dataset: Dataset) -> Dict[str, float]:
        """评估模型"""
        if self.trainer is None:
            raise ValueError("训练器未初始化，请先调用setup()方法")
        
        logger.info("开始评估模型...")
        eval_result = self.trainer.evaluate(eval_dataset=eval_dataset)
        
        logger.info(f"评估结果: {eval_result}")
        return eval_result


class LoRAMerger:
    """LoRA权重合并器"""
    
    @staticmethod
    def load_and_merge_lora(base_model_path: str, 
                           lora_path: str, 
                           output_path: str,
                           torch_dtype: torch.dtype = torch.float16):
        """加载并合并LoRA权重"""
        if not HAS_PEFT:
            raise ImportError("需要安装peft库: pip install peft")
        
        logger.info(f"加载基础模型: {base_model_path}")
        
        # 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载LoRA配置和权重
        logger.info(f"加载LoRA权重: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        
        # 合并权重
        logger.info("合并LoRA权重...")
        merged_model = model.merge_and_unload()
        
        # 保存合并后的模型
        logger.info(f"保存合并后的模型: {output_path}")
        merged_model.save_pretrained(output_path, save_function=torch.save)
        
        # 保存分词器
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
        
        logger.info("LoRA权重合并完成")


# 工具函数
def create_finetune_config(model_path: str, **kwargs) -> FinetuneConfig:
    """创建微调配置的便捷函数"""
    return FinetuneConfig(
        model_name_or_path=model_path,
        **kwargs
    )


def quick_finetune(model_path: str, 
                  train_data: List[Dict],
                  output_dir: str = "./finetune_output",
                  **config_kwargs) -> FineTuner:
    """快速微调函数"""
    # 创建配置
    config = create_finetune_config(
        model_path=model_path,
        output_dir=output_dir,
        **config_kwargs
    )
    
    # 创建微调器
    finetuner = FineTuner(config)
    finetuner.setup()
    
    # 处理数据
    processor = DataProcessor(finetuner.tokenizer, config)
    train_dataset = processor.process_instruction_dataset(train_data)
    train_dataset = processor.tokenize_dataset(train_dataset)
    
    # 开始微调
    finetuner.finetune_with_trainer(train_dataset)
    
    return finetuner


# 预设配置
FINETUNE_PRESETS = {
    "lora_7b": FinetuneConfig(
        model_name_or_path="",
        use_lora=True,
        lora_config=LoRAConfig(r=16, lora_alpha=32),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        load_in_4bit=True,
    ),
    "lora_13b": FinetuneConfig(
        model_name_or_path="",
        use_lora=True,
        lora_config=LoRAConfig(r=8, lora_alpha=16),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        num_train_epochs=3,
        load_in_4bit=True,
    ),
    "full_finetune": FinetuneConfig(
        model_name_or_path="",
        use_lora=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        num_train_epochs=1,
        fp16=True,
    ),
}


def get_finetune_preset(preset_name: str, model_path: str) -> FinetuneConfig:
    """获取预设配置"""
    if preset_name not in FINETUNE_PRESETS:
        raise ValueError(f"未知预设: {preset_name}，可用预设: {list(FINETUNE_PRESETS.keys())}")
    
    config = FINETUNE_PRESETS[preset_name]
    config.model_name_or_path = model_path
    
    return config 