#!/usr/bin/env python3
"""
DCU-in-Action 训练脚本
提供完整的模型训练功能，支持分布式训练、混合精度、检查点管理等
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as e:
    print(f"请安装PyTorch: {e}")
    sys.exit(1)

try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoConfig,
        DataCollatorWithPadding,
        default_data_collator
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from common.utils.logger import setup_global_logging, get_logger
from common.llm.training_utils import (
    TrainingConfig, TrainingLoop,
    create_distributed_dataloader,
    get_preset_config
)
from common.dcu.device_manager import DCUDeviceManager

logger = get_logger(__name__)


class SimpleDataset(Dataset):
    """简单的演示数据集"""
    
    def __init__(self, size: int = 1000, seq_length: int = 512):
        self.size = size
        self.seq_length = seq_length
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 生成随机数据用于演示
        return {
            'input_ids': torch.randint(0, 1000, (self.seq_length,)),
            'attention_mask': torch.ones(self.seq_length),
            'labels': torch.randint(0, 1000, (self.seq_length,))
        }


class SimpleModel(nn.Module):
    """简单的演示模型"""
    
    def __init__(self, vocab_size: int = 1000, hidden_size: int = 768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                batch_first=True
            ),
            num_layers=6
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # 嵌入
        hidden_states = self.embedding(input_ids)
        
        # Transformer
        if attention_mask is not None:
            # 转换attention_mask格式
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        hidden_states = self.transformer(
            hidden_states,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # 语言模型头
        logits = self.lm_head(hidden_states)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # 返回类似transformers的输出
        class ModelOutput:
            def __init__(self, loss, logits):
                self.loss = loss
                self.logits = logits
        
        return ModelOutput(loss, logits)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"配置文件加载成功: {config_path}")
        return config
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        raise


def create_training_config(config_dict: Dict[str, Any], args: argparse.Namespace) -> TrainingConfig:
    """创建训练配置"""
    
    # 从配置文件提取参数
    model_config = config_dict.get('model', {})
    training_config = config_dict.get('training', {})
    data_config = config_dict.get('data', {})
    mixed_precision_config = config_dict.get('mixed_precision', {})
    distributed_config = config_dict.get('distributed', {})
    output_config = config_dict.get('output', {})
    
    # 创建训练配置
    training_cfg = TrainingConfig(
        # 模型配置
        model_name_or_path=args.model_name_or_path or model_config.get('name_or_path', ''),
        model_type=model_config.get('model_type', 'auto'),
        
        # 训练参数
        batch_size=args.batch_size or training_config.get('batch_size', 8),
        learning_rate=args.learning_rate or training_config.get('learning_rate', 5e-5),
        num_epochs=args.num_epochs or training_config.get('num_epochs', 3),
        warmup_ratio=training_config.get('warmup_ratio', 0.1),
        weight_decay=training_config.get('weight_decay', 0.01),
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        
        # 优化器配置
        optimizer_type=training_config.get('optimizer_type', 'adamw'),
        scheduler_type=training_config.get('scheduler_type', 'linear'),
        
        # 混合精度
        fp16=args.fp16 or mixed_precision_config.get('fp16', False),
        bf16=args.bf16 or mixed_precision_config.get('bf16', False),
        gradient_checkpointing=mixed_precision_config.get('gradient_checkpointing', False),
        
        # 分布式训练
        distributed=distributed_config.get('enabled', False),
        local_rank=args.local_rank,
        
        # 数据配置
        max_seq_length=data_config.get('max_seq_length', 512),
        dataloader_num_workers=data_config.get('dataloader_num_workers', 4),
        dataloader_pin_memory=data_config.get('pin_memory', True),
        
        # 输出配置
        output_dir=args.output_dir or output_config.get('output_dir', './output'),
        save_steps=output_config.get('save_steps', 500),
        logging_steps=output_config.get('logging_steps', 10),
        eval_steps=output_config.get('eval_steps', 500),
        save_total_limit=output_config.get('save_total_limit', 3),
        
        # 其他
        seed=config_dict.get('seed', 42),
        resume_from_checkpoint=args.resume_from_checkpoint,
        gradient_accumulation_steps=args.gradient_accumulation_steps or 1
    )
    
    return training_cfg


def setup_model_and_tokenizer(config: TrainingConfig):
    """设置模型和分词器"""
    
    if config.model_name_or_path and HAS_TRANSFORMERS:
        try:
            # 使用Transformers模型
            model = AutoModel.from_pretrained(config.model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
            
            # 添加语言模型头
            if not hasattr(model, 'lm_head'):
                model.lm_head = nn.Linear(model.config.hidden_size, model.config.vocab_size)
            
            logger.info(f"模型加载成功: {config.model_name_or_path}")
            return model, tokenizer
            
        except Exception as e:
            logger.warning(f"Transformers模型加载失败: {e}，使用简单模型")
    
    # 使用简单演示模型
    model = SimpleModel()
    tokenizer = None
    logger.info("使用简单演示模型")
    
    return model, tokenizer


def create_datasets(config: TrainingConfig, tokenizer=None):
    """创建数据集"""
    
    # 演示数据集
    train_dataset = SimpleDataset(
        size=10000, 
        seq_length=config.max_seq_length
    )
    
    eval_dataset = SimpleDataset(
        size=1000,
        seq_length=config.max_seq_length
    )
    
    logger.info(f"训练数据集大小: {len(train_dataset)}")
    logger.info(f"验证数据集大小: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DCU-in-Action 训练脚本")
    
    # 基础参数
    parser.add_argument("--config", type=str, default="configs/training/default.yaml",
                       help="配置文件路径")
    parser.add_argument("--model_name_or_path", type=str, default=None,
                       help="模型路径或名称")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=None,
                       help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="学习率")
    parser.add_argument("--num_epochs", type=int, default=None,
                       help="训练轮数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                       help="梯度累积步数")
    
    # 混合精度
    parser.add_argument("--fp16", action="store_true",
                       help="启用FP16混合精度")
    parser.add_argument("--bf16", action="store_true",
                       help="启用BF16混合精度")
    
    # 分布式训练
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="本地rank（分布式训练）")
    
    # 其他
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="从检查点恢复")
    parser.add_argument("--preset", type=str, default=None,
                       help="预设配置名称")
    parser.add_argument("--debug", action="store_true",
                       help="调试模式")
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = "DEBUG" if args.debug else "INFO"
    setup_global_logging(level=log_level)
    
    try:
        # DCU设备检查
        device_manager = DCUDeviceManager()
        device_info = device_manager.get_device_info()
        logger.info(f"DCU设备信息: {device_info}")
        
        # 加载配置
        if args.preset:
            # 使用预设配置
            logger.info(f"使用预设配置: {args.preset}")
            config = get_preset_config(args.preset)
            
            # 应用命令行覆盖
            if args.output_dir:
                config.output_dir = args.output_dir
            if args.batch_size:
                config.batch_size = args.batch_size
            if args.learning_rate:
                config.learning_rate = args.learning_rate
            if args.num_epochs:
                config.num_epochs = args.num_epochs
            if args.fp16:
                config.fp16 = True
            if args.bf16:
                config.bf16 = True
            if args.resume_from_checkpoint:
                config.resume_from_checkpoint = args.resume_from_checkpoint
                
        else:
            # 从配置文件加载
            config_dict = load_config(args.config)
            config = create_training_config(config_dict, args)
        
        logger.info(f"训练配置: {config}")
        
        # 设置模型
        model, tokenizer = setup_model_and_tokenizer(config)
        
        # 创建数据集
        train_dataset, eval_dataset = create_datasets(config, tokenizer)
        
        # 创建数据加载器
        train_dataloader = create_distributed_dataloader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.dataloader_num_workers,
            pin_memory=config.dataloader_pin_memory
        )
        
        eval_dataloader = create_distributed_dataloader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.dataloader_num_workers,
            pin_memory=config.dataloader_pin_memory
        )
        
        # 创建训练循环
        training_loop = TrainingLoop(config)
        
        logger.info("开始训练...")
        
        # 开始训练
        training_loop.train(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader
        )
        
        logger.info("训练完成！")
        
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 