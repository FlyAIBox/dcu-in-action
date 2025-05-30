#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLaMA Factory模型训练脚本
作者: DCU实战项目组
"""

import os
import sys
import yaml
import json
import torch
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

# 添加LLaMA Factory路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../LLaMA-Factory/src'))

try:
    from llamafactory.train.tuner import run_exp
    from llamafactory.extras.logging import get_logger
except ImportError:
    print("错误：无法导入LLaMA Factory模块")
    print("请确保已正确安装LLaMA Factory并在正确的环境中运行")
    sys.exit(1)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """模型训练管理器"""
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
            **kwargs: 直接传入的配置参数
        """
        self.config = self._load_config(config_path, **kwargs)
        self.start_time = None
        self.training_process = None
        
    def _load_config(self, config_path: Optional[str], **kwargs) -> Dict[str, Any]:
        """加载配置"""
        config = {}
        
        # 从配置文件加载
        if config_path and os.path.exists(config_path):
            logger.info(f"加载配置文件: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
        
        # 命令行参数覆盖配置文件
        config.update({k: v for k, v in kwargs.items() if v is not None})
        
        # 设置默认值
        default_config = {
            "model_name_or_path": "THUDM/chatglm3-6b",
            "dataset": "alpaca_zh",
            "template": "chatglm3",
            "finetuning_type": "lora",
            "output_dir": "./saves/default_train",
            "stage": "sft",
            "do_train": True,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "save_steps": 1000,
            "learning_rate": 5e-5,
            "num_train_epochs": 3,
            "max_grad_norm": 1.0,
            "val_size": 0.1,
            "evaluation_strategy": "steps",
            "eval_steps": 500,
            "load_best_model_at_end": True,
            "plot_loss": True
        }
        
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
        
        return config
    
    def validate_config(self) -> bool:
        """验证配置参数"""
        logger.info("验证训练配置...")
        
        required_fields = ["model_name_or_path", "dataset", "output_dir"]
        for field in required_fields:
            if field not in self.config:
                logger.error(f"缺少必要配置: {field}")
                return False
        
        # 检查输出目录
        output_dir = Path(self.config["output_dir"])
        if output_dir.exists() and any(output_dir.iterdir()):
            logger.warning(f"输出目录非空: {output_dir}")
            response = input("继续训练将覆盖现有文件，是否继续？[y/N]: ")
            if response.lower() != 'y':
                logger.info("训练已取消")
                return False
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查GPU可用性
        try:
            if torch.cuda.is_available():
                logger.info(f"检测到 {torch.cuda.device_count()} 个GPU")
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                logger.warning("未检测到可用GPU，将使用CPU训练（速度较慢）")
        except ImportError:
            logger.warning("PyTorch未安装")
            return False
        
        logger.info("配置验证通过")
        return True
    
    def prepare_training_args(self) -> list:
        """准备训练参数"""
        args = [
            "--stage", str(self.config["stage"]),
            "--model_name_or_path", str(self.config["model_name_or_path"]),
            "--dataset", str(self.config["dataset"]),
            "--template", str(self.config.get("template", "default")),
            "--finetuning_type", str(self.config["finetuning_type"]),
            "--output_dir", str(self.config["output_dir"]),
            "--overwrite_output_dir",
            "--per_device_train_batch_size", str(self.config["per_device_train_batch_size"]),
            "--gradient_accumulation_steps", str(self.config["gradient_accumulation_steps"]),
            "--lr_scheduler_type", str(self.config["lr_scheduler_type"]),
            "--logging_steps", str(self.config["logging_steps"]),
            "--save_steps", str(self.config["save_steps"]),
            "--learning_rate", str(self.config["learning_rate"]),
            "--num_train_epochs", str(self.config["num_train_epochs"]),
            "--max_grad_norm", str(self.config["max_grad_norm"]),
            "--val_size", str(self.config["val_size"]),
            "--evaluation_strategy", str(self.config["evaluation_strategy"]),
            "--eval_steps", str(self.config["eval_steps"]),
        ]
        
        # 可选参数
        optional_bool_args = [
            "do_train", "load_best_model_at_end", "plot_loss", 
            "use_unsloth", "use_rslora"
        ]
        
        for arg in optional_bool_args:
            if self.config.get(arg, False):
                args.append(f"--{arg}")
        
        # LoRA特定参数
        if self.config["finetuning_type"] == "lora":
            lora_params = [
                "lora_target", "lora_rank", "lora_alpha", "lora_dropout",
                "modules_to_save", "loraplus_lr_ratio"
            ]
            
            for param in lora_params:
                if param in self.config:
                    args.extend([f"--{param}", str(self.config[param])])
        
        # 量化参数
        if "quantization_bit" in self.config:
            args.extend(["--quantization_bit", str(self.config["quantization_bit"])])
        
        # 其他可选参数
        other_params = [
            "max_samples", "cutoff_len", "warmup_ratio", "weight_decay",
            "adam_beta1", "adam_beta2", "adam_epsilon", "max_source_length",
            "max_target_length", "report_to"
        ]
        
        for param in other_params:
            if param in self.config:
                args.extend([f"--{param}", str(self.config[param])])
        
        return args
    
    def save_config(self):
        """保存训练配置"""
        config_file = os.path.join(self.config["output_dir"], "training_config.yaml")
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"训练配置已保存: {config_file}")
    
    def start_training(self) -> bool:
        """开始训练"""
        if not self.validate_config():
            return False
        
        self.save_config()
        
        # 检查LLaMA Factory是否安装
        try:
            import llamafactory
            logger.info(f"LLaMA Factory版本: {llamafactory.__version__}")
        except ImportError:
            logger.error("LLaMA Factory未安装，请先运行安装脚本")
            return False
        
        args = self.prepare_training_args()
        
        logger.info("🚀 开始模型训练...")
        logger.info(f"训练参数: {' '.join(args)}")
        
        self.start_time = time.time()
        
        try:
            # 导入并运行训练
            from llamafactory.train.tuner import run_exp
            
            # 捕获训练过程
            logger.info("正在启动训练进程...")
            run_exp(args)
            
            training_time = time.time() - self.start_time
            logger.info(f"✅ 训练完成！耗时: {training_time/3600:.2f} 小时")
            
            # 生成训练报告
            self._generate_training_report(training_time)
            
            return True
            
        except KeyboardInterrupt:
            logger.warning("训练被用户中断")
            return False
        except Exception as e:
            logger.error(f"❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_training_report(self, training_time: float):
        """生成训练报告"""
        report_file = os.path.join(self.config["output_dir"], "training_report.md")
        
        report_content = f"""
# 训练报告

## 基本信息
- **训练开始时间**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}
- **训练结束时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **训练耗时**: {training_time/3600:.2f} 小时
- **模型**: {self.config['model_name_or_path']}
- **数据集**: {self.config['dataset']}
- **微调类型**: {self.config['finetuning_type']}

## 训练配置
```yaml
{yaml.dump(self.config, default_flow_style=False, allow_unicode=True)}
```

## 输出文件
- 模型权重: `{self.config['output_dir']}/adapter_model.bin`
- 配置文件: `{self.config['output_dir']}/adapter_config.json`
- 训练日志: `{self.config['output_dir']}/trainer_log.jsonl`
- 损失曲线: `{self.config['output_dir']}/training_loss.png`

## 后续步骤
1. 模型评估：使用测试集评估模型性能
2. 模型合并：将LoRA权重合并到基础模型
3. 模型部署：部署到推理环境

## 使用示例
```bash
# 模型推理
llamafactory-cli chat \\
    --model_name_or_path {self.config['model_name_or_path']} \\
    --adapter_name_or_path {self.config['output_dir']} \\
    --template {self.config.get('template', 'default')}

# 模型合并
llamafactory-cli export \\
    --model_name_or_path {self.config['model_name_or_path']} \\
    --adapter_name_or_path {self.config['output_dir']} \\
    --template {self.config.get('template', 'default')} \\
    --finetuning_type {self.config['finetuning_type']} \\
    --export_dir ./merged_model
```
        """
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"训练报告已生成: {report_file}")

def create_sample_config():
    """创建示例配置文件"""
    config = {
        "model_name_or_path": "THUDM/chatglm3-6b",
        "dataset": "alpaca_zh",
        "template": "chatglm3",
        "finetuning_type": "lora",
        "output_dir": "./saves/ChatGLM3-6B/lora/sample_train",
        
        # LoRA配置
        "lora_target": "query_key_value",
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "modules_to_save": "embed_tokens,lm_head",
        
        # 训练参数
        "stage": "sft",
        "do_train": True,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "save_steps": 1000,
        "learning_rate": 5e-5,
        "num_train_epochs": 3,
        "max_grad_norm": 1.0,
        "quantization_bit": 4,
        "use_unsloth": True,
        "val_size": 0.1,
        "evaluation_strategy": "steps",
        "eval_steps": 500,
        "load_best_model_at_end": True,
        "plot_loss": True,
        "report_to": "none"
    }
    
    config_file = "sample_training_config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"示例配置文件已创建: {config_file}")
    return config_file

def main():
    parser = argparse.ArgumentParser(description="LLaMA Factory 模型训练工具")
    
    # 配置文件
    parser.add_argument("--config", help="训练配置文件路径 (YAML/JSON)")
    
    # 基本参数
    parser.add_argument("--model", dest="model_name_or_path", help="模型名称或路径")
    parser.add_argument("--dataset", help="数据集名称")
    parser.add_argument("--template", help="对话模板")
    parser.add_argument("--output_dir", help="输出目录")
    parser.add_argument("--finetuning_type", choices=["lora", "qlora", "full"], help="微调类型")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, dest="per_device_train_batch_size", help="批次大小")
    parser.add_argument("--learning_rate", type=float, help="学习率")
    parser.add_argument("--num_epochs", type=int, dest="num_train_epochs", help="训练轮数")
    parser.add_argument("--max_samples", type=int, help="最大样本数")
    
    # LoRA参数
    parser.add_argument("--lora_rank", type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, help="LoRA dropout")
    parser.add_argument("--lora_target", help="LoRA目标模块")
    
    # 其他选项
    parser.add_argument("--quantization_bit", type=int, choices=[4, 8], help="量化位数")
    parser.add_argument("--create_config", action="store_true", help="创建示例配置文件")
    parser.add_argument("--dry_run", action="store_true", help="仅验证配置，不执行训练")
    
    args = parser.parse_args()
    
    # 创建示例配置
    if args.create_config:
        create_sample_config()
        return
    
    # 转换参数为字典
    config_args = {k: v for k, v in vars(args).items() 
                   if v is not None and k not in ['config', 'create_config', 'dry_run']}
    
    # 创建训练器
    trainer = ModelTrainer(args.config, **config_args)
    
    # 打印配置信息
    logger.info("训练配置:")
    for key, value in trainer.config.items():
        logger.info(f"  {key}: {value}")
    
    # 仅验证配置
    if args.dry_run:
        if trainer.validate_config():
            logger.info("✅ 配置验证通过")
        else:
            logger.error("❌ 配置验证失败")
        return
    
    # 开始训练
    success = trainer.start_training()
    
    if success:
        logger.info("🎉 训练任务完成！")
        sys.exit(0)
    else:
        logger.error("❌ 训练任务失败！")
        sys.exit(1)

if __name__ == "__main__":
    main() 