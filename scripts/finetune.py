#!/usr/bin/env python3
"""
DCU-in-Action 模型微调脚本
提供LoRA微调、全参数微调、指令微调、SFT等功能
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    from datasets import Dataset
except ImportError as e:
    print(f"请安装必要的依赖: {e}")
    sys.exit(1)

from common.utils.logger import setup_global_logging, get_logger
from common.llm.finetune_utils import (
    LoRAConfig,
    FinetuneConfig,
    FineTuner,
    DataProcessor,
    LoRAMerger,
    create_finetune_config,
    quick_finetune,
    get_finetune_preset
)
from common.dcu.device_manager import DCUDeviceManager

logger = get_logger(__name__)


def load_dataset_from_file(file_path: str) -> List[Dict]:
    """从文件加载数据集"""
    if not os.path.exists(file_path):
        logger.error(f"数据集文件不存在: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                data = json.load(f)
            elif file_path.endswith('.jsonl'):
                data = []
                for line in f:
                    data.append(json.loads(line.strip()))
            else:
                logger.error(f"不支持的文件格式: {file_path}")
                return []
        
        logger.info(f"成功加载数据集: {len(data)} 条记录")
        return data
        
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        return []


def create_sample_dataset(num_samples: int = 100) -> List[Dict]:
    """创建示例数据集"""
    sample_data = [
        {
            "instruction": "解释什么是机器学习",
            "output": "机器学习是人工智能的一个子领域，它使计算机能够在没有明确编程的情况下从数据中学习。通过算法分析大量数据，机器学习系统可以识别模式并做出预测或决策。"
        },
        {
            "instruction": "什么是深度学习？",
            "output": "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的学习过程。它特别擅长处理图像、语音和文本等复杂数据，在计算机视觉、自然语言处理等领域取得了突破性进展。"
        },
        {
            "instruction": "介绍一下Python编程语言",
            "output": "Python是一种高级、解释型编程语言，以其简洁明了的语法而闻名。它广泛应用于Web开发、数据科学、人工智能、自动化等领域。Python拥有丰富的库生态系统，是初学者友好的编程语言。"
        },
        {
            "instruction": "如何学习编程？",
            "output": "学习编程建议：1. 选择一门适合的编程语言开始；2. 掌握基础语法和概念；3. 通过实际项目练习；4. 阅读他人代码学习最佳实践；5. 持续练习和学习新技术。重要的是保持耐心和持续的实践。"
        },
        {
            "instruction": "什么是算法？",
            "output": "算法是解决问题的一系列明确定义的步骤。在计算机科学中，算法是执行特定任务的指令集合。好的算法应该具备正确性、效率性、可读性和健壮性等特点。"
        }
    ]
    
    # 重复样本数据以达到指定数量
    dataset = []
    for i in range(num_samples):
        sample = sample_data[i % len(sample_data)].copy()
        sample["id"] = i + 1
        dataset.append(sample)
    
    return dataset


def create_finetune_config_from_args(args: argparse.Namespace) -> FinetuneConfig:
    """从命令行参数创建微调配置"""
    
    # LoRA配置
    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules.split(',') if args.target_modules else None,
        lora_dropout=args.lora_dropout
    )
    
    # 微调配置
    config = FinetuneConfig(
        model_name_or_path=args.model_path,
        output_dir=args.output_dir,
        use_lora=args.use_lora,
        lora_config=lora_config,
        
        # 训练参数
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        
        # 量化配置
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        
        # 混合精度
        fp16=args.fp16,
        bf16=args.bf16,
        
        # 其他配置
        max_seq_length=args.max_seq_length,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=args.gradient_checkpointing
    )
    
    return config


def lora_finetune(args: argparse.Namespace):
    """LoRA微调"""
    logger.info("开始LoRA微调...")
    
    # 创建配置
    if args.preset:
        config = get_finetune_preset(args.preset, args.model_path)
        # 应用命令行覆盖
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.num_epochs is not None:
            config.num_train_epochs = args.num_epochs
        if args.learning_rate is not None:
            config.learning_rate = args.learning_rate
    else:
        config = create_finetune_config_from_args(args)
        config.use_lora = True  # 确保使用LoRA
    
    logger.info(f"微调配置: {config}")
    
    # 加载数据集
    if args.dataset_file:
        train_data = load_dataset_from_file(args.dataset_file)
    else:
        logger.info("使用示例数据集")
        train_data = create_sample_dataset(args.num_samples)
    
    if not train_data:
        logger.error("无法加载训练数据")
        return
    
    # 分割训练和验证集
    val_ratio = args.val_ratio
    val_size = int(len(train_data) * val_ratio)
    val_data = train_data[:val_size] if val_size > 0 else []
    train_data = train_data[val_size:]
    
    logger.info(f"训练集大小: {len(train_data)}")
    logger.info(f"验证集大小: {len(val_data)}")
    
    try:
        # 创建微调器
        finetuner = FineTuner(config)
        finetuner.setup()
        
        # 处理数据
        processor = DataProcessor(finetuner.tokenizer, config)
        
        # 处理训练数据
        train_dataset = processor.process_instruction_dataset(train_data)
        train_dataset = processor.tokenize_dataset(train_dataset)
        
        # 处理验证数据
        eval_dataset = None
        if val_data:
            eval_dataset = processor.process_instruction_dataset(val_data)
            eval_dataset = processor.tokenize_dataset(eval_dataset)
        
        # 开始微调
        if args.use_sft:
            # 使用SFT训练器
            logger.info("使用SFT训练器进行微调...")
            train_result = finetuner.finetune_with_sft(train_dataset, eval_dataset)
        else:
            # 使用标准训练器
            logger.info("使用标准训练器进行微调...")
            train_result = finetuner.finetune_with_trainer(train_dataset, eval_dataset)
        
        # 保存LoRA权重
        if config.use_lora:
            lora_output_dir = os.path.join(config.output_dir, "lora_weights")
            finetuner.save_lora_weights(lora_output_dir)
            
            # 合并并保存完整模型（可选）
            if args.merge_lora:
                merged_output_dir = os.path.join(config.output_dir, "merged_model")
                finetuner.merge_and_save_model(merged_output_dir)
        
        logger.info("LoRA微调完成！")
        
    except Exception as e:
        logger.error(f"LoRA微调失败: {e}", exc_info=True)
        sys.exit(1)


def full_finetune(args: argparse.Namespace):
    """全参数微调"""
    logger.info("开始全参数微调...")
    
    # 创建配置
    config = create_finetune_config_from_args(args)
    config.use_lora = False  # 不使用LoRA
    
    # 全参数微调通常需要更小的学习率
    if args.learning_rate is None:
        config.learning_rate = 1e-5
    
    logger.info(f"微调配置: {config}")
    
    # 加载数据集
    if args.dataset_file:
        train_data = load_dataset_from_file(args.dataset_file)
    else:
        logger.info("使用示例数据集")
        train_data = create_sample_dataset(args.num_samples)
    
    if not train_data:
        logger.error("无法加载训练数据")
        return
    
    try:
        # 创建微调器
        finetuner = FineTuner(config)
        finetuner.setup()
        
        # 处理数据
        processor = DataProcessor(finetuner.tokenizer, config)
        train_dataset = processor.process_instruction_dataset(train_data)
        train_dataset = processor.tokenize_dataset(train_dataset)
        
        # 开始微调
        train_result = finetuner.finetune_with_trainer(train_dataset)
        
        logger.info("全参数微调完成！")
        
    except Exception as e:
        logger.error(f"全参数微调失败: {e}", exc_info=True)
        sys.exit(1)


def merge_lora_weights(args: argparse.Namespace):
    """合并LoRA权重"""
    logger.info("开始合并LoRA权重...")
    
    base_model_path = args.model_path
    lora_path = args.lora_path
    output_path = args.output_dir
    
    if not os.path.exists(lora_path):
        logger.error(f"LoRA权重路径不存在: {lora_path}")
        return
    
    try:
        LoRAMerger.load_and_merge_lora(
            base_model_path=base_model_path,
            lora_path=lora_path,
            output_path=output_path,
            torch_dtype=torch.float16 if args.fp16 else torch.float32
        )
        
        logger.info(f"LoRA权重合并完成，输出路径: {output_path}")
        
    except Exception as e:
        logger.error(f"LoRA权重合并失败: {e}", exc_info=True)
        sys.exit(1)


def evaluate_model(args: argparse.Namespace):
    """评估微调后的模型"""
    logger.info("开始模型评估...")
    
    # 创建配置
    config = FinetuneConfig(
        model_name_or_path=args.model_path,
        output_dir=args.output_dir
    )
    
    # 加载测试数据
    if args.dataset_file:
        eval_data = load_dataset_from_file(args.dataset_file)
    else:
        eval_data = create_sample_dataset(50)
    
    try:
        # 创建微调器
        finetuner = FineTuner(config)
        finetuner.setup()
        
        # 处理数据
        processor = DataProcessor(finetuner.tokenizer, config)
        eval_dataset = processor.process_instruction_dataset(eval_data)
        eval_dataset = processor.tokenize_dataset(eval_dataset)
        
        # 评估模型
        eval_result = finetuner.evaluate_model(eval_dataset)
        
        print("\n=== 模型评估结果 ===")
        for key, value in eval_result.items():
            print(f"{key}: {value}")
        
        # 保存评估结果
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(eval_result, f, indent=2, ensure_ascii=False)
            logger.info(f"评估结果已保存到: {args.output_file}")
        
    except Exception as e:
        logger.error(f"模型评估失败: {e}", exc_info=True)
        sys.exit(1)


def quick_finetune_demo(args: argparse.Namespace):
    """快速微调演示"""
    logger.info("开始快速微调演示...")
    
    # 准备示例数据
    train_data = create_sample_dataset(args.num_samples)
    
    logger.info(f"使用 {len(train_data)} 条示例数据进行快速微调")
    
    try:
        # 快速微调
        finetuner = quick_finetune(
            model_path=args.model_path,
            train_data=train_data,
            output_dir=args.output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            learning_rate=1e-4,
            load_in_4bit=True,
            use_lora=True
        )
        
        logger.info("快速微调演示完成！")
        
    except Exception as e:
        logger.error(f"快速微调演示失败: {e}", exc_info=True)
        sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DCU-in-Action 模型微调脚本")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="微调命令")
    
    # LoRA微调
    lora_parser = subparsers.add_parser("lora", help="LoRA微调")
    lora_parser.add_argument("--model_path", type=str, required=True, help="基础模型路径")
    lora_parser.add_argument("--dataset_file", type=str, help="训练数据集文件")
    lora_parser.add_argument("--output_dir", type=str, default="./lora_output", help="输出目录")
    lora_parser.add_argument("--preset", type=str, help="预设配置")
    lora_parser.add_argument("--merge_lora", action="store_true", help="合并LoRA权重")
    
    # 全参数微调
    full_parser = subparsers.add_parser("full", help="全参数微调")
    full_parser.add_argument("--model_path", type=str, required=True, help="基础模型路径")
    full_parser.add_argument("--dataset_file", type=str, help="训练数据集文件")
    full_parser.add_argument("--output_dir", type=str, default="./full_output", help="输出目录")
    
    # 合并LoRA权重
    merge_parser = subparsers.add_parser("merge", help="合并LoRA权重")
    merge_parser.add_argument("--model_path", type=str, required=True, help="基础模型路径")
    merge_parser.add_argument("--lora_path", type=str, required=True, help="LoRA权重路径")
    merge_parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    
    # 模型评估
    eval_parser = subparsers.add_parser("evaluate", help="评估模型")
    eval_parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    eval_parser.add_argument("--dataset_file", type=str, help="评估数据集文件")
    eval_parser.add_argument("--output_file", type=str, help="评估结果输出文件")
    eval_parser.add_argument("--output_dir", type=str, default="./eval_output", help="输出目录")
    
    # 快速微调
    quick_parser = subparsers.add_parser("quick", help="快速微调演示")
    quick_parser.add_argument("--model_path", type=str, required=True, help="基础模型路径")
    quick_parser.add_argument("--output_dir", type=str, default="./quick_output", help="输出目录")
    quick_parser.add_argument("--num_samples", type=int, default=50, help="样本数量")
    
    # 公共参数
    for sub_parser in [lora_parser, full_parser]:
        # 训练参数
        sub_parser.add_argument("--num_epochs", type=float, default=3.0, help="训练轮数")
        sub_parser.add_argument("--batch_size", type=int, default=4, help="训练批量大小")
        sub_parser.add_argument("--eval_batch_size", type=int, default=4, help="评估批量大小")
        sub_parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
        sub_parser.add_argument("--learning_rate", type=float, help="学习率")
        sub_parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
        sub_parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热比例")
        sub_parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪")
        
        # LoRA参数
        sub_parser.add_argument("--use_lora", action="store_true", default=True, help="使用LoRA")
        sub_parser.add_argument("--lora_r", type=int, default=16, help="LoRA秩")
        sub_parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
        sub_parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
        sub_parser.add_argument("--target_modules", type=str, help="目标模块（逗号分隔）")
        
        # 量化配置
        sub_parser.add_argument("--load_in_4bit", action="store_true", help="使用4bit量化")
        sub_parser.add_argument("--load_in_8bit", action="store_true", help="使用8bit量化")
        
        # 混合精度
        sub_parser.add_argument("--fp16", action="store_true", help="使用FP16")
        sub_parser.add_argument("--bf16", action="store_true", help="使用BF16")
        
        # 其他配置
        sub_parser.add_argument("--max_seq_length", type=int, default=2048, help="最大序列长度")
        sub_parser.add_argument("--save_steps", type=int, default=500, help="保存步数")
        sub_parser.add_argument("--logging_steps", type=int, default=10, help="日志步数")
        sub_parser.add_argument("--eval_steps", type=int, default=500, help="评估步数")
        sub_parser.add_argument("--save_total_limit", type=int, default=3, help="保存检查点数量限制")
        sub_parser.add_argument("--gradient_checkpointing", action="store_true", help="梯度检查点")
        
        # 数据配置
        sub_parser.add_argument("--num_samples", type=int, default=100, help="使用的样本数量")
        sub_parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
        sub_parser.add_argument("--use_sft", action="store_true", help="使用SFT训练器")
        
        # 调试
        sub_parser.add_argument("--debug", action="store_true", help="调试模式")
    
    # 为合并命令添加参数
    merge_parser.add_argument("--fp16", action="store_true", help="使用FP16精度")
    merge_parser.add_argument("--debug", action="store_true", help="调试模式")
    
    # 为评估命令添加参数
    eval_parser.add_argument("--debug", action="store_true", help="调试模式")
    
    # 为快速微调添加参数
    quick_parser.add_argument("--debug", action="store_true", help="调试模式")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 设置日志
    log_level = "DEBUG" if args.debug else "INFO"
    setup_global_logging(level=log_level)
    
    try:
        # DCU设备检查
        device_manager = DCUDeviceManager()
        device_info = device_manager.get_device_info()
        logger.info(f"DCU设备信息: {device_info}")
        
        # 执行相应命令
        if args.command == "lora":
            lora_finetune(args)
        elif args.command == "full":
            full_finetune(args)
        elif args.command == "merge":
            merge_lora_weights(args)
        elif args.command == "evaluate":
            evaluate_model(args)
        elif args.command == "quick":
            quick_finetune_demo(args)
        
        logger.info("微调任务完成！")
        
    except Exception as e:
        logger.error(f"微调任务失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 