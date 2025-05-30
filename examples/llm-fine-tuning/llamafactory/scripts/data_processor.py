#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理脚本 - 将各种格式的数据转换为LLaMA Factory支持的格式
作者: DCU实战项目组
"""

import json
import pandas as pd
import argparse
import os
import re
from typing import List, Dict, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """数据处理器，支持多种格式转换为Alpaca格式"""
    
    def __init__(self, quality_threshold: float = 0.8):
        self.quality_threshold = quality_threshold
        self.alpaca_format = {
            "instruction": "",
            "input": "",
            "output": ""
        }
    
    def csv_to_alpaca(self, csv_file: str, instruction_col: str, 
                     input_col: Optional[str] = None, output_col: str = None) -> List[Dict]:
        """
        将CSV文件转换为Alpaca格式
        
        Args:
            csv_file: CSV文件路径
            instruction_col: 指令列名
            input_col: 输入列名（可选）
            output_col: 输出列名
            
        Returns:
            转换后的Alpaca格式数据列表
        """
        logger.info(f"开始处理CSV文件: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding='gbk')
        
        logger.info(f"CSV文件包含 {len(df)} 行数据")
        
        # 检查列是否存在
        required_cols = [instruction_col]
        if output_col:
            required_cols.append(output_col)
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV文件中缺少列: {missing_cols}")
        
        alpaca_data = []
        skipped_count = 0
        
        for idx, row in df.iterrows():
            try:
                # 处理指令
                instruction = str(row[instruction_col]).strip()
                if pd.isna(row[instruction_col]) or len(instruction) < 5:
                    skipped_count += 1
                    continue
                
                # 处理输入（可选）
                input_text = ""
                if input_col and input_col in df.columns and pd.notna(row[input_col]):
                    input_text = str(row[input_col]).strip()
                
                # 处理输出
                output_text = ""
                if output_col and pd.notna(row[output_col]):
                    output_text = str(row[output_col]).strip()
                
                # 创建Alpaca格式数据
                item = {
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text
                }
                
                alpaca_data.append(item)
                
            except Exception as e:
                logger.warning(f"处理第 {idx} 行数据时出错: {e}")
                skipped_count += 1
                continue
        
        logger.info(f"转换完成: {len(alpaca_data)} 条有效数据，跳过 {skipped_count} 条")
        return alpaca_data
    
    def validate_data(self, data: List[Dict]) -> List[Dict]:
        """
        数据质量验证
        
        Args:
            data: 待验证的数据列表
            
        Returns:
            验证通过的数据列表
        """
        logger.info("开始数据质量验证...")
        
        validated_data = []
        validation_stats = {
            "missing_fields": 0,
            "too_short": 0,
            "low_quality": 0,
            "duplicates": 0,
            "passed": 0
        }
        
        seen_instructions = set()
        
        for item in data:
            # 检查必要字段
            if not all(key in item for key in ["instruction", "output"]):
                validation_stats["missing_fields"] += 1
                continue
            
            # 长度检查
            if len(item["instruction"]) < 5 or len(item["output"]) < 5:
                validation_stats["too_short"] += 1
                continue
            
            # 去重检查
            instruction_key = item["instruction"].lower().strip()
            if instruction_key in seen_instructions:
                validation_stats["duplicates"] += 1
                continue
            seen_instructions.add(instruction_key)
            
            # 内容质量检查
            if not self._quality_check(item):
                validation_stats["low_quality"] += 1
                continue
            
            validated_data.append(item)
            validation_stats["passed"] += 1
        
        # 打印验证统计
        logger.info("数据验证完成:")
        for key, count in validation_stats.items():
            logger.info(f"  {key}: {count}")
        
        logger.info(f"验证通过率: {validation_stats['passed']}/{len(data)} ({validation_stats['passed']/len(data)*100:.1f}%)")
        
        return validated_data
    
    def _quality_check(self, item: Dict) -> bool:
        """
        内容质量检查
        
        Args:
            item: 单条数据
            
        Returns:
            是否通过质量检查
        """
        text = item["instruction"] + " " + item.get("input", "") + " " + item["output"]
        
        # 去除重复字符过多的内容
        if len(text) > 0 and len(set(text)) / len(text) < 0.3:
            return False
        
        # 检查是否包含足够的信息
        words = text.split()
        if len(words) < 10:
            return False
        
        # 检查是否包含过多特殊字符
        special_char_ratio = len(re.findall(r'[^\w\s\u4e00-\u9fff]', text)) / len(text)
        if special_char_ratio > 0.3:
            return False
        
        # 检查指令和输出的相关性（简单检查）
        instruction_words = set(item["instruction"].lower().split())
        output_words = set(item["output"].lower().split())
        
        # 如果指令和输出完全没有共同词汇（除了停用词），可能质量有问题
        stop_words = {"的", "了", "在", "是", "我", "你", "他", "她", "它", "们", "这", "那", "with", "the", "a", "an", "and", "or", "but", "is", "are", "was", "were"}
        instruction_meaningful = instruction_words - stop_words
        output_meaningful = output_words - stop_words
        
        if len(instruction_meaningful) > 3 and len(output_meaningful) > 3:
            overlap = len(instruction_meaningful & output_meaningful)
            if overlap == 0 and len(item["output"]) < 50:  # 短回答且无关联性
                return False
        
        return True
    
    def augment_data(self, data: List[Dict], augment_ratio: float = 0.2) -> List[Dict]:
        """
        数据增强
        
        Args:
            data: 原始数据
            augment_ratio: 增强比例
            
        Returns:
            增强后的数据
        """
        if augment_ratio <= 0:
            return data
        
        logger.info(f"开始数据增强，增强比例: {augment_ratio}")
        
        augmented_data = data.copy()
        num_to_augment = int(len(data) * augment_ratio)
        
        import random
        random.seed(42)
        
        for i in range(num_to_augment):
            original_item = random.choice(data)
            
            # 简单的数据增强：同义词替换、句式变化等
            augmented_item = {
                "instruction": self._augment_text(original_item["instruction"]),
                "input": self._augment_text(original_item.get("input", "")),
                "output": original_item["output"]  # 保持输出不变
            }
            
            augmented_data.append(augmented_item)
        
        logger.info(f"数据增强完成: {len(data)} -> {len(augmented_data)}")
        return augmented_data
    
    def _augment_text(self, text: str) -> str:
        """
        文本增强（简单实现）
        """
        if not text or len(text) < 10:
            return text
        
        # 简单的同义词替换表
        synonyms = {
            "请": "麻烦",
            "帮我": "协助我", 
            "如何": "怎样",
            "什么": "啥",
            "怎么": "如何",
            "介绍": "说明",
            "解释": "阐述"
        }
        
        result = text
        for original, replacement in synonyms.items():
            if original in result:
                result = result.replace(original, replacement, 1)  # 只替换一次
                break
        
        return result
    
    def split_dataset(self, data: List[Dict], test_size: float = 0.1, 
                     val_size: float = 0.1, random_state: int = 42) -> Dict[str, List[Dict]]:
        """
        数据集分割
        
        Args:
            data: 完整数据集
            test_size: 测试集比例
            val_size: 验证集比例
            random_state: 随机种子
            
        Returns:
            分割后的数据集字典
        """
        logger.info(f"开始数据集分割: 训练集 {1-test_size-val_size:.1%}, 验证集 {val_size:.1%}, 测试集 {test_size:.1%}")
        
        # 训练集和临时集分割
        train_data, temp_data = train_test_split(
            data, 
            test_size=(test_size + val_size), 
            random_state=random_state,
            shuffle=True
        )
        
        # 验证集和测试集分割
        if val_size > 0:
            val_data, test_data = train_test_split(
                temp_data, 
                test_size=(test_size/(test_size + val_size)), 
                random_state=random_state
            )
        else:
            val_data = []
            test_data = temp_data
        
        result = {
            "train": train_data,
            "validation": val_data,
            "test": test_data
        }
        
        # 打印分割统计
        for split_name, split_data in result.items():
            logger.info(f"  {split_name}: {len(split_data)} 条记录")
        
        return result
    
    def save_dataset(self, data: Dict[str, List[Dict]], output_dir: str):
        """
        保存数据集
        
        Args:
            data: 数据集字典
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, split_data in data.items():
            if len(split_data) == 0:
                continue
                
            output_file = os.path.join(output_dir, f"{split_name}.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"保存 {split_name} 数据集：{len(split_data)} 条记录 -> {output_file}")
        
        # 保存数据统计信息
        stats_file = os.path.join(output_dir, "dataset_stats.json")
        stats = {
            "total_samples": sum(len(split_data) for split_data in data.values()),
            "splits": {name: len(split_data) for name, split_data in data.items()},
            "sample_data": {name: split_data[:3] if split_data else [] for name, split_data in data.items()}
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存数据统计信息 -> {stats_file}")

def main():
    parser = argparse.ArgumentParser(description="数据处理工具 - CSV转Alpaca格式")
    parser.add_argument("--input", required=True, help="输入CSV文件路径")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--instruction_col", required=True, help="指令列名")
    parser.add_argument("--input_col", help="输入列名（可选）")
    parser.add_argument("--output_col", required=True, help="输出列名")
    parser.add_argument("--test_size", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--val_size", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--quality_threshold", type=float, default=0.8, help="质量阈值")
    parser.add_argument("--augment_ratio", type=float, default=0.0, help="数据增强比例")
    parser.add_argument("--no_validation", action="store_true", help="跳过数据验证")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        return
    
    processor = DataProcessor(quality_threshold=args.quality_threshold)
    
    try:
        # 转换格式
        logger.info("🔄 开始数据转换...")
        alpaca_data = processor.csv_to_alpaca(
            args.input, 
            args.instruction_col, 
            args.input_col, 
            args.output_col
        )
        
        if len(alpaca_data) == 0:
            logger.error("没有成功转换任何数据，请检查列名和数据格式")
            return
        
        # 验证数据
        if not args.no_validation:
            logger.info("✅ 开始数据验证...")
            validated_data = processor.validate_data(alpaca_data)
            
            if len(validated_data) == 0:
                logger.error("没有数据通过验证，请检查数据质量")
                return
        else:
            validated_data = alpaca_data
        
        # 数据增强
        if args.augment_ratio > 0:
            logger.info("🔄 开始数据增强...")
            validated_data = processor.augment_data(validated_data, args.augment_ratio)
        
        # 分割数据集
        logger.info("📊 开始数据集分割...")
        dataset_splits = processor.split_dataset(
            validated_data, 
            args.test_size, 
            args.val_size
        )
        
        # 保存数据集
        logger.info("💾 保存数据集...")
        processor.save_dataset(dataset_splits, args.output)
        
        logger.info("🎉 数据处理完成！")
        
        # 显示处理结果摘要
        total_samples = sum(len(split_data) for split_data in dataset_splits.values())
        logger.info(f"处理结果摘要:")
        logger.info(f"  总样本数: {total_samples}")
        logger.info(f"  训练集: {len(dataset_splits['train'])}")
        logger.info(f"  验证集: {len(dataset_splits['validation'])}")
        logger.info(f"  测试集: {len(dataset_splits['test'])}")
        logger.info(f"  输出目录: {args.output}")
        
    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 