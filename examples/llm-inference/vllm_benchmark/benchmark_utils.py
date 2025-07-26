#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_utils.py - 基准测试工具函数模块

本模块提供了基准测试过程中使用的各种工具函数和辅助类。
主要功能包括：
1. 数据格式转换：将基准测试结果转换为PyTorch OSS基准测试数据库格式
2. JSON序列化：提供自定义的JSON编码器，处理无穷大值等特殊情况
3. 文件写入：安全地将测试结果写入JSON文件

工具函数说明：
- convert_to_pytorch_benchmark_format: 转换为PyTorch基准测试格式
- InfEncoder: 自定义JSON编码器，处理无穷大值
- write_to_json: 写入JSON文件的工具函数

作者：vLLM团队
修改：添加详细中文注释
"""

# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import os
from typing import Any


def convert_to_pytorch_benchmark_format(
    args: argparse.Namespace, metrics: dict[str, list], extra_info: dict[str, Any]
) -> list:
    """
    将基准测试结果转换为PyTorch OSS基准测试数据库格式
    
    将vLLM基准测试的结果转换为PyTorch开源基准测试数据库要求的格式，
    每个指标对应一个记录。这种格式便于与PyTorch生态系统中的其他
    基准测试结果进行比较和集成。
    
    Args:
        args: 包含命令行参数的命名空间对象
        metrics: 指标名称到值列表的字典映射
        extra_info: 包含额外信息的字典
        
    Returns:
        list: 符合PyTorch基准测试格式的记录列表
        
    References:
        PyTorch OSS基准测试数据库集成指南：
        https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    records = []
    # 检查环境变量，决定是否保存为PyTorch基准测试格式
    if not os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):
        return records

    # 为每个指标创建一个单独的记录
    for name, benchmark_values in metrics.items():
        record = {
            "benchmark": {
                "name": "vLLM benchmark",
                "extra_info": {
                    "args": vars(args),  # 转换命令行参数为字典
                },
            },
            "model": {
                "name": args.model,
            },
            "metric": {
                "name": name,
                "benchmark_values": benchmark_values,
                "extra_info": extra_info,
            },
        }

        tp = record["benchmark"]["extra_info"]["args"].get("tensor_parallel_size")
        # 如果元数据中包含tensor_parallel_size参数，则保存它
        if not tp and "tensor_parallel_size" in extra_info:
            record["benchmark"]["extra_info"]["args"]["tensor_parallel_size"] = (
                extra_info["tensor_parallel_size"]
            )

        records.append(record)

    return records


class InfEncoder(json.JSONEncoder):
    """
    自定义JSON编码器，处理无穷大值
    
    这个编码器扩展了标准的JSON编码器，能够正确处理Python中的
    无穷大值（float('inf')），将其转换为字符串"inf"，
    避免JSON序列化时出现错误。
    """
    
    def clear_inf(self, o: Any):
        """
        递归清理数据结构中的无穷大值
        
        Args:
            o: 要处理的任意类型数据
            
        Returns:
            处理后的数据，无穷大值被替换为字符串"inf"
        """
        if isinstance(o, dict):
            return {k: self.clear_inf(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self.clear_inf(v) for v in o]
        elif isinstance(o, float) and math.isinf(o):
            return "inf"
        return o

    def iterencode(self, o: Any, *args, **kwargs) -> Any:
        """
        重写iterencode方法，在编码前清理无穷大值
        
        Args:
            o: 要编码的对象
            *args, **kwargs: 传递给父类的其他参数
            
        Returns:
            编码后的JSON字符串迭代器
        """
        return super().iterencode(self.clear_inf(o), *args, **kwargs)


def write_to_json(filename: str, records: list) -> None:
    """
    将记录列表写入JSON文件
    
    使用自定义的InfEncoder来处理可能存在的无穷大值，
    确保JSON文件能够正确序列化和保存。
    
    Args:
        filename: 要写入的文件名
        records: 要保存的记录列表
        
    Note:
        该函数会覆盖现有文件，使用UTF-8编码
    """
    with open(filename, "w") as f:
        json.dump(records, f, cls=InfEncoder)