# SPDX-License-Identifier: Apache-2.0
"""
基准测试工具函数模块

这个模块提供了基准测试结果处理的工具函数，包括：
- 结果格式转换
- JSON序列化处理
- 与外部基准测试系统的集成

主要功能：
- 将vLLM基准测试结果转换为PyTorch OSS基准测试数据库格式
- 处理JSON序列化中的特殊值（如无穷大）
- 提供统一的结果保存接口
"""

import argparse
import json
import math
import os
from typing import Any


def convert_to_pytorch_benchmark_format(args: argparse.Namespace,
                                        metrics: dict[str, list],
                                        extra_info: dict[str, Any]) -> list:
    """
    将基准测试结果转换为PyTorch OSS基准测试格式

    这个函数将vLLM的基准测试结果转换为PyTorch开源基准测试数据库
    所使用的标准格式，每个指标对应一条记录。

    参考文档：
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database

    Args:
        args: 命令行参数命名空间，包含测试配置
        metrics: 指标字典，键为指标名称，值为指标值列表
        extra_info: 额外信息字典，包含元数据

    Returns:
        list: PyTorch基准测试格式的记录列表
    """
    records = []
    # 只有设置了环境变量才进行格式转换
    if not os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):
        return records

    # 为每个指标创建一条记录
    for name, benchmark_values in metrics.items():
        record = {
            "benchmark": {
                "name": "vLLM benchmark",           # 基准测试名称
                "extra_info": {
                    "args": vars(args),             # 命令行参数
                },
            },
            "model": {
                "name": args.model,                 # 模型名称
            },
            "metric": {
                "name": name,                       # 指标名称
                "benchmark_values": benchmark_values, # 指标值列表
                "extra_info": extra_info,           # 额外信息
            },
        }

        # 处理tensor_parallel_size参数
        tp = record["benchmark"]["extra_info"]["args"].get(
            "tensor_parallel_size")
        # 如果参数中没有但元数据中有，则添加到参数中
        if not tp and "tensor_parallel_size" in extra_info:
            record["benchmark"]["extra_info"]["args"][
                "tensor_parallel_size"] = extra_info["tensor_parallel_size"]

        records.append(record)

    return records


class InfEncoder(json.JSONEncoder):
    """
    自定义JSON编码器，处理无穷大值

    这个编码器继承自json.JSONEncoder，专门用于处理基准测试结果中
    可能出现的无穷大值。在某些测试场景下（如无限制请求速率），
    会产生float('inf')值，标准JSON编码器无法处理这种值。

    主要功能：
    - 递归遍历数据结构
    - 将float('inf')转换为字符串"inf"
    - 保持其他数据类型不变
    """

    def clear_inf(self, o: Any):
        """
        递归清理数据结构中的无穷大值

        Args:
            o: 要处理的对象，可以是任意类型

        Returns:
            处理后的对象，无穷大值被转换为"inf"字符串
        """
        if isinstance(o, dict):
            # 递归处理字典的每个键值对
            return {k: self.clear_inf(v) for k, v in o.items()}
        elif isinstance(o, list):
            # 递归处理列表的每个元素
            return [self.clear_inf(v) for v in o]
        elif isinstance(o, float) and math.isinf(o):
            # 将无穷大浮点数转换为字符串
            return "inf"
        return o

    def iterencode(self, o: Any, *args, **kwargs) -> Any:
        """
        重写iterencode方法，在编码前清理无穷大值

        Args:
            o: 要编码的对象
            *args, **kwargs: 传递给父类的参数

        Returns:
            编码后的JSON字符串迭代器
        """
        return super().iterencode(self.clear_inf(o), *args, **kwargs)


def write_to_json(filename: str, records: list) -> None:
    """
    将记录列表写入JSON文件

    使用自定义的InfEncoder来处理可能存在的无穷大值，
    确保所有基准测试结果都能正确序列化为JSON格式。

    Args:
        filename: 输出文件名
        records: 要写入的记录列表
    """
    with open(filename, "w") as f:
        json.dump(records, f, cls=InfEncoder)
