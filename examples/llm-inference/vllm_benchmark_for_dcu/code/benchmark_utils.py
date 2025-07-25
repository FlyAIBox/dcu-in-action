# SPDX-License-Identifier: Apache-2.0
"""
基准测试工具模块 - 提供结果格式化和数据处理功能

这个模块包含了基准测试结果处理的辅助函数，主要用于：
1. 将测试结果转换为PyTorch基准测试格式
2. 处理JSON序列化中的特殊值 (如无穷大)
3. 标准化测试结果的输出格式

适用于海光DCU大模型推理性能测试的结果后处理。
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
    将基准测试结果转换为PyTorch OSS基准测试数据库格式

    这个函数将vLLM基准测试的结果转换为PyTorch开源基准测试数据库的标准格式，
    便于与其他基准测试结果进行比较和分析。

    参数:
        args: argparse.Namespace - 命令行参数对象，包含测试配置信息
        metrics: dict[str, list] - 性能指标字典，键为指标名称，值为指标数值列表
        extra_info: dict[str, Any] - 额外信息字典，包含测试环境和配置信息

    返回:
        list: PyTorch基准测试格式的记录列表，每个指标一条记录

    格式说明:
        每条记录包含以下字段：
        - benchmark: 基准测试信息 (名称、参数等)
        - model: 模型信息 (名称、路径等)
        - metric: 指标信息 (名称、数值、额外信息等)

    参考文档:
        https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    records = []
    if not os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):
        return records

    for name, benchmark_values in metrics.items():
        record = {
            "benchmark": {
                "name": "vLLM benchmark",
                "extra_info": {
                    "args": vars(args),
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

        tp = record["benchmark"]["extra_info"]["args"].get(
            "tensor_parallel_size")
        # Save tensor_parallel_size parameter if it's part of the metadata
        if not tp and "tensor_parallel_size" in extra_info:
            record["benchmark"]["extra_info"]["args"][
                "tensor_parallel_size"] = extra_info["tensor_parallel_size"]

        records.append(record)

    return records


class InfEncoder(json.JSONEncoder):
    """
    自定义JSON编码器 - 处理无穷大值的序列化

    在基准测试中，某些指标可能出现无穷大值 (如请求速率设为inf)，
    标准的JSON编码器无法处理这些值。这个自定义编码器将无穷大值
    转换为字符串 "inf"，确保JSON序列化的正确性。
    """

    def clear_inf(self, o: Any):
        """
        递归清理数据结构中的无穷大值

        参数:
            o: Any - 待处理的数据对象

        返回:
            Any - 处理后的数据对象，无穷大值被替换为 "inf" 字符串
        """
        if isinstance(o, dict):
            # 递归处理字典中的每个键值对
            return {k: self.clear_inf(v) for k, v in o.items()}
        elif isinstance(o, list):
            # 递归处理列表中的每个元素
            return [self.clear_inf(v) for v in o]
        elif isinstance(o, float) and math.isinf(o):
            # 将无穷大浮点数转换为字符串
            return "inf"
        return o

    def iterencode(self, o: Any, *args, **kwargs) -> Any:
        """
        重写JSON编码的迭代方法，应用无穷大值清理
        """
        return super().iterencode(self.clear_inf(o), *args, **kwargs)


def write_to_json(filename: str, records: list) -> None:
    """
    将记录列表写入JSON文件

    使用自定义的InfEncoder处理无穷大值，确保所有基准测试结果
    都能正确序列化到JSON文件中。

    参数:
        filename: str - 输出文件名
        records: list - 待写入的记录列表
    """
    with open(filename, "w") as f:
        json.dump(records, f, cls=InfEncoder)
