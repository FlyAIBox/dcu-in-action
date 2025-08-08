# 文件: benchmark_utils.py
# 功能: vLLM压测工具函数集合
# 许可证: Apache-2.0

import argparse
import json
import math
import os
from typing import Any


def convert_to_pytorch_benchmark_format(
    args: argparse.Namespace, metrics: dict[str, list], extra_info: dict[str, Any]
) -> list:
    """
    将压测结果转换为PyTorch OSS基准测试数据库格式
    
    参数:
        args: 命令行参数命名空间
        metrics: 性能指标字典，每个指标对应一个数值列表
        extra_info: 额外的元信息字典
    
    返回:
        list: 格式化后的记录列表，每个指标一条记录
        
    参考:
        https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    records = []
    # 只有设置了环境变量才启用PyTorch格式转换
    if not os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):
        return records

    # 为每个性能指标创建一条记录
    for name, benchmark_values in metrics.items():
        record = {
            "benchmark": {
                "name": "vLLM benchmark",           # 基准测试名称
                "extra_info": {
                    "args": vars(args),             # 命令行参数转为字典
                },
            },
            "model": {
                "name": args.model,                 # 模型名称
            },
            "metric": {
                "name": name,                       # 指标名称
                "benchmark_values": benchmark_values, # 指标数值列表
                "extra_info": extra_info,           # 额外信息
            },
        }

        # 处理tensor_parallel_size参数
        tp = record["benchmark"]["extra_info"]["args"].get("tensor_parallel_size")
        # 如果额外信息中包含tensor_parallel_size但参数中没有，则添加到参数中
        if not tp and "tensor_parallel_size" in extra_info:
            record["benchmark"]["extra_info"]["args"]["tensor_parallel_size"] = (
                extra_info["tensor_parallel_size"]
            )

        records.append(record)

    return records


class InfEncoder(json.JSONEncoder):
    """
    自定义JSON编码器，用于处理无穷大值
    
    在JSON序列化时将float('inf')转换为字符串"inf"，
    避免JSON序列化错误
    """
    
    def clear_inf(self, o: Any):
        """
        递归清理对象中的无穷大值
        
        参数:
            o: 要处理的对象
            
        返回:
            处理后的对象，无穷大值被替换为"inf"字符串
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
        """
        return super().iterencode(self.clear_inf(o), *args, **kwargs)


def write_to_json(filename: str, records: list) -> None:
    """
    将记录列表写入JSON文件
    
    参数:
        filename: 输出文件名
        records: 要写入的记录列表
        
    使用自定义的InfEncoder处理可能存在的无穷大值
    """
    with open(filename, "w") as f:
        json.dump(records, f, cls=InfEncoder)