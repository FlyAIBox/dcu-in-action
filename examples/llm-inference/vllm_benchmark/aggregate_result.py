#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aggregate_result.py - 基准测试结果聚合模块

本模块负责将多个基准测试运行产生的JSON结果文件聚合成一个统一的CSV文件。
主要功能包括：
1. 批量读取results目录下的所有JSON测试结果文件
2. 从文件名中解析输入/输出长度参数
3. 将所有结果标准化并合并到一个DataFrame中
4. 导出为CSV格式，便于进一步分析和可视化

文件命名规范：
- JSON文件名应包含"io{input_len}x{output_len}"格式的参数信息
- 例如：bench_io256x256_mc32_np128.json 表示输入长度256，输出长度256

依赖要求：pip install pandas

作者：vLLM团队
修改：添加详细中文注释
"""

import glob
import json
import pandas as pd
import os
import re

# 结果文件存储目录
RESULT_DIR = "results"
# 聚合结果输出CSV文件路径
OUT_CSV = os.path.join(RESULT_DIR, "aggregate_results.csv")

def parse_input_output_lengths(filename):
    """
    从文件名中解析输入和输出长度参数
    
    解析基准测试结果文件名中的输入输出长度信息。
    文件名应遵循包含"io{input_len}x{output_len}"模式的命名规范。
    
    Args:
        filename: 基准测试结果文件名
        
    Returns:
        tuple[int, int]: (输入长度, 输出长度)，如果解析失败则返回(None, None)
        
    Example:
        bench_io256x256_mc32_np128.json → input_len=256, output_len=256
    """
    match = re.search(r"io(\d+)x(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def main():
    """
    主函数：聚合所有基准测试结果文件
    
    执行以下步骤来聚合基准测试结果：
    1. 扫描results目录下的所有JSON文件
    2. 逐个读取JSON文件并解析文件名中的参数
    3. 将所有数据合并到pandas DataFrame中
    4. 导出为CSV格式供后续分析使用
    """
    # 1) 获取results目录下所有JSON文件的路径列表
    json_paths = glob.glob(os.path.join(RESULT_DIR, "*.json"))
    if not json_paths:
        print("在results/目录中没有找到JSON文件。")
        return

    # 2) 读取每个JSON文件并添加输入/输出长度信息
    records = []
    for p in json_paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        filename = os.path.basename(p)
        input_len, output_len = parse_input_output_lengths(filename)
        # 添加解析出的参数到数据中
        data["input_len"] = input_len
        data["output_len"] = output_len
        data["filename"] = filename

        records.append(data)

    # 3) 创建pandas DataFrame并导出为CSV文件
    df = pd.json_normalize(records)  # 将嵌套的JSON结构展平
    df.to_csv(OUT_CSV, index=False)
    print(f"已聚合 {len(records)} 个测试运行结果 → {OUT_CSV}")

if __name__ == "__main__":
    main()
