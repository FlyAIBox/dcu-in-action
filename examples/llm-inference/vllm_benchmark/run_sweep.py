#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_sweep.py - 基准测试批量运行脚本

本模块提供了批量执行基准测试的功能，支持多参数组合的自动化测试。
主要功能包括：
1. 从YAML配置文件读取测试参数组合
2. 自动生成不同参数组合的基准测试命令
3. 批量执行测试并保存结果到指定目录
4. 提供错误处理和进度反馈

目录结构：
├── combos.yaml      # YAML格式的参数组合定义文件
└── run_sweep.py     # 批量执行基准测试的驱动脚本

配置文件说明：
- combos.yaml包含模型、服务URL、分词器等基础配置
- 以及输入/输出长度、并发数、提示数量等参数组合

"""

import yaml
import subprocess
import os

# benchmark_serving.py脚本的路径
BENCHMARK_SCRIPT = "benchmark_serving.py"
# YAML配置文件路径
CONFIG_FILE = "combos.yaml"


def run_benchmark(common_args, input_len, output_len, concurrency, num_prompts):
    """
    运行单个参数组合的基准测试
    
    根据给定的参数组合构建完整的基准测试命令并执行，
    将结果保存到指定的JSON文件中。
    
    Args:
        common_args: 公共命令行参数列表
        input_len: 输入token长度
        output_len: 输出token长度  
        concurrency: 并发请求数
        num_prompts: 提示数量
        
    Note:
        结果文件命名格式：bench_io{input_len}x{output_len}_mc{concurrency}_np{num_prompts}.json
    """
    args = common_args.copy()
    # 添加输入/输出token长度参数
    args += ["--random-input-len", str(input_len), "--random-output-len", str(output_len)]
    # 设置并发数和提示数量
    args += ["--max-concurrency", str(concurrency)]
    args += ["--num-prompts", str(num_prompts)]

    # 创建结果保存目录
    result_dir = "results"
    os.makedirs(result_dir, exist_ok=True)
    outfile = os.path.join(
        result_dir,
        f"bench_io{input_len}x{output_len}_mc{concurrency}_np{num_prompts}.json"
    )
    args += ["--save-result", "--result-filename", outfile]

    print(f"正在运行: {' '.join(args)}")
    ret = subprocess.run(args, capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"参数组合 io=({input_len},{output_len}), mc={concurrency}, np={num_prompts} 执行出错: {ret.stderr}")
    else:
        print(f"参数组合 io=({input_len},{output_len}), mc={concurrency}, np={num_prompts} 执行完成，结果已保存: {outfile}")


def main():
    """
    主函数：执行批量基准测试
    
    从YAML配置文件加载参数组合，然后对所有参数组合执行
    笛卡尔积操作，生成完整的测试矩阵并依次执行。
    
    处理流程：
    1. 从combos.yaml加载基础配置和参数组合
    2. 构建公共命令行参数
    3. 对所有输入/输出长度和并发/提示数量组合执行测试
    """
    # 从YAML文件加载配置和参数列表
    with open(CONFIG_FILE, "r") as f:
        cfg = yaml.safe_load(f)
    model = cfg["model"]
    base_url = cfg["base_url"]
    tokenizer = cfg["tokenizer"]
    io_pairs = cfg.get("input_output", [])           # 输入/输出长度对列表
    cp_pairs = cfg.get("concurrency_prompts", [])   # 并发数/提示数量对列表

    # 构建公共命令行参数
    common_args = [
        "python3", BENCHMARK_SCRIPT,
        "--backend", "vllm",
        "--model", model,
        "--base-url", base_url,
        "--tokenizer", tokenizer,
        "--dataset-name", "random",                   # 使用随机数据集
        "--percentile-metrics", "ttft,tpot,itl,e2el" # 计算关键指标的百分位数
    ]

    # 执行交叉组合测试（每个io_pair与所有concurrency-num_prompts对组合）
    for input_len, output_len in io_pairs:
        for concurrency, num_prompts in cp_pairs:
            run_benchmark(common_args, input_len, output_len, concurrency, num_prompts)


if __name__ == "__main__":
    main()
