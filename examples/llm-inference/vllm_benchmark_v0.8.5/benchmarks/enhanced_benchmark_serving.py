#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
增强版在线服务基准测试脚本

这是vLLM基准测试的增强版本，在原有功能基础上添加了：
1. YAML配置文件支持 - 简化参数管理
2. 自动可视化生成 - 图表和HTML报告
3. 详细的中文注释 - 便于理解和学习
4. 结构化的结果保存 - 便于后续分析

主要功能：
- 支持多种数据集（ShareGPT、Random、Sonnet等）
- 支持多种后端（vLLM、OpenAI、TGI等）
- 自动生成性能分析图表
- 生成HTML格式的详细报告
- 保持与原版脚本的完全兼容性

使用方法：
1. 配置文件方式: python enhanced_benchmark_serving.py --config config.yaml
2. 命令行方式: python enhanced_benchmark_serving.py --model xxx --dataset-name xxx
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional

# 添加当前目录到Python路径，确保能导入本地模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入增强功能模块
from config_manager import ConfigManager, BenchmarkConfig
from visualization import BenchmarkVisualizer, create_html_report

# 导入原始基准测试模块，保持兼容性
from benchmark_serving import (
    BenchmarkMetrics, get_request, send_request, calculate_metrics,
    ASYNC_REQUEST_FUNCS
)
from benchmark_dataset import (
    ShareGPTDataset, RandomDataset, SonnetDataset, BurstGPTDataset,
    HuggingFaceDataset, VisionArenaDataset, InstructCoderDataset, AIMODataset
)

# 尝试导入vLLM的tokenizer，如果失败则使用备用方案
try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer


class EnhancedBenchmarkRunner:
    """
    增强版基准测试运行器

    这个类是整个增强版基准测试的核心，负责：
    1. 管理测试配置和参数
    2. 协调数据集加载和请求生成
    3. 执行异步基准测试
    4. 收集和分析性能指标
    5. 生成可视化图表和报告
    6. 保存和管理测试结果

    相比原版脚本，增加了配置管理、可视化和结果结构化等功能，
    同时保持了与原版的完全兼容性。
    """

    def __init__(self, config: BenchmarkConfig):
        """
        初始化增强版基准测试运行器

        Args:
            config: 基准测试配置对象，包含所有测试参数
        """
        self.config = config                                    # 保存测试配置
        self.visualizer = BenchmarkVisualizer(config.result_dir)  # 初始化可视化器
        self.results = {}                                       # 存储测试结果

    def setup_result_directory(self) -> str:
        """
        设置和创建结果保存目录

        根据配置创建结果目录，支持自定义文件名或自动生成时间戳。
        目录结构便于组织和管理多次测试的结果。

        Returns:
            str: 创建的结果目录路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 如果指定了结果文件名，使用该名称作为目录名
        if self.config.result_filename:
            result_dir = os.path.join(self.config.result_dir,
                                    self.config.result_filename.replace('.json', ''))
        else:
            # 否则使用时间戳生成唯一的目录名
            result_dir = os.path.join(self.config.result_dir, f"serving_test_{timestamp}")

        # 确保目录存在
        os.makedirs(result_dir, exist_ok=True)
        return result_dir
    
    def load_dataset(self):
        """
        加载和准备测试数据集

        这个方法负责：
        1. 根据配置选择合适的数据集类
        2. 初始化数据集并加载数据
        3. 获取对应模型的tokenizer
        4. 采样指定数量的测试请求
        5. 返回处理好的请求列表和tokenizer

        支持的数据集类型：
        - sharegpt: ShareGPT对话数据集
        - random: 随机生成的测试数据
        - sonnet: 诗歌文本数据集
        - burstgpt: BurstGPT数据集
        - hf: HuggingFace数据集
        - vision_arena: 视觉对话数据集
        - instruct_coder: 代码指令数据集
        - aimo: AIMO数学数据集

        Returns:
            tuple: (请求列表, tokenizer对象)
        """
        print(f"正在加载数据集: {self.config.dataset_name}")

        # 数据集类型到类的映射
        dataset_classes = {
            "sharegpt": ShareGPTDataset,           # ShareGPT对话数据
            "random": RandomDataset,               # 随机生成数据
            "sonnet": SonnetDataset,               # 诗歌文本数据
            "burstgpt": BurstGPTDataset,           # BurstGPT数据
            "hf": HuggingFaceDataset,              # HuggingFace数据集
            "vision_arena": VisionArenaDataset,    # 视觉对话数据
            "instruct_coder": InstructCoderDataset, # 代码指令数据
            "aimo": AIMODataset                    # AIMO数学数据
        }

        # 检查数据集类型是否支持
        if self.config.dataset_name not in dataset_classes:
            raise ValueError(f"不支持的数据集: {self.config.dataset_name}")

        # 创建数据集实例
        dataset_class = dataset_classes[self.config.dataset_name]
        dataset = dataset_class(
            dataset_path=self.config.dataset_path,  # 数据集文件路径
            random_seed=self.config.seed           # 随机种子，确保结果可重现
        )

        # 加载数据集内容
        dataset.load_data()

        # 获取模型对应的tokenizer，用于文本编码
        tokenizer = get_tokenizer(
            self.config.model,
            trust_remote_code=self.config.trust_remote_code
        )

        # 从数据集中采样指定数量的请求
        requests = dataset.sample_requests(
            num_requests=self.config.num_prompts,
            tokenizer=tokenizer,
            # 对于random数据集，使用固定的输出长度
            fixed_output_len=self.config.output_len if self.config.dataset_name == "random" else None
        )

        print(f"成功加载 {len(requests)} 个请求")
        return requests, tokenizer
    
    async def run_benchmark(self) -> Dict[str, Any]:
        """运行基准测试"""
        print("=" * 60)
        print("开始运行增强版vLLM在线服务基准测试")
        print("=" * 60)
        
        # 打印配置
        config_manager = ConfigManager()
        config_manager.config = self.config
        config_manager.print_config()
        
        # 加载数据集
        requests, tokenizer = self.load_dataset()
        
        # 构建API URL
        api_url = f"http://{self.config.host}:{self.config.port}{self.config.endpoint}"
        print(f"API端点: {api_url}")
        
        # 准备请求参数
        request_func_input_list = []
        for request in requests:
            request_func_input_list.append({
                "prompt": request.prompt,
                "api_url": api_url,
                "prompt_len": request.prompt_len,
                "output_len": request.expected_output_len,
                "model": self.config.model,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "max_tokens": self.config.max_tokens or request.expected_output_len,
                "multi_modal_content": request.multi_modal_data
            })
        
        # 获取请求函数
        if self.config.backend not in ASYNC_REQUEST_FUNCS:
            raise ValueError(f"不支持的后端: {self.config.backend}")
        
        request_func = ASYNC_REQUEST_FUNCS[self.config.backend]
        
        print(f"开始发送 {len(request_func_input_list)} 个请求...")
        print(f"请求速率: {self.config.request_rate} req/s")
        print(f"突发性: {self.config.burstiness}")
        
        # 运行基准测试
        benchmark_start_time = time.perf_counter()
        
        # 创建请求生成器
        input_requests = [
            {"prompt": req["prompt"], "prompt_len": req["prompt_len"], 
             "expected_output_len": req["output_len"]}
            for req in request_func_input_list
        ]
        
        request_generator = get_request(
            input_requests, self.config.request_rate, self.config.burstiness
        )
        
        # 发送请求并收集结果
        tasks = []
        async for request in request_generator:
            # 找到对应的请求参数
            req_input = next(
                (r for r in request_func_input_list 
                 if r["prompt"] == request.prompt), 
                request_func_input_list[0]
            )
            
            task = asyncio.create_task(request_func(req_input))
            tasks.append(task)
        
        # 等待所有请求完成
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        
        benchmark_end_time = time.perf_counter()
        benchmark_duration = benchmark_end_time - benchmark_start_time
        
        print(f"基准测试完成，耗时: {benchmark_duration:.2f}秒")
        
        # 计算指标
        successful_requests = [o for o in outputs if not isinstance(o, Exception) and o.success]
        
        if not successful_requests:
            raise RuntimeError("没有成功的请求，请检查服务器状态和配置")
        
        # 提取指标数据
        ttft_ms = [r.ttft * 1000 for r in successful_requests if r.ttft > 0]
        tpot_ms = [r.tpot * 1000 for r in successful_requests if r.tpot > 0]
        itl_ms = []
        for r in successful_requests:
            if r.itl:
                itl_ms.extend([itl * 1000 for itl in r.itl])
        
        total_input_tokens = sum(r.prompt_len for r in successful_requests)
        total_output_tokens = sum(r.output_tokens for r in successful_requests)
        
        # 计算统计指标
        metrics = {
            "completed": len(successful_requests),
            "total_input": total_input_tokens,
            "total_output": total_output_tokens,
            "request_throughput": len(successful_requests) / benchmark_duration,
            "output_throughput": total_output_tokens / benchmark_duration,
            "total_token_throughput": (total_input_tokens + total_output_tokens) / benchmark_duration,
            "benchmark_duration_s": benchmark_duration,
            "mean_ttft_ms": sum(ttft_ms) / len(ttft_ms) if ttft_ms else 0,
            "median_ttft_ms": sorted(ttft_ms)[len(ttft_ms)//2] if ttft_ms else 0,
            "p90_ttft_ms": sorted(ttft_ms)[int(len(ttft_ms)*0.9)] if ttft_ms else 0,
            "p95_ttft_ms": sorted(ttft_ms)[int(len(ttft_ms)*0.95)] if ttft_ms else 0,
            "p99_ttft_ms": sorted(ttft_ms)[int(len(ttft_ms)*0.99)] if ttft_ms else 0,
            "mean_tpot_ms": sum(tpot_ms) / len(tpot_ms) if tpot_ms else 0,
            "median_tpot_ms": sorted(tpot_ms)[len(tpot_ms)//2] if tpot_ms else 0,
            "p90_tpot_ms": sorted(tpot_ms)[int(len(tpot_ms)*0.9)] if tpot_ms else 0,
            "p95_tpot_ms": sorted(tpot_ms)[int(len(tpot_ms)*0.95)] if tpot_ms else 0,
            "p99_tpot_ms": sorted(tpot_ms)[int(len(tpot_ms)*0.99)] if tpot_ms else 0,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "itl_ms": itl_ms
        }
        
        # 保存结果
        result_data = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.__dict__,
            "metrics": metrics,
            "raw_outputs": [
                {
                    "success": r.success,
                    "latency": r.latency,
                    "ttft": r.ttft,
                    "tpot": r.tpot,
                    "output_tokens": r.output_tokens,
                    "prompt_len": r.prompt_len
                } for r in successful_requests
            ]
        }
        
        return result_data
    
    def save_and_visualize_results(self, results: Dict[str, Any]) -> None:
        """保存结果并生成可视化"""
        result_dir = self.setup_result_directory()
        
        # 保存JSON结果
        json_path = os.path.join(result_dir, "results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {json_path}")
        
        # 生成可视化
        if self.config.enable_visualization:
            print("正在生成可视化图表...")
            
            try:
                # 生成图表
                chart_path = self.visualizer.visualize_serving_results(
                    results["metrics"], 
                    results["config"],
                    os.path.join(result_dir, "serving_results.png")
                )
                print(f"图表已保存到: {chart_path}")
                
                # 生成HTML报告
                html_path = create_html_report(
                    results, 
                    [chart_path],
                    os.path.join(result_dir, "report.html")
                )
                print(f"HTML报告已保存到: {html_path}")
                
            except Exception as e:
                print(f"可视化生成失败: {e}")
                print("请确保已安装matplotlib和seaborn: pip install matplotlib seaborn")
        
        # 打印文本结果
        self.print_results(results["metrics"])
    
    def print_results(self, metrics: Dict[str, Any]) -> None:
        """打印测试结果"""
        print("\n" + "=" * 60)
        print("基准测试结果")
        print("=" * 60)
        print(f"成功请求数:                    {metrics['completed']}")
        print(f"测试持续时间 (s):               {metrics['benchmark_duration_s']:.2f}")
        print(f"总输入tokens:                  {metrics['total_input']}")
        print(f"总输出tokens:                  {metrics['total_output']}")
        print(f"请求吞吐量 (req/s):             {metrics['request_throughput']:.2f}")
        print(f"输出token吞吐量 (tok/s):        {metrics['output_throughput']:.2f}")
        print(f"总token吞吐量 (tok/s):          {metrics['total_token_throughput']:.2f}")
        print("-" * 60)
        print("首Token时间 (TTFT)")
        print("-" * 60)
        print(f"平均 TTFT (ms):                {metrics['mean_ttft_ms']:.2f}")
        print(f"中位数 TTFT (ms):              {metrics['median_ttft_ms']:.2f}")
        print(f"P90 TTFT (ms):                 {metrics['p90_ttft_ms']:.2f}")
        print(f"P95 TTFT (ms):                 {metrics['p95_ttft_ms']:.2f}")
        print(f"P99 TTFT (ms):                 {metrics['p99_ttft_ms']:.2f}")
        print("-" * 60)
        print("每Token时间 (TPOT)")
        print("-" * 60)
        print(f"平均 TPOT (ms):                {metrics['mean_tpot_ms']:.2f}")
        print(f"中位数 TPOT (ms):              {metrics['median_tpot_ms']:.2f}")
        print(f"P90 TPOT (ms):                 {metrics['p90_tpot_ms']:.2f}")
        print(f"P95 TPOT (ms):                 {metrics['p95_tpot_ms']:.2f}")
        print(f"P99 TPOT (ms):                 {metrics['p99_tpot_ms']:.2f}")
        print("=" * 60)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="增强版vLLM在线服务基准测试")
    
    # 配置文件参数
    parser.add_argument("--config", type=str, help="YAML配置文件路径")
    
    # 基本参数
    parser.add_argument("--model", type=str, help="模型名称或路径")
    parser.add_argument("--backend", type=str, choices=["vllm", "openai", "openai-chat", "tgi"], help="后端类型")
    parser.add_argument("--endpoint", type=str, help="API端点")
    parser.add_argument("--host", type=str, default="localhost", help="服务器主机")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    
    # 数据集参数
    parser.add_argument("--dataset-name", type=str, help="数据集名称")
    parser.add_argument("--dataset-path", type=str, help="数据集路径")
    parser.add_argument("--num-prompts", type=int, help="请求数量")
    
    # 测试参数
    parser.add_argument("--request-rate", type=float, help="请求速率 (req/s)")
    parser.add_argument("--seed", type=int, help="随机种子")
    
    # 输出参数
    parser.add_argument("--result-dir", type=str, help="结果保存目录")
    parser.add_argument("--no-visualization", action="store_true", help="禁用可视化")
    
    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_arguments()
    
    # 创建配置管理器
    config_manager = ConfigManager()
    
    # 加载配置文件
    if args.config:
        if not os.path.exists(args.config):
            print(f"错误: 配置文件不存在: {args.config}")
            return
        config_manager.load_from_yaml(args.config)
        print(f"已加载配置文件: {args.config}")
    
    # 合并命令行参数
    config_manager.merge_with_args(args)
    
    # 设置可视化选项
    if args.no_visualization:
        config_manager.config.enable_visualization = False
    
    # 创建并运行基准测试
    runner = EnhancedBenchmarkRunner(config_manager.get_config())
    
    try:
        results = await runner.run_benchmark()
        runner.save_and_visualize_results(results)
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
