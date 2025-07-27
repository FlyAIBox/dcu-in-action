#!/usr/bin/env python3
"""
可视化功能演示脚本
生成模拟数据并展示可视化功能
"""

import json
import os
import sys
import numpy as np
from datetime import datetime

# 添加当前目录到路径
sys.path.append('.')

from visualization import BenchmarkVisualizer, create_html_report

def generate_mock_data():
    """生成模拟测试数据"""
    np.random.seed(42)
    
    # 模拟性能指标
    num_requests = 100
    ttft_ms = np.random.gamma(2, 50, num_requests).tolist()  # 首token时间
    tpot_ms = np.random.gamma(1.5, 20, num_requests).tolist()  # 每token时间
    
    metrics = {
        "completed": num_requests,
        "total_input": 25600,
        "total_output": 6400,
        "request_throughput": 8.5,
        "output_throughput": 425.6,
        "total_token_throughput": 1067.2,
        "benchmark_duration_s": 11.76,
        "mean_ttft_ms": np.mean(ttft_ms),
        "median_ttft_ms": np.median(ttft_ms),
        "p90_ttft_ms": np.percentile(ttft_ms, 90),
        "p95_ttft_ms": np.percentile(ttft_ms, 95),
        "p99_ttft_ms": np.percentile(ttft_ms, 99),
        "mean_tpot_ms": np.mean(tpot_ms),
        "median_tpot_ms": np.median(tpot_ms),
        "p90_tpot_ms": np.percentile(tpot_ms, 90),
        "p95_tpot_ms": np.percentile(tpot_ms, 95),
        "p99_tpot_ms": np.percentile(tpot_ms, 99),
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms
    }
    
    config = {
        "model": "meta-llama/Llama-2-7b-hf",
        "backend": "vllm",
        "dataset_name": "random",
        "num_prompts": num_requests,
        "request_rate": 2.0
    }
    
    return {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": metrics
    }

def main():
    print("生成可视化演示...")
    
    # 创建结果目录
    os.makedirs("./example_results/visualization_demo", exist_ok=True)
    
    # 生成模拟数据
    mock_results = generate_mock_data()
    
    # 保存模拟数据
    with open("./example_results/visualization_demo/mock_results.json", "w") as f:
        json.dump(mock_results, f, indent=2)
    
    # 创建可视化器
    visualizer = BenchmarkVisualizer("./example_results/visualization_demo")
    
    # 生成图表
    chart_path = visualizer.visualize_serving_results(
        mock_results["metrics"],
        mock_results["config"],
        "./example_results/visualization_demo/demo_chart.png"
    )
    
    # 生成HTML报告
    html_path = create_html_report(
        mock_results,
        [chart_path],
        "./example_results/visualization_demo/demo_report.html"
    )
    
    print(f"✅ 可视化演示完成!")
    print(f"📊 图表: {chart_path}")
    print(f"📄 HTML报告: {html_path}")
    print(f"📁 结果目录: ./example_results/visualization_demo/")

if __name__ == "__main__":
    main()
