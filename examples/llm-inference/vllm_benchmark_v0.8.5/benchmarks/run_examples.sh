#!/bin/bash
# vLLM增强版基准测试示例脚本

set -e

echo "=========================================="
echo "vLLM 0.8.5 增强版基准测试示例"
echo "=========================================="

# 检查Python环境
echo "检查Python环境..."
python -c "import yaml, matplotlib, seaborn, pandas, numpy" 2>/dev/null || {
    echo "错误: 缺少必要的Python包"
    echo "请运行: pip install pyyaml matplotlib seaborn pandas numpy"
    exit 1
}

echo "✅ Python环境检查通过"

# 检查vLLM服务
echo "检查vLLM服务状态..."
curl -s http://localhost:8000/health >/dev/null 2>&1 || {
    echo "⚠️  警告: vLLM服务未运行"
    echo "请先启动vLLM服务:"
    echo "vllm serve meta-llama/Llama-2-7b-hf --disable-log-requests"
    echo ""
    echo "继续运行示例（将使用模拟数据）..."
}

# 创建示例数据目录
mkdir -p ./example_results
mkdir -p ./example_data

echo ""
echo "=========================================="
echo "示例1: 使用配置文件运行基本测试"
echo "=========================================="

# 创建示例配置
cat > ./example_data/basic_test.yaml << EOF
# 基本测试配置
model: "meta-llama/Llama-2-7b-hf"
backend: "vllm"
endpoint: "/v1/completions"
host: "localhost"
port: 8000

# 使用随机数据集（不需要下载）
dataset_name: "random"
num_prompts: 20
input_len: 512
output_len: 64

# 测试参数
request_rate: 1.0
seed: 42

# 输出配置
save_result: true
result_dir: "./example_results/basic_test"
enable_visualization: true
EOF

echo "配置文件已创建: ./example_data/basic_test.yaml"
echo "运行命令:"
echo "python enhanced_benchmark_serving.py --config ./example_data/basic_test.yaml"
echo ""

echo "=========================================="
echo "示例2: 性能对比测试"
echo "=========================================="

# 创建对比测试脚本
cat > ./example_data/comparison_test.sh << 'EOF'
#!/bin/bash
echo "运行性能对比测试..."

# 测试不同请求速率的性能
for rate in 0.5 1.0 2.0 5.0; do
    echo "测试请求速率: ${rate} req/s"
    python enhanced_benchmark_serving.py \
        --model "meta-llama/Llama-2-7b-hf" \
        --backend "vllm" \
        --dataset-name "random" \
        --num-prompts 50 \
        --input-len 256 \
        --output-len 32 \
        --request-rate $rate \
        --result-dir "./example_results/rate_${rate}" \
        --seed 42
done

echo "对比测试完成！结果保存在 ./example_results/ 目录"
EOF

chmod +x ./example_data/comparison_test.sh
echo "对比测试脚本已创建: ./example_data/comparison_test.sh"
echo ""

echo "=========================================="
echo "示例3: 不同数据集测试"
echo "=========================================="

# 创建数据集测试配置
cat > ./example_data/dataset_test.yaml << EOF
# 数据集测试配置
model: "meta-llama/Llama-2-7b-hf"
backend: "vllm"
endpoint: "/v1/completions"

# 基本参数
num_prompts: 30
request_rate: 1.5
seed: 42

# 输出配置
save_result: true
enable_visualization: true
EOF

echo "数据集测试配置已创建: ./example_data/dataset_test.yaml"
echo ""
echo "可用的数据集类型:"
echo "1. random - 随机生成的数据（推荐用于测试）"
echo "2. sonnet - 使用内置的诗歌文本"
echo "3. sharegpt - ShareGPT对话数据（需要下载）"
echo ""
echo "运行示例:"
echo "python enhanced_benchmark_serving.py --config ./example_data/dataset_test.yaml --dataset-name random --result-dir ./example_results/random_dataset"
echo "python enhanced_benchmark_serving.py --config ./example_data/dataset_test.yaml --dataset-name sonnet --result-dir ./example_results/sonnet_dataset"
echo ""

echo "=========================================="
echo "示例4: 可视化功能演示"
echo "=========================================="

# 创建可视化演示脚本
cat > ./example_data/visualization_demo.py << 'EOF'
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
EOF

chmod +x ./example_data/visualization_demo.py
echo "可视化演示脚本已创建: ./example_data/visualization_demo.py"
echo "运行命令: python ./example_data/visualization_demo.py"
echo ""

echo "=========================================="
echo "示例5: 完整的性能分析流程"
echo "=========================================="

# 创建完整分析脚本
cat > ./example_data/full_analysis.sh << 'EOF'
#!/bin/bash
echo "开始完整的性能分析流程..."

# 1. 基线测试
echo "1. 运行基线测试..."
python enhanced_benchmark_serving.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --backend "vllm" \
    --dataset-name "random" \
    --num-prompts 100 \
    --input-len 512 \
    --output-len 128 \
    --request-rate inf \
    --result-dir "./example_results/analysis/baseline" \
    --seed 42

# 2. 负载测试
echo "2. 运行负载测试..."
for rate in 1 2 5 10; do
    echo "  测试负载: ${rate} req/s"
    python enhanced_benchmark_serving.py \
        --model "meta-llama/Llama-2-7b-hf" \
        --backend "vllm" \
        --dataset-name "random" \
        --num-prompts 50 \
        --input-len 512 \
        --output-len 128 \
        --request-rate $rate \
        --result-dir "./example_results/analysis/load_${rate}" \
        --seed 42
done

# 3. 输入长度影响测试
echo "3. 测试输入长度影响..."
for len in 256 512 1024 2048; do
    echo "  测试输入长度: ${len}"
    python enhanced_benchmark_serving.py \
        --model "meta-llama/Llama-2-7b-hf" \
        --backend "vllm" \
        --dataset-name "random" \
        --num-prompts 30 \
        --input-len $len \
        --output-len 128 \
        --request-rate 2.0 \
        --result-dir "./example_results/analysis/input_${len}" \
        --seed 42
done

echo "✅ 完整性能分析完成！"
echo "📁 结果保存在: ./example_results/analysis/"
echo "📊 查看各个子目录中的HTML报告获取详细结果"
EOF

chmod +x ./example_data/full_analysis.sh
echo "完整分析脚本已创建: ./example_data/full_analysis.sh"
echo ""

echo "=========================================="
echo "使用说明"
echo "=========================================="
echo ""
echo "🚀 快速开始:"
echo "1. 确保vLLM服务运行: vllm serve meta-llama/Llama-2-7b-hf --disable-log-requests"
echo "2. 运行基本测试: python enhanced_benchmark_serving.py --config ./example_data/basic_test.yaml"
echo "3. 查看结果: 打开 ./example_results/basic_test/report.html"
echo ""
echo "📊 可视化演示 (无需vLLM服务):"
echo "python ./example_data/visualization_demo.py"
echo ""
echo "🔍 性能分析:"
echo "bash ./example_data/comparison_test.sh"
echo "bash ./example_data/full_analysis.sh"
echo ""
echo "📚 详细文档:"
echo "查看 ENHANCED_README.md 获取完整使用指南"
echo ""
echo "=========================================="
echo "示例脚本创建完成！"
echo "=========================================="
