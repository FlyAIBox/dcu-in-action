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
