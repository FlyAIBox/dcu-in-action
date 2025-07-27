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
