#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  # 使用所有8张卡

# 测试配置
MODELS=(
    "deepseek-ai/deepseek-llm-7b-base"
    "Qwen/Qwen-7B"
    "THUDM/chatglm3-6b"
)

FRAMEWORKS=(
    "vllm"
    "xinference"
)

# 创建结果目录
RESULTS_DIR="benchmark_results"
mkdir -p $RESULTS_DIR

# 获取当前时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 记录系统信息
echo "系统信息:" > "$RESULTS_DIR/system_info_$TIMESTAMP.txt"
echo "===================" >> "$RESULTS_DIR/system_info_$TIMESTAMP.txt"
echo "操作系统:" >> "$RESULTS_DIR/system_info_$TIMESTAMP.txt"
uname -a >> "$RESULTS_DIR/system_info_$TIMESTAMP.txt"
echo "" >> "$RESULTS_DIR/system_info_$TIMESTAMP.txt"

echo "CPU信息:" >> "$RESULTS_DIR/system_info_$TIMESTAMP.txt"
lscpu >> "$RESULTS_DIR/system_info_$TIMESTAMP.txt"
echo "" >> "$RESULTS_DIR/system_info_$TIMESTAMP.txt"

echo "内存信息:" >> "$RESULTS_DIR/system_info_$TIMESTAMP.txt"
free -h >> "$RESULTS_DIR/system_info_$TIMESTAMP.txt"
echo "" >> "$RESULTS_DIR/system_info_$TIMESTAMP.txt"

echo "GPU信息:" >> "$RESULTS_DIR/system_info_$TIMESTAMP.txt"
rocm-smi >> "$RESULTS_DIR/system_info_$TIMESTAMP.txt"
echo "" >> "$RESULTS_DIR/system_info_$TIMESTAMP.txt"

# 运行测试
for model in "${MODELS[@]}"; do
    for framework in "${FRAMEWORKS[@]}"; do
        echo "开始测试: $model on $framework"
        
        # 构建输出文件名
        model_name=$(echo $model | tr '/' '_')
        output_file="$RESULTS_DIR/${model_name}_${framework}_${TIMESTAMP}.json"
        
        # 运行测试
        python benchmark_test.py \
            --model-path $model \
            --model-name $model \
            --framework $framework \
            --batch-sizes 1 4 8 16 32 \
            --sequence-lengths 128 256 512 1024 2048 \
            --num-iterations 5 \
            --output $output_file
        
        # 检查测试是否成功
        if [ $? -eq 0 ]; then
            echo "✅ 测试完成: $model on $framework"
        else
            echo "❌ 测试失败: $model on $framework"
        fi
        
        # 等待GPU冷却
        sleep 30
    done
done

# 生成测试报告
echo "生成测试报告..."
python generate_report.py \
    --results-dir $RESULTS_DIR \
    --timestamp $TIMESTAMP \
    --output "$RESULTS_DIR/benchmark_report_$TIMESTAMP.md"

echo "测试完成！"
echo "结果保存在: $RESULTS_DIR" 