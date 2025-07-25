#!/bin/bash
# =============================================================================
# 海光DCU大模型推理性能基准测试脚本
# =============================================================================
#
# 功能说明:
#   这个脚本用于在海光DCU环境下对大模型进行全面的推理性能测试
#   支持多种批次大小和输入输出长度组合的性能评估
#
# 测试场景:
#   - 不同批次大小: 1, 2, 4, 6, 8, 10, 16, 20, 24, 32, 64
#   - 不同输入输出长度组合: (512,512), (1024,1024)
#   - 使用8卡DCU张量并行推理
#
# 输出结果:
#   - 详细日志文件: ./log/目录下
#   - 汇总CSV文件: r1-awq-0705.csv
#   - 包含吞吐量、延迟等关键性能指标
# =============================================================================

# DCU设备配置 - 指定使用所有8个DCU设备
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 创建CSV结果文件并写入表头
# 包含所有关键性能指标的列名
echo "tp,data_type,batch,prompt_tokens,completion_tokens,TOTAL_THROUGHPUT(toks/s),generate_throughput(toks/s), TTFT(ms),TPOT(ms),ITL(ms)" > r1-awq-0705.csv

# 测试参数配置
pairs=( "512 512" "1024 1024" )                                    # 输入输出长度组合 (prompt_tokens completion_tokens)
model_path="/data/model/cognitivecomputations/DeepSeek-R1-awq/"    # 模型路径 - DeepSeek-R1 AWQ量化版本
tp=8                                                                # 张量并行度 - 使用8卡DCU
data_type="float16"                                                 # 数据类型 - FP16精度
port=8010
# 创建日志目录
mkdir -p ./log/
# =============================================================================
# 主测试循环 - 遍历所有测试配置组合
# =============================================================================
for batch in 1 2 4 6 8 10 16 20 24 32 64; do                      # 批次大小循环
    for pair in "${pairs[@]}"; do                                  # 输入输出长度组合循环
        # 解析输入输出长度参数
        prompt_tokens=${pair%% *}                                   # 提取输入token数量
        completion_tokens=${pair#* }                                # 提取输出token数量

        # 打印当前测试配置
        echo "data_type: $data_type,batch: $batch, prompt_tokens: $prompt_tokens, completion_tokens: $completion_tokens, tp: ${tp}"

        # 生成日志文件路径
        log_path="log/vllm_${model}_batch_${batch}_prompt_tokens_${prompt_tokens}_completion_tokens_${completion_tokens}_tp_${tp}.log"
        touch $log_path

        # =================================================================
        # 执行基准测试 - 调用benchmark_serving.py进行性能测试
        # =================================================================
        python benchmark_serving.py \
                --backend openai \                                  # 使用OpenAI兼容API (vLLM)
                --port ${port} \                                    # vLLM服务端口
                --model ${model_path} \                             # 模型路径
                --trust-remote-code \                               # 信任远程代码 (某些模型需要)
                --dataset-name random \                             # 使用随机生成的测试数据
                --ignore-eos \                                      # 忽略结束符，强制生成指定长度
                --random-input-len ${prompt_tokens} \               # 随机输入长度
                --random-output-len ${completion_tokens} \          # 随机输出长度
                --num-prompts ${batch}  \                           # 请求数量 (等于批次大小)
                2>&1 | tee  $log_path                               # 输出重定向到日志文件
        # =================================================================
        # 性能指标提取 - 从日志文件中解析关键性能数据
        # =================================================================

        # 基础性能指标提取
        E2E_TIME=`grep "^Benchmark duration" $log_path | awk -F ' ' '{print $4}'`        # 端到端测试时间 (秒)
        REQ_THROUGHPUT=`grep "^Request throughput"  $log_path| awk -F ' ' '{print $4}'`  # 请求吞吐量 (req/s)
        GEN_THROUGHPUT=`grep "^Output token"  $log_path| awk -F ' ' '{print $5}'`        # 生成吞吐量 (tok/s)
        TOTAL_THROUGHPUT=`grep "^Total Token" $log_path| awk -F ' ' '{print $5}'`        # 总token吞吐量 (tok/s)

        # 延迟指标提取 (平均值)
        TTFT=`grep "^Mean TTFT"  $log_path| awk -F ' ' '{print $4}'`                     # 平均首次token时间 (ms)
        TPOT=`grep "^Mean TPOT"  $log_path| awk -F ' ' '{print $4}'`                     # 平均每token时间 (ms)
        ITL=`grep "^Mean ITL"  $log_path| awk -F ' ' '{print $4}'`                       # 平均迭代延迟 (ms)

        # 延迟指标提取 (99百分位数) - 用于评估最坏情况性能
        P99_ITL=`grep "^P99 ITL"  $log_path| awk -F ' ' '{print $4}'`                    # P99迭代延迟 (ms)
        P99_TTFT=`grep "^P99 TTFT"  $log_path| awk -F ' ' '{print $4}'`                  # P99首次token时间 (ms)
        P99_TPOT=`grep "^P99 TPOT"  $log_path| awk -F ' ' '{print $4}'`                  # P99每token时间 (ms)

        # 将结果写入CSV文件 - 格式化输出所有关键指标
        echo "$tp,$data_type,$batch,$prompt_tokens,$completion_tokens,$TOTAL_THROUGHPUT,$GEN_THROUGHPUT,$TTFT,$TPOT, $ITL" >> r1-awq-0705.csv
    done
done

# =============================================================================
# 测试完成提示
# =============================================================================
echo "基准测试完成！"
echo "结果文件: r1-awq-0705.csv"
echo "详细日志: ./log/ 目录"
