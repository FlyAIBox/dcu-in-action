# 🛠️ vLLM基准测试框架实战示例

## 🎯 概述

本文档提供了vLLM基准测试框架的详细使用示例，涵盖从基础测试到高级分析的完整流程，帮助初学者快速上手并掌握最佳实践。

---

## 🚀 快速开始示例

### 示例1：基础性能测试

```bash
# 1. 启动vLLM服务
vllm serve microsoft/DialoGPT-medium \
  --host 0.0.0.0 \
  --port 8000 \
  --swap-space 16 \
  --disable-log-requests

# 2. 运行基础测试
python3 benchmark_serving.py \
  --backend vllm \
  --model microsoft/DialoGPT-medium \
  --dataset-name random \
  --num-prompts 100 \
  --request-rate 10 \
  --random-input-len 512 \
  --random-output-len 256
```

**预期输出**：
```
============= Serving Benchmark Result =============
Successful requests:                100
Benchmark duration (s):            12.50
Total input tokens:                 51200
Total generated tokens:             25600
Request throughput (req/s):         8.00
Output token throughput (tok/s):    2048.00
Total Token throughput (tok/s):     6144.00

------------------ Time to First Token ------------------
Mean TTFT (ms):                     125.30
Median TTFT (ms):                   120.50
P99 TTFT (ms):                      180.20
```

---

## 📊 不同测试场景示例

### 示例2：并发性能测试

```bash
# 测试不同并发级别的性能表现
for concurrency in 1 4 8 16 32; do
  echo "Testing concurrency: $concurrency"
  python3 benchmark_serving.py \
    --backend vllm \
    --model your_model_name \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts $((concurrency * 10)) \
    --max-concurrency $concurrency \
    --request-rate inf \
    --save-result \
    --result-filename "concurrency_${concurrency}.json"
done
```

### 示例3：不同输入长度测试

```bash
# 测试不同输入长度对性能的影响
for input_len in 256 512 1024 2048; do
  python3 benchmark_serving.py \
    --backend vllm \
    --model your_model_name \
    --dataset-name random \
    --num-prompts 50 \
    --random-input-len $input_len \
    --random-output-len 256 \
    --request-rate 5 \
    --save-result \
    --result-filename "input_len_${input_len}.json"
done
```

### 示例4：流量模拟测试

```bash
# 模拟真实用户访问模式
python3 benchmark_serving.py \
  --backend vllm \
  --model your_model_name \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000 \
  --request-rate 20 \
  --burstiness 0.8 \
  --max-concurrency 16 \
  --save-result \
  --metadata environment=production load_test=true
```

---

## 🔧 高级配置示例

### 示例5：SLA和Goodput测试

```bash
# 测试服务质量指标
python3 benchmark_serving.py \
  --backend vllm \
  --model your_model_name \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500 \
  --request-rate 15 \
  --goodput ttft:200 tpot:50 e2el:5000 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99 \
  --save-result
```

**Goodput配置说明**：
- `ttft:200`：首个token时间不超过200ms
- `tpot:50`：每token时间不超过50ms  
- `e2el:5000`：端到端延迟不超过5000ms

### 示例6：多模态测试

```bash
# 测试视觉对话模型
python3 benchmark_serving.py \
  --backend openai-chat \
  --model llava-v1.6-vicuna-7b \
  --dataset-name hf \
  --dataset-path lmms-lab/VisionArena \
  --num-prompts 100 \
  --request-rate 2 \
  --save-result \
  --save-detailed
```

### 示例7：性能分析模式

```bash
# 启用详细性能分析
export VLLM_TORCH_PROFILER_DIR=/tmp/vllm_profiles

python3 benchmark_serving.py \
  --backend vllm \
  --model your_model_name \
  --dataset-name random \
  --num-prompts 50 \
  --random-input-len 1024 \
  --random-output-len 512 \
  --request-rate 5 \
  --profile \
  --save-result \
  --save-detailed
```

---

## 📈 批量测试和结果分析

### 示例8：使用combos.yaml批量测试

**配置文件 (combos.yaml)**：
```yaml
model: "microsoft/DialoGPT-medium"
base_url: "http://localhost:8000"
tokenizer: "microsoft/DialoGPT-medium"

# 测试不同的输入输出长度组合
input_output:
  - [256, 128]    # 短对话
  - [512, 256]    # 中等对话
  - [1024, 512]   # 长对话
  - [2048, 1024]  # 超长对话

# 测试不同的并发级别
concurrency_prompts:
  - [1, 20]       # 单用户
  - [4, 80]       # 小团队
  - [8, 160]      # 中等负载
  - [16, 320]     # 高负载
  - [32, 640]     # 极限负载
```

**运行批量测试**：
```bash
# 执行所有配置组合的测试
python3 run_sweep.py

# 聚合所有结果
python3 aggregate_result.py

# 生成可视化报告
python3 visualize.py --all
```

### 示例9：结果分析和可视化

```bash
# 生成吞吐量分析图
python3 visualize.py --throughput

# 生成延迟分析图  
python3 visualize.py --latency

# 生成交互式仪表板
python3 visualize.py --interactive

# 生成完整性能报告
python3 visualize.py --report

# 自定义数据源分析
python3 visualize.py --csv custom_results.csv --output custom_analysis/
```

---

## 🎯 特定场景测试示例

### 示例10：压力测试

```bash
# 极限并发压力测试
python3 benchmark_serving.py \
  --backend vllm \
  --model your_model_name \
  --dataset-name random \
  --num-prompts 2000 \
  --random-input-len 1024 \
  --random-output-len 512 \
  --request-rate inf \
  --max-concurrency 64 \
  --save-result \
  --result-filename "stress_test.json"
```

### 示例11：稳定性测试

```bash
# 长时间稳定性测试
python3 benchmark_serving.py \
  --backend vllm \
  --model your_model_name \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 5000 \
  --request-rate 10 \
  --max-concurrency 8 \
  --save-result \
  --metadata test_type=stability duration=long
```

### 示例12：A/B对比测试

```bash
# 测试配置A
python3 benchmark_serving.py \
  --backend vllm \
  --model your_model_name \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500 \
  --request-rate 15 \
  --max-concurrency 8 \
  --save-result \
  --result-filename "config_a.json" \
  --metadata config=A gpu_memory=24GB

# 测试配置B  
python3 benchmark_serving.py \
  --backend vllm \
  --model your_model_name \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500 \
  --request-rate 15 \
  --max-concurrency 16 \
  --save-result \
  --result-filename "config_b.json" \
  --metadata config=B gpu_memory=24GB
```

---

## 🔍 结果解读示例

### 示例13：性能指标分析

**典型测试输出解读**：
```
============= Serving Benchmark Result =============
Successful requests:                1000        # ✅ 所有请求都成功
Benchmark duration (s):            50.25       # ⏱️ 总测试时间
Total input tokens:                 512000      # 📥 输入token总数
Total generated tokens:             256000      # 📤 输出token总数
Request throughput (req/s):         19.90       # 🚀 请求吞吐量
Request goodput (req/s):            18.50       # ✨ 有效吞吐量 (满足SLA)
Output token throughput (tok/s):    5094.04     # 📊 输出token吞吐量
Total Token throughput (tok/s):     15282.11    # 📈 总token吞吐量

------------------ Time to First Token ------------------
Mean TTFT (ms):                     125.30      # 📊 平均首token时间
Median TTFT (ms):                   120.50      # 📊 中位数首token时间
P99 TTFT (ms):                      180.20      # 📊 99%分位数

--------------- Time per Output Token (excl. 1st token) ---------------
Mean TPOT (ms):                     25.40       # 📊 平均每token时间
Median TPOT (ms):                   24.80       # 📊 中位数每token时间
P99 TPOT (ms):                      35.60       # 📊 99%分位数
```

**性能评估标准**：
- **优秀**：TTFT < 100ms, TPOT < 20ms
- **良好**：TTFT < 200ms, TPOT < 40ms  
- **可接受**：TTFT < 500ms, TPOT < 80ms
- **需优化**：TTFT > 500ms, TPOT > 80ms

---

## 🛠️ 故障排除示例

### 示例14：常见问题诊断

```bash
# 1. 连接测试
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your_model_name",
    "prompt": "Hello",
    "max_tokens": 10
  }'

# 2. 简单功能测试
python3 benchmark_serving.py \
  --backend vllm \
  --model your_model_name \
  --dataset-name random \
  --num-prompts 1 \
  --random-input-len 10 \
  --random-output-len 5

# 3. 逐步增加负载测试
for rate in 1 2 5 10 20; do
  echo "Testing rate: $rate req/s"
  python3 benchmark_serving.py \
    --backend vllm \
    --model your_model_name \
    --dataset-name random \
    --num-prompts 20 \
    --request-rate $rate \
    --random-input-len 256 \
    --random-output-len 128
done
```

---

## 📋 最佳实践清单

### ✅ 测试前准备
- [ ] 确认vLLM服务正常运行
- [ ] 验证模型加载成功
- [ ] 检查系统资源充足
- [ ] 准备合适的测试数据集

### ✅ 测试执行
- [ ] 从小规模测试开始
- [ ] 逐步增加负载
- [ ] 记录测试环境信息
- [ ] 保存详细测试结果

### ✅ 结果分析
- [ ] 关注关键性能指标
- [ ] 分析性能瓶颈
- [ ] 对比不同配置结果
- [ ] 生成可视化报告

### ✅ 优化改进
- [ ] 基于结果调整配置
- [ ] 重复测试验证改进
- [ ] 建立性能基线
- [ ] 持续监控性能变化

通过这些实战示例，初学者可以快速掌握vLLM基准测试框架的使用方法，并建立起系统的性能测试流程。
