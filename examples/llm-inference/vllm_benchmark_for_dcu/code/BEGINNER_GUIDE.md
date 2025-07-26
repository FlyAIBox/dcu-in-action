# 🚀 大模型推理压测初学者完全指南

## 📚 什么是大模型推理压测？

大模型推理压测是评估大语言模型在实际部署环境中性能表现的重要手段。通过模拟真实用户请求，我们可以了解模型的：

- **响应速度**: 用户发送请求到收到回复的时间
- **处理能力**: 系统能同时处理多少个请求
- **稳定性**: 在高负载下系统是否稳定运行
- **资源利用率**: CPU、GPU、内存的使用效率

## 🎯 为什么需要压测？

### 1. 用户体验优化
- 确保用户获得流畅的对话体验
- 避免响应时间过长导致用户流失
- 优化系统配置以提供最佳性能

### 2. 成本控制
- 合理配置硬件资源，避免浪费
- 找到性价比最优的部署方案
- 预测扩容需求和成本

### 3. 系统稳定性
- 发现系统在高负载下的问题
- 验证系统的容错能力
- 制定合理的限流策略

## 🔧 核心概念解释

### 延迟指标 (Latency Metrics)

#### TTFT (Time To First Token) - 首次响应时间
```
用户: "你好，请介绍一下自己"
系统: [等待125ms] "你" [继续生成...]
      ↑
   TTFT = 125ms
```
- **含义**: 从发送请求到收到第一个字符的时间
- **重要性**: 直接影响用户感知的响应速度
- **优秀标准**: < 100ms (优秀), < 200ms (良好), > 500ms (需优化)

#### TPOT (Time Per Output Token) - 生成速度
```
系统生成: "你好，我是AI助手，很高兴为你服务"
每个字符间隔: 35ms
TPOT = 35ms
```
- **含义**: 生成每个字符/词的平均时间
- **重要性**: 影响整体对话的流畅度
- **优秀标准**: < 50ms (优秀), < 100ms (良好), > 200ms (需优化)

#### ITL (Inter-Token Latency) - 字符间延迟
```
"你" [30ms] "好" [40ms] "，" [35ms] "我" [32ms] ...
ITL = [30, 40, 35, 32, ...]
```
- **含义**: 相邻两个字符之间的时间间隔
- **重要性**: 反映流式输出的稳定性
- **分析方法**: 查看ITL的方差，方差越小越稳定

### 吞吐量指标 (Throughput Metrics)

#### Request Throughput - 请求处理能力
```
1秒内处理了22个用户请求
Request Throughput = 22 req/s
```
- **含义**: 系统每秒能处理多少个用户请求
- **重要性**: 决定系统能支撑多少并发用户
- **影响因素**: 硬件配置、模型大小、请求复杂度

#### Token Throughput - 文本生成速度
```
1秒内生成了2830个字符/词
Token Throughput = 2830 tok/s
```
- **含义**: 系统每秒能生成多少个字符/词
- **重要性**: 反映模型的实际生成效率
- **对比标准**: 不同模型和硬件配置差异很大

## 🛠️ 测试工具使用指南

### 基础命令结构
```bash
python benchmark_serving.py \
    --backend vllm \              # 使用vLLM后端
    --model /path/to/model \      # 模型路径
    --dataset-name random \       # 使用随机数据
    --num-prompts 100 \          # 发送100个请求
    --random-input-len 512 \     # 输入长度512个token
    --random-output-len 128      # 输出长度128个token
```

### 参数详解

#### 后端选择 (--backend)
- `vllm`: 推荐，性能最优，支持DCU
- `openai`: 兼容OpenAI API的服务
- `tgi`: Text Generation Inference

#### 数据集选择 (--dataset-name)
- `random`: 随机生成数据，适合压力测试
- `sharegpt`: 真实对话数据，适合实际场景测试
- `sonnet`: 文学文本，适合长文本生成测试

#### 负载控制参数详解

##### `--num-prompts`: 总请求数量
```bash
--num-prompts 1000    # 发送1000个测试请求
```
- **含义**: 本次测试总共要发送多少个请求
- **影响**: 决定测试的样本量和测试时长
- **选择建议**:
  - 快速验证: 10-50个请求
  - 功能测试: 100-500个请求
  - 性能基准: 1000-5000个请求
  - 压力测试: 10000+个请求

##### `--request-rate`: 请求发送速率 (req/s)
```bash
--request-rate inf     # 立即发送所有请求 (批量模式)
--request-rate 10      # 每秒发送10个请求 (流量控制模式)
--request-rate 0.5     # 每2秒发送1个请求 (低频测试)
```
- **含义**: 控制请求发送的时间间隔，模拟真实用户访问模式
- **两种模式对比**:

| 模式 | request-rate | 适用场景 | 优缺点 |
|------|-------------|----------|--------|
| 批量模式 | `inf` | 压力测试、最大性能测试 | ✅ 测试系统极限性能<br/>❌ 不符合真实使用场景 |
| 流量控制 | 有限值 | 真实场景模拟、稳定性测试 | ✅ 更真实的负载模式<br/>❌ 测试时间较长 |

- **选择建议**:
  - 系统极限测试: `--request-rate inf`
  - 真实场景模拟: `--request-rate 1-50` (根据预期QPS)
  - 稳定性测试: `--request-rate 5-20`

##### `--max-concurrency`: 最大并发数 🔥 **核心参数**
```bash
--max-concurrency 50   # 最多同时处理50个请求
--max-concurrency 100  # 最多同时处理100个请求
```
- **含义**: 限制同时进行的请求数量，防止客户端压垮服务器
- **重要性**: 这是测试大模型并发能力的关键参数！
- **与request-rate的关系**:
  ```
  实际QPS = min(request-rate, 服务器处理能力)

  如果 max-concurrency 设置过小 → 限制了并发度 → QPS上不去
  如果 max-concurrency 设置过大 → 可能压垮服务器 → 请求失败
  ```

#### 🚀 大模型并发能力测试 - 核心关注点

##### 什么是大模型的并发能力？
大模型的并发能力指系统能够**同时处理多少个用户请求**而不出现明显的性能下降或服务中断。

##### 如何测试大模型支持多少路并发？

**方法1: 渐进式并发测试 (推荐)**
```bash
#!/bin/bash
# 测试不同并发度下的性能表现
for concurrency in 1 5 10 20 50 100 200 500; do
    echo "🧪 测试并发度: $concurrency"
    python benchmark_serving.py \
        --backend vllm \
        --model /path/to/model \
        --dataset-name random \
        --num-prompts 200 \
        --request-rate inf \
        --max-concurrency $concurrency \
        --random-input-len 512 \
        --random-output-len 128 \
        --result-filename "concurrency_${concurrency}.json"

    # 检查成功率
    success_rate=$(grep "Successful requests" log.txt | awk '{print $3}')
    if [ "$success_rate" -lt 190 ]; then  # 成功率 < 95%
        echo "⚠️  并发度 $concurrency 时成功率下降，可能接近系统极限"
        break
    fi
done
```

**方法2: 固定QPS下的并发测试**
```bash
# 固定每秒20个请求，测试需要多少并发度
python benchmark_serving.py \
    --backend vllm \
    --model /path/to/model \
    --dataset-name random \
    --num-prompts 1000 \
    --request-rate 20 \
    --max-concurrency 100 \
    --random-input-len 512 \
    --random-output-len 128
```

##### 并发能力的评估指标

1. **最大稳定并发数**: 系统能稳定处理的最大并发请求数
2. **并发效率**: 并发度增加时，吞吐量的提升比例
3. **并发延迟**: 不同并发度下的平均响应时间
4. **并发成功率**: 高并发下的请求成功率

##### 哪个参数更重要？

**对于测试大模型并发能力，参数重要性排序：**

1. **🥇 `--max-concurrency`** (最重要)
   - 直接控制并发度
   - 决定能测试到的最大并发数
   - 影响系统资源利用率

2. **🥈 `--request-rate`** (次重要)
   - 控制请求到达模式
   - 影响测试的真实性
   - 与并发度配合使用

3. **🥉 `--num-prompts`** (辅助)
   - 保证足够的样本量
   - 影响测试结果的统计意义
   - 通常设置为并发数的5-10倍

### 测试场景设计

#### 1. 基础功能测试
```bash
# 目标: 验证系统基本功能
python benchmark_serving.py \
    --backend vllm \
    --model /path/to/model \
    --dataset-name random \
    --num-prompts 10 \
    --random-input-len 100 \
    --random-output-len 50
```

#### 2. 性能基准测试
```bash
# 目标: 建立性能基准线
python benchmark_serving.py \
    --backend vllm \
    --model /path/to/model \
    --dataset-name random \
    --num-prompts 1000 \
    --random-input-len 512 \
    --random-output-len 128
```

#### 3. 并发压力测试
```bash
# 目标: 测试系统并发处理能力
python benchmark_serving.py \
    --backend vllm \
    --model /path/to/model \
    --dataset-name random \
    --num-prompts 1000 \
    --request-rate 50 \
    --max-concurrency 100 \
    --random-input-len 512 \
    --random-output-len 128
```

#### 4. 真实场景测试
```bash
# 目标: 模拟真实用户使用场景
python benchmark_serving.py \
    --backend vllm \
    --model /path/to/model \
    --dataset-name sharegpt \
    --dataset-path /path/to/sharegpt.json \
    --num-prompts 500 \
    --request-rate 10
```

## 📊 结果解读指南

### 控制台输出解读
```
=============== Serving Benchmark Result ===============
Successful requests:                     1000      # ✅ 成功处理的请求数
Benchmark duration (s):                 45.23     # ⏱️ 总测试时间
Total input tokens:                      512000    # 📝 输入token总数
Total generated tokens:                  128000    # 🎯 生成token总数
Request throughput (req/s):              22.11     # 🚀 请求吞吐量
Output token throughput (tok/s):         2830.45   # ⚡ 输出吞吐量
Total Token throughput (tok/s):          14152.25  # 💫 总吞吐量

----------------------- Time to First Token -----------------------
Mean TTFT (ms):                          125.34    # 📊 平均首次响应时间
Median TTFT (ms):                        118.67    # 📈 中位数首次响应时间
P99 TTFT (ms):                          245.89     # 🔺 99%用户的响应时间

----------------------- Time per Output Token ----------------------
Mean TPOT (ms):                          35.67     # 📊 平均生成速度
Median TPOT (ms):                        32.45     # 📈 中位数生成速度
P99 TPOT (ms):                          78.23      # 🔺 99%情况下的生成速度
```

### 性能评估标准

#### 优秀性能指标 (参考值)
| 指标 | 优秀 | 良好 | 需优化 |
|------|------|------|--------|
| TTFT | < 100ms | < 200ms | > 500ms |
| TPOT | < 50ms | < 100ms | > 200ms |
| 请求吞吐量 | > 20 req/s | > 10 req/s | < 5 req/s |
| 成功率 | > 99% | > 95% | < 90% |

#### 性能问题诊断

**问题1: TTFT过高 (> 500ms)**
- 可能原因: 模型加载慢、GPU内存不足、网络延迟
- 解决方案: 预热模型、增加GPU内存、优化网络配置

**问题2: TPOT不稳定 (方差大)**
- 可能原因: 系统负载不均、内存碎片、调度问题
- 解决方案: 调整批处理大小、重启服务、优化系统配置

**问题3: 吞吐量低**
- 可能原因: 并发度不够、硬件瓶颈、配置不当
- 解决方案: 增加并发数、升级硬件、调整参数

## 🎯 实战练习

### 练习1: 基础测试
1. 启动vLLM服务
2. 运行10个简单请求
3. 观察和记录TTFT、TPOT指标
4. 分析结果是否符合预期

### 练习2: 负载测试
1. 逐步增加请求数量: 10 → 100 → 1000
2. 观察性能指标的变化趋势
3. 找到系统的性能拐点
4. 记录最优配置参数

### 练习3: 优化实验
1. 调整不同的参数组合
2. 对比不同配置的性能表现
3. 找到最适合你的业务场景的配置
4. 制定部署和监控策略

通过这些练习，你将掌握大模型推理压测的核心技能，能够独立进行性能评估和优化工作。
