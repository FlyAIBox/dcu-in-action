# 🚀 大模型并发能力测试专业指南

## 🎯 核心问题：我的大模型支持多少路并发？

这是每个大模型部署者最关心的问题。本指南将详细解答如何科学地测试和评估大模型的并发处理能力。

## 🔧 三个核心参数深度解析

### 1. `--max-concurrency`: 并发度控制 🔥 **最重要**

**定义**: 限制同时进行的HTTP请求数量

```bash
# 示例对比
--max-concurrency 1    # 串行处理，一个接一个
--max-concurrency 50   # 最多50个请求同时进行
--max-concurrency 200  # 最多200个请求同时进行
```

**工作原理**:
```python
# 内部实现原理 (简化版)
semaphore = asyncio.Semaphore(max_concurrency)  # 信号量控制

async def limited_request_func(request):
    async with semaphore:  # 获取信号量
        return await send_request(request)  # 发送请求
    # 请求完成后自动释放信号量
```

**影响分析**:
- **过小**: 无法充分利用服务器性能，QPS上不去
- **过大**: 可能压垮服务器，导致请求超时或失败
- **合适**: 在系统稳定性和性能之间找到平衡点

### 2. `--request-rate`: 请求发送速率

**定义**: 控制每秒发送多少个新请求

```bash
--request-rate inf      # 立即发送所有请求 (压力测试)
--request-rate 10       # 每秒发送10个新请求 (模拟真实流量)
--request-rate 0.1      # 每10秒发送1个请求 (低频测试)
```

**两种模式对比**:

| 参数值 | 模式 | 请求发送方式 | 适用场景 |
|--------|------|-------------|----------|
| `inf` | 批量模式 | 瞬间发送所有请求 | 压力测试、找系统极限 |
| 有限值 | 流量控制 | 按指定速率逐步发送 | 真实场景模拟 |

**与并发度的关系**:
```
实际并发数 = min(
    max_concurrency,           # 设定的最大并发限制
    request_rate × 平均响应时间  # 自然形成的并发数
)
```

### 3. `--num-prompts`: 请求总数

**定义**: 本次测试要发送的请求总数

```bash
--num-prompts 100    # 发送100个请求后结束测试
--num-prompts 1000   # 发送1000个请求后结束测试
```

**选择原则**:
- **统计意义**: 至少是最大并发数的5-10倍
- **测试时长**: 考虑测试执行时间的合理性
- **资源消耗**: 避免过多请求导致资源耗尽

## 🧪 并发能力测试方法

### 方法1: 渐进式并发测试 (推荐)

**目标**: 找到系统的最大稳定并发数

```bash
#!/bin/bash
# concurrency_test.sh - 渐进式并发测试脚本

MODEL_PATH="/path/to/your/model"
RESULT_DIR="./concurrency_results"
mkdir -p $RESULT_DIR

echo "🚀 开始渐进式并发测试..."

# 测试不同并发度
for concurrency in 1 2 5 10 20 50 100 200 500 1000; do
    echo "📊 测试并发度: $concurrency"
    
    # 执行测试
    python benchmark_serving.py \
        --backend vllm \
        --model $MODEL_PATH \
        --dataset-name random \
        --num-prompts $((concurrency * 10)) \
        --request-rate inf \
        --max-concurrency $concurrency \
        --random-input-len 512 \
        --random-output-len 128 \
        --save-result \
        --result-filename "$RESULT_DIR/concurrency_${concurrency}.json" \
        2>&1 | tee "$RESULT_DIR/concurrency_${concurrency}.log"
    
    # 提取关键指标
    SUCCESS_RATE=$(grep "Successful requests:" "$RESULT_DIR/concurrency_${concurrency}.log" | awk '{print $3}')
    TOTAL_REQUESTS=$((concurrency * 10))
    SUCCESS_PERCENTAGE=$((SUCCESS_RATE * 100 / TOTAL_REQUESTS))
    
    THROUGHPUT=$(grep "Request throughput" "$RESULT_DIR/concurrency_${concurrency}.log" | awk '{print $4}')
    MEAN_TTFT=$(grep "Mean TTFT" "$RESULT_DIR/concurrency_${concurrency}.log" | awk '{print $4}')
    
    echo "✅ 并发度 $concurrency: 成功率 ${SUCCESS_PERCENTAGE}%, 吞吐量 ${THROUGHPUT} req/s, TTFT ${MEAN_TTFT}ms"
    
    # 判断是否达到系统极限
    if [ "$SUCCESS_PERCENTAGE" -lt 95 ]; then
        echo "⚠️  成功率低于95%，可能接近系统极限"
        echo "🎯 建议最大并发数: $((concurrency / 2))"
        break
    fi
    
    # 检查延迟是否急剧增加
    if (( $(echo "$MEAN_TTFT > 1000" | bc -l) )); then
        echo "⚠️  TTFT超过1秒，性能显著下降"
        echo "🎯 建议最大并发数: $((concurrency / 2))"
        break
    fi
done

echo "📋 测试完成！结果保存在: $RESULT_DIR"
```

### 方法2: 固定QPS并发测试

**目标**: 在指定QPS下测试所需的并发度

```bash
#!/bin/bash
# qps_concurrency_test.sh - 固定QPS测试脚本

TARGET_QPS=20  # 目标QPS
MODEL_PATH="/path/to/your/model"

echo "🎯 目标QPS: $TARGET_QPS"

# 测试不同并发度下能否达到目标QPS
for concurrency in 10 20 50 100 200; do
    echo "📊 测试并发度: $concurrency"
    
    python benchmark_serving.py \
        --backend vllm \
        --model $MODEL_PATH \
        --dataset-name random \
        --num-prompts 1000 \
        --request-rate $TARGET_QPS \
        --max-concurrency $concurrency \
        --random-input-len 512 \
        --random-output-len 128 \
        --result-filename "qps_${TARGET_QPS}_concurrency_${concurrency}.json"
    
    # 检查实际达到的QPS
    ACTUAL_QPS=$(grep "Request throughput" *.log | tail -1 | awk '{print $4}')
    echo "实际QPS: $ACTUAL_QPS"
    
    # 如果达到目标QPS，记录最小所需并发度
    if (( $(echo "$ACTUAL_QPS >= $TARGET_QPS * 0.95" | bc -l) )); then
        echo "✅ 达到目标QPS！最小并发度: $concurrency"
        break
    fi
done
```

## 📊 结果分析和解读

### 关键指标分析

#### 1. 并发效率曲线
```python
# 分析并发效率的Python脚本示例
import json
import matplotlib.pyplot as plt

def analyze_concurrency_efficiency():
    concurrency_levels = [1, 5, 10, 20, 50, 100, 200]
    throughputs = []
    
    for c in concurrency_levels:
        with open(f'concurrency_{c}.json', 'r') as f:
            data = json.load(f)
            throughputs.append(data['request_throughput'])
    
    # 计算并发效率
    efficiency = [t/c for t, c in zip(throughputs, concurrency_levels)]
    
    # 绘制图表
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(concurrency_levels, throughputs, 'b-o')
    plt.xlabel('并发度')
    plt.ylabel('吞吐量 (req/s)')
    plt.title('吞吐量 vs 并发度')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(concurrency_levels, efficiency, 'r-o')
    plt.xlabel('并发度')
    plt.ylabel('并发效率 (req/s per concurrency)')
    plt.title('并发效率 vs 并发度')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('concurrency_analysis.png')
    plt.show()
    
    # 找到最优并发度
    max_efficiency_idx = efficiency.index(max(efficiency))
    optimal_concurrency = concurrency_levels[max_efficiency_idx]
    
    print(f"🎯 最优并发度: {optimal_concurrency}")
    print(f"📊 最大吞吐量: {max(throughputs):.2f} req/s")
    print(f"⚡ 最高效率: {max(efficiency):.3f} req/s per concurrency")

analyze_concurrency_efficiency()
```

## 📋 参数选择速查表

### 测试场景与参数配置对照表

| 测试目标 | `--max-concurrency` | `--request-rate` | `--num-prompts` | 推荐配置 |
|----------|-------------------|------------------|----------------|----------|
| **快速验证** | 10 | inf | 50 | 验证基本功能是否正常 |
| **找系统极限** | 1→2→5→10→20→50... | inf | 并发数×10 | 渐进式增加直到失败 |
| **真实场景模拟** | 根据测试调整 | 预期QPS | 1000+ | 模拟用户访问模式 |
| **容量规划** | 目标并发×1.5 | 目标QPS | 2000+ | 为业务增长预留空间 |
| **稳定性验证** | 最优并发×0.8 | 目标QPS×0.8 | 大量 | 长时间运行测试 |

### 不同业务场景的参数建议

#### 🎯 在线客服系统
```bash
# 特点: 中等并发，响应要求快
--max-concurrency 50
--request-rate 20
--num-prompts 1000
--random-input-len 256
--random-output-len 128
```

#### 🎯 内容生成平台  
```bash
# 特点: 高并发，可接受较长延迟
--max-concurrency 200
--request-rate 100
--num-prompts 2000
--random-input-len 512
--random-output-len 512
```

#### 🎯 代码助手工具
```bash
# 特点: 中低并发，输入输出都较长
--max-concurrency 20
--request-rate 10
--num-prompts 500
--random-input-len 1024
--random-output-len 256
```

## 🎯 总结：用户最关心的问题

### Q: 我的大模型支持多少路并发？
**A: 使用 `--max-concurrency` 参数进行渐进式测试**

1. **快速估算**: 从10开始测试，观察成功率和延迟
2. **精确测试**: 使用渐进式脚本找到准确的极限值
3. **安全部署**: 使用测试结果的70-80%作为生产配置

### Q: 三个参数哪个最重要？
**A: 对于并发能力测试，重要性排序：**

1. **🥇 `--max-concurrency`** - 直接控制并发度，最重要
2. **🥈 `--request-rate`** - 控制测试模式，影响结果真实性  
3. **🥉 `--num-prompts`** - 保证统计意义，辅助参数

### Q: 如何选择合适的参数组合？
**A: 根据测试目标选择：**

- **找极限**: `--max-concurrency` 递增 + `--request-rate inf`
- **模拟真实**: `--request-rate` 固定 + `--max-concurrency` 调优
- **验证稳定**: 两个参数都固定 + 长时间运行

### Q: 什么时候停止增加并发度？
**A: 出现以下情况之一时停止：**

1. 成功率 < 95%
2. TTFT > 1000ms (或业务要求的阈值)
3. 吞吐量增长率 < 20%
4. 系统资源耗尽 (GPU内存、CPU等)

通过这个专业指南，你可以准确评估大模型的并发处理能力，为生产部署提供可靠的性能数据。
