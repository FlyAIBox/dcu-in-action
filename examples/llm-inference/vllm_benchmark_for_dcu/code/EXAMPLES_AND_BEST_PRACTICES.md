# 🎯 使用示例和最佳实践

## 📋 常用测试场景示例

### 1. 快速功能验证测试

**目标**: 验证系统基本功能是否正常

```bash
# 最简单的测试命令
python benchmark_serving.py \
    --backend vllm \
    --model /path/to/your/model \
    --dataset-name random \
    --num-prompts 5 \
    --random-input-len 50 \
    --random-output-len 20

# 预期结果: 5个请求全部成功，获得基础性能数据
```

**适用场景**:
- 新部署服务的功能验证
- 代码修改后的回归测试
- 快速检查服务状态

### 2. 性能基准测试

**目标**: 建立系统性能基准线

```bash
# 标准基准测试
python benchmark_serving.py \
    --backend vllm \
    --model /path/to/your/model \
    --dataset-name random \
    --num-prompts 1000 \
    --random-input-len 512 \
    --random-output-len 128 \
    --save-result \
    --result-dir ./benchmarks \
    --metadata model_version=v1.0 hardware=8xDCU

# 预期结果: 获得1000个请求的详细性能数据
```

**关键参数说明**:
- `--num-prompts 1000`: 足够的样本量确保统计意义
- `--save-result`: 保存详细结果用于后续分析
- `--metadata`: 添加元数据便于结果管理

### 3. 并发压力测试

**目标**: 测试系统在高并发下的表现

```bash
# 渐进式并发测试
for concurrency in 10 20 50 100 200; do
    echo "Testing with concurrency: $concurrency"
    python benchmark_serving.py \
        --backend vllm \
        --model /path/to/your/model \
        --dataset-name random \
        --num-prompts 500 \
        --request-rate 50 \
        --max-concurrency $concurrency \
        --random-input-len 256 \
        --random-output-len 64 \
        --save-result \
        --result-filename "concurrency_${concurrency}.json"
done

# 分析不同并发度下的性能变化
```

**分析要点**:
- 观察吞吐量随并发度的变化
- 找到性能拐点 (最优并发度)
- 监控系统资源使用情况

### 4. 真实场景模拟测试

**目标**: 使用真实数据模拟用户使用场景

```bash
# 下载ShareGPT数据集
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# 真实场景测试
python benchmark_serving.py \
    --backend vllm \
    --model /path/to/your/model \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 200 \
    --request-rate 5 \
    --save-result \
    --save-detailed

# 预期结果: 更接近真实使用场景的性能数据
```

**优势**:
- 输入长度分布更真实
- 对话内容更贴近实际使用
- 能发现合成数据测试中遗漏的问题

### 5. 长文本处理测试

**目标**: 测试系统处理长文本的能力

```bash
# 长文本测试
python benchmark_serving.py \
    --backend vllm \
    --model /path/to/your/model \
    --dataset-name random \
    --num-prompts 100 \
    --random-input-len 4096 \
    --random-output-len 1024 \
    --request-rate 2 \
    --max-concurrency 10

# 预期结果: 了解长文本处理的性能特征
```

**注意事项**:
- 长文本会显著增加内存使用
- 需要调整模型的max_model_len参数
- 可能需要降低并发度避免OOM

### 6. 服务质量 (SLO) 测试

**目标**: 验证系统是否满足服务级别目标

```bash
# SLO测试
python benchmark_serving.py \
    --backend vllm \
    --model /path/to/your/model \
    --dataset-name random \
    --num-prompts 1000 \
    --request-rate 20 \
    --goodput ttft:100 tpot:50 e2el:2000 \
    --random-input-len 512 \
    --random-output-len 128

# 预期结果: 获得满足SLO的请求比例 (goodput)
```

**SLO配置说明**:
- `ttft:100`: 首次响应时间 < 100ms
- `tpot:50`: 每token生成时间 < 50ms  
- `e2el:2000`: 端到端延迟 < 2000ms

## 🏆 最佳实践指南

### 1. 测试环境准备

#### 硬件环境
```bash
# 检查DCU状态
rocm-smi

# 设置DCU环境
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_MAX_NCHANNELS=16
export NCCL_P2P_LEVEL=SYS

# 检查内存和存储
free -h
df -h
```

#### 软件环境
```bash
# 创建独立的测试环境
conda create -n benchmark python=3.10
conda activate benchmark

# 安装依赖
pip install vllm transformers datasets aiohttp tqdm numpy pandas

# 验证安装
python -c "import vllm; print('vLLM installed successfully')"
```

### 2. 测试策略设计

#### 分层测试策略
```
1. 冒烟测试 (Smoke Test)
   └── 5-10个请求，验证基本功能

2. 功能测试 (Functional Test)  
   └── 100-200个请求，验证各项功能

3. 性能测试 (Performance Test)
   └── 1000+请求，建立性能基准

4. 压力测试 (Stress Test)
   └── 高并发、长时间测试

5. 稳定性测试 (Stability Test)
   └── 24小时持续运行测试
```

#### 参数选择原则
```bash
# 输入长度选择
--random-input-len 512    # 短文本场景
--random-input-len 1024   # 中等长度场景  
--random-input-len 2048   # 长文本场景

# 输出长度选择
--random-output-len 64    # 简短回复
--random-output-len 128   # 标准回复
--random-output-len 256   # 详细回复

# 请求数量选择
--num-prompts 10          # 快速验证
--num-prompts 100         # 功能测试
--num-prompts 1000        # 性能基准
--num-prompts 10000       # 压力测试
```

### 3. 结果分析方法

#### 性能指标解读
```python
# 延迟指标分析
def analyze_latency_metrics(results):
    """分析延迟指标的方法"""
    
    # 1. 查看平均值和中位数的差异
    ttft_mean = results['mean_ttft_ms']
    ttft_median = results['median_ttft_ms']
    
    if ttft_mean > ttft_median * 1.5:
        print("⚠️  TTFT存在长尾延迟问题")
    
    # 2. 分析P99指标
    ttft_p99 = results['p99_ttft_ms']
    if ttft_p99 > ttft_mean * 3:
        print("⚠️  系统存在严重的性能不稳定")
    
    # 3. 评估用户体验
    if ttft_median < 100:
        print("✅ 用户体验优秀")
    elif ttft_median < 200:
        print("✅ 用户体验良好")
    else:
        print("❌ 用户体验需要优化")
```

#### 趋势分析
```bash
# 批量测试脚本示例
#!/bin/bash
for batch_size in 1 2 4 8 16 32 64; do
    echo "Testing batch size: $batch_size"
    python benchmark_serving.py \
        --backend vllm \
        --model /path/to/model \
        --dataset-name random \
        --num-prompts $batch_size \
        --save-result \
        --result-filename "batch_${batch_size}.json"
done

# 分析批次大小对性能的影响
python analyze_batch_results.py
```

### 4. 常见问题和解决方案

#### 问题1: 连接超时
```bash
# 症状: requests.exceptions.ConnectTimeout
# 原因: vLLM服务未启动或端口不正确
# 解决: 
curl http://localhost:8000/v1/models  # 验证服务状态
netstat -tlnp | grep 8000            # 检查端口占用
```

#### 问题2: 内存不足 (OOM)
```bash
# 症状: CUDA out of memory
# 原因: 模型太大或并发度过高
# 解决:
--gpu-memory-utilization 0.8         # 降低GPU内存使用率
--max-concurrency 10                 # 限制并发数
--max-model-len 2048                 # 减少最大序列长度
```

#### 问题3: 性能异常波动
```bash
# 症状: 性能指标不稳定，方差很大
# 原因: 系统负载、网络抖动、资源竞争
# 解决:
# 1. 系统预热
for i in {1..10}; do
    curl -s http://localhost:8000/v1/completions \
        -d '{"model":"test","prompt":"hello","max_tokens":10}' \
        -H "Content-Type: application/json" > /dev/null
done

# 2. 监控系统资源
htop &
rocm-smi -l 1 &

# 3. 多次测试取平均值
for i in {1..5}; do
    python benchmark_serving.py ... --result-filename "run_${i}.json"
done
```

### 5. 自动化测试脚本

#### 完整的自动化测试流程
```bash
#!/bin/bash
# automated_benchmark.sh

set -e  # 遇到错误立即退出

# 配置参数
MODEL_PATH="/path/to/your/model"
RESULT_DIR="./results/$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULT_DIR

# 1. 环境检查
echo "🔍 检查测试环境..."
python -c "import vllm, torch; print(f'vLLM: OK, PyTorch: {torch.__version__}')"
curl -s http://localhost:8000/v1/models > /dev/null || {
    echo "❌ vLLM服务未启动"
    exit 1
}

# 2. 冒烟测试
echo "🚀 执行冒烟测试..."
python benchmark_serving.py \
    --backend vllm \
    --model $MODEL_PATH \
    --dataset-name random \
    --num-prompts 5 \
    --result-filename "$RESULT_DIR/smoke_test.json"

# 3. 性能基准测试
echo "📊 执行性能基准测试..."
python benchmark_serving.py \
    --backend vllm \
    --model $MODEL_PATH \
    --dataset-name random \
    --num-prompts 1000 \
    --save-result \
    --result-filename "$RESULT_DIR/benchmark.json"

# 4. 并发测试
echo "⚡ 执行并发测试..."
for concurrency in 10 20 50; do
    python benchmark_serving.py \
        --backend vllm \
        --model $MODEL_PATH \
        --dataset-name random \
        --num-prompts 200 \
        --max-concurrency $concurrency \
        --result-filename "$RESULT_DIR/concurrency_${concurrency}.json"
done

# 5. 生成测试报告
echo "📋 生成测试报告..."
python generate_report.py $RESULT_DIR

echo "✅ 测试完成！结果保存在: $RESULT_DIR"
```

通过这些示例和最佳实践，你可以系统性地进行大模型推理性能测试，获得可靠的性能数据，并持续优化系统性能。
