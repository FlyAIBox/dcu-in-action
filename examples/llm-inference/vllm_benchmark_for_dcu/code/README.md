# 海光DCU大模型推理性能基准测试工具

## 📖 项目概述

这是一个专门为海光DCU优化的vLLM大模型推理性能测试工具，用于评估大模型在DCU硬件上的推理性能。该工具提供了全面的性能指标分析，支持多种推理后端和数据集，是大模型推理压测的完整解决方案。

### 🎯 主要功能

- **多后端支持**: vLLM、TGI、OpenAI API、TensorRT-LLM等
- **多数据集支持**: ShareGPT、Random、Sonnet、HuggingFace等
- **全面性能分析**: TTFT、TPOT、ITL、E2EL等关键指标
- **并发控制**: 支持请求速率控制和最大并发限制
- **DCU优化**: 针对海光DCU硬件的专门优化
- **结果导出**: 支持JSON、CSV等多种格式输出

## 🏗️ 系统架构

### 核心组件

1. **benchmark_serving.py** - 主测试脚本
   - 命令行参数解析
   - 测试流程控制
   - 性能指标计算
   - 结果输出管理

2. **backend_request_func.py** - 后端通信模块
   - 统一的请求接口
   - 异步HTTP通信
   - 流式响应处理
   - 性能数据收集

3. **benchmark_dataset.py** - 数据集处理模块
   - 多种数据集支持
   - 数据采样和预处理
   - 多模态数据处理

4. **benchmark_utils.py** - 工具函数模块
   - 结果格式转换
   - JSON序列化处理

## 📊 关键性能指标

### 延迟指标 (Latency Metrics)

- **TTFT (Time To First Token)**: 首次token时间
  - 定义: 从发送请求到接收第一个token的时间
  - 重要性: 反映用户感知的响应速度
  - 单位: 毫秒 (ms)

- **TPOT (Time Per Output Token)**: 每token时间
  - 定义: 生成每个输出token的平均时间
  - 重要性: 反映生成速度的稳定性
  - 单位: 毫秒 (ms)

- **ITL (Inter-Token Latency)**: 迭代延迟
  - 定义: 相邻两个token之间的时间间隔
  - 重要性: 反映流式输出的流畅度
  - 单位: 毫秒 (ms)

- **E2EL (End-to-End Latency)**: 端到端延迟
  - 定义: 从请求发送到完整响应接收的总时间
  - 重要性: 反映整体处理时间
  - 单位: 毫秒 (ms)

### 吞吐量指标 (Throughput Metrics)

- **Request Throughput**: 请求吞吐量
  - 定义: 每秒处理的请求数量
  - 单位: 请求/秒 (req/s)

- **Output Token Throughput**: 输出吞吐量
  - 定义: 每秒生成的token数量
  - 单位: token/秒 (tok/s)

- **Total Token Throughput**: 总token吞吐量
  - 定义: 每秒处理的总token数量(输入+输出)
  - 单位: token/秒 (tok/s)

### 服务质量指标 (Quality of Service)

- **Goodput**: 良好吞吐量
  - 定义: 满足SLO(服务级别目标)的请求比例
  - 重要性: 反映服务质量的稳定性

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install vllm transformers datasets aiohttp tqdm numpy pandas

# 设置DCU环境变量
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### 2. 启动vLLM服务

```bash
# 使用提供的服务启动脚本
bash server.sh

# 或手动启动
vllm serve /path/to/your/model \
    --trust-remote-code \
    --dtype float16 \
    --max-model-len 32768 \
    -tp 8 \
    --gpu-memory-utilization 0.9 \
    --port 8000
```

### 3. 运行基准测试

#### 基础测试
```bash
python benchmark_serving.py \
    --backend vllm \
    --model /path/to/your/model \
    --dataset-name random \
    --num-prompts 100 \
    --random-input-len 512 \
    --random-output-len 128
```

#### 高级测试配置
```bash
python benchmark_serving.py \
    --backend vllm \
    --model /path/to/your/model \
    --dataset-name sharegpt \
    --dataset-path /path/to/sharegpt.json \
    --num-prompts 1000 \
    --request-rate 10 \
    --max-concurrency 50 \
    --save-result \
    --result-dir ./results
```

### 4. 批量测试

```bash
# 使用提供的批量测试脚本
bash test.sh
```

## 📋 命令行参数详解

### 基础参数

- `--backend`: 推理后端选择 (vllm, openai, tgi等)
- `--model`: 模型名称或路径
- `--host`: 服务器地址 (默认: 127.0.0.1)
- `--port`: 服务器端口 (默认: 8000)

### 数据集参数

- `--dataset-name`: 数据集类型 (sharegpt, random, sonnet, hf)
- `--dataset-path`: 数据集文件路径
- `--num-prompts`: 测试请求数量 (默认: 1000)

### 性能控制参数

- `--request-rate`: 请求发送速率 (req/s, 默认: inf)
- `--max-concurrency`: 最大并发数
- `--burstiness`: 请求突发性因子 (默认: 1.0)

### 输出控制参数

- `--save-result`: 保存结果到JSON文件
- `--result-dir`: 结果保存目录
- `--percentile-metrics`: 百分位数指标选择
- `--metric-percentiles`: 百分位数值选择

## 📈 结果解读

### 控制台输出示例

```
=============== Serving Benchmark Result ===============
Successful requests:                     1000
Benchmark duration (s):                 45.23
Total input tokens:                      512000
Total generated tokens:                  128000
Request throughput (req/s):              22.11
Output token throughput (tok/s):         2830.45
Total Token throughput (tok/s):          14152.25

----------------------- Time to First Token -----------------------
Mean TTFT (ms):                          125.34
Median TTFT (ms):                        118.67
P99 TTFT (ms):                          245.89

----------------------- Time per Output Token ----------------------
Mean TPOT (ms):                          35.67
Median TPOT (ms):                        32.45
P99 TPOT (ms):                          78.23
```

### 性能评估标准

#### 优秀性能指标参考值 (仅供参考)

- **TTFT**: < 100ms (优秀), < 200ms (良好), > 500ms (需优化)
- **TPOT**: < 50ms (优秀), < 100ms (良好), > 200ms (需优化)
- **吞吐量**: 根据硬件配置和模型大小而定

## 🔧 高级配置

### DCU优化配置

```bash
# NCCL通信优化
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_P2P_LEVEL=SYS

# NUMA绑定
export VLLM_NUMA_BIND=0
```

### 服务质量目标 (SLO) 配置

```bash
python benchmark_serving.py \
    --goodput ttft:100 tpot:50 e2el:1000 \
    # 其他参数...
```

## 📝 最佳实践

### 1. 测试前准备
- 确保DCU驱动和运行时环境正确安装
- 预热模型服务，避免冷启动影响
- 监控系统资源使用情况

### 2. 测试参数选择
- 从小批量开始，逐步增加负载
- 选择代表性的输入输出长度组合
- 考虑真实业务场景的请求模式

### 3. 结果分析
- 关注P99等高百分位数指标
- 分析不同负载下的性能变化趋势
- 结合系统资源使用情况进行综合评估

## 🐛 常见问题

### Q: 测试过程中出现连接超时
A: 检查vLLM服务是否正常启动，调整AIOHTTP_TIMEOUT设置

### Q: 内存不足错误
A: 降低--gpu-memory-utilization参数或减少并发数

### Q: 性能指标异常
A: 检查模型是否正确加载，确认DCU设备配置

## 📚 参考资料

- [vLLM官方文档](https://docs.vllm.ai/)
- [海光DCU开发指南](https://developer.hygon.cn/)
- [大模型推理优化最佳实践](https://github.com/vllm-project/vllm)

## 🎓 初学者完整教程

### 第一步: 环境搭建

1. **安装基础依赖**
```bash
# 创建虚拟环境 (推荐)
conda create -n vllm-benchmark python=3.10
conda activate vllm-benchmark

# 安装核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
pip install vllm transformers datasets aiohttp tqdm numpy pandas
```

2. **配置DCU环境**
```bash
# 检查DCU设备
rocm-smi

# 设置环境变量
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_P2P_LEVEL=SYS
```

### 第二步: 准备模型

1. **下载模型**
```bash
# 使用HuggingFace Hub下载
huggingface-cli download microsoft/DialoGPT-medium --local-dir ./models/DialoGPT-medium

# 或使用git lfs
git lfs clone https://huggingface.co/microsoft/DialoGPT-medium ./models/DialoGPT-medium
```

2. **验证模型**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./models/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("./models/DialoGPT-medium")
print("模型加载成功!")
```

### 第三步: 启动vLLM服务

1. **修改服务启动脚本**
```bash
# 编辑 server.sh
vim server.sh

# 修改模型路径
vllm serve ./models/DialoGPT-medium \
    --trust-remote-code \
    --dtype float16 \
    --max-model-len 2048 \
    -tp 4 \
    --gpu-memory-utilization 0.9 \
    --port 8000
```

2. **启动服务**
```bash
bash server.sh
```

3. **验证服务**
```bash
# 测试API连通性
curl http://localhost:8000/v1/models

# 发送测试请求
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./models/DialoGPT-medium",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```

### 第四步: 运行基准测试

1. **简单测试**
```bash
# 最基础的测试命令
python benchmark_serving.py \
    --backend vllm \
    --model ./models/DialoGPT-medium \
    --dataset-name random \
    --num-prompts 10 \
    --random-input-len 100 \
    --random-output-len 50
```

2. **查看结果**
```
=============== Serving Benchmark Result ===============
Successful requests:                     10
Benchmark duration (s):                 5.23
Total input tokens:                      1000
Total generated tokens:                  500
Request throughput (req/s):              1.91
Output token throughput (tok/s):         95.60
Total Token throughput (tok/s):          286.81

----------------------- Time to First Token -----------------------
Mean TTFT (ms):                          125.34
Median TTFT (ms):                        118.67
P99 TTFT (ms):                          245.89
```

### 第五步: 理解性能指标

1. **延迟指标解读**
- **TTFT < 200ms**: 用户感觉响应很快
- **TTFT 200-500ms**: 可接受的响应速度
- **TTFT > 500ms**: 用户会感觉明显延迟

2. **吞吐量指标解读**
- **Request Throughput**: 系统每秒能处理多少个请求
- **Token Throughput**: 系统每秒能生成多少个token
- **数值越高表示性能越好**

### 第六步: 高级测试配置

1. **压力测试**
```bash
python benchmark_serving.py \
    --backend vllm \
    --model ./models/DialoGPT-medium \
    --dataset-name random \
    --num-prompts 1000 \
    --request-rate 10 \
    --max-concurrency 50 \
    --random-input-len 512 \
    --random-output-len 128
```

2. **真实数据测试**
```bash
# 下载ShareGPT数据集
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# 使用真实对话数据测试
python benchmark_serving.py \
    --backend vllm \
    --model ./models/DialoGPT-medium \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 100
```

### 第七步: 结果分析和优化

1. **性能瓶颈识别**
```bash
# 监控系统资源
htop
rocm-smi

# 分析日志
tail -f /var/log/vllm.log
```

2. **优化建议**
- **内存不足**: 降低 `--gpu-memory-utilization`
- **延迟过高**: 减少 `--max-model-len` 或增加GPU数量
- **吞吐量低**: 调整 `--max-concurrency` 和批处理大小

### 常见错误和解决方案

1. **CUDA/HIP错误**
```bash
# 检查驱动
rocm-smi
export HIP_VISIBLE_DEVICES=0,1,2,3
```

2. **内存不足**
```bash
# 降低内存使用
--gpu-memory-utilization 0.8
--max-model-len 1024
```

3. **连接超时**
```bash
# 增加超时时间
export AIOHTTP_TIMEOUT=3600
```

这个教程将帮助初学者从零开始掌握大模型推理性能测试的完整流程。
