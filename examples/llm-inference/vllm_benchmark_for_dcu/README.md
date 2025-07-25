# 海光DCU大模型推理性能基准测试工具

## 项目概述

这是一个专门为海光DCU优化的vLLM大模型推理性能测试工具包，基于vLLM 0.8.5版本开发。该工具提供了完整的大模型推理性能评估解决方案，支持多种测试场景和详细的性能指标分析。

## 主要特性

### 🚀 DCU硬件优化
- 针对海光DCU硬件进行了深度优化
- 支持多卡DCU张量并行推理
- 优化的NCCL通信配置
- NUMA绑定优化，提升内存访问效率

### 📊 全面的性能指标
- **吞吐量指标**: 请求吞吐量、输出吞吐量、总token吞吐量
- **延迟指标**: TTFT (首次token时间)、TPOT (每token时间)、ITL (迭代延迟)、E2EL (端到端延迟)
- **统计分析**: 均值、中位数、标准差、百分位数 (P50, P90, P95, P99)
- **服务质量**: Goodput指标，评估满足SLO的请求比例

### 🔧 灵活的测试配置
- 支持多种批次大小测试 (1-64)
- 可配置的输入输出长度组合
- 多种数据集支持 (Random, ShareGPT, Sonnet等)
- 请求速率和并发控制

### 🌐 多后端支持
- vLLM (主要目标)
- TGI (Text Generation Inference)
- TensorRT-LLM
- OpenAI兼容API
- 其他推理后端

## 项目结构

```
vllm_benchmark_for_dcu/
├── code/                           # 核心代码目录
│   ├── benchmark_serving.py       # 主测试脚本 - 核心基准测试逻辑
│   ├── backend_request_func.py    # 后端请求处理 - 多后端统一接口
│   ├── benchmark_dataset.py       # 数据集处理 - 测试数据生成和管理
│   ├── benchmark_utils.py         # 工具函数 - 结果格式化和数据处理
│   ├── server.sh                  # vLLM服务启动脚本
│   ├── test.sh                    # 自动化测试脚本
│   └── log/                       # 测试日志目录
├── 大模型性能测试指南.md           # 详细使用指南
└── README.md                      # 项目说明文档
```

## 核心组件说明

### 1. benchmark_serving.py - 主测试脚本
- **功能**: 基准测试的核心控制逻辑
- **特性**: 
  - 异步请求处理，支持高并发测试
  - 详细的性能指标计算和统计分析
  - 灵活的测试参数配置
  - 完整的错误处理和日志记录

### 2. backend_request_func.py - 后端请求处理
- **功能**: 提供统一的多后端请求接口
- **特性**:
  - 支持多种推理后端的统一调用
  - 流式响应处理和性能指标收集
  - 异步HTTP通信优化
  - 完善的错误处理机制

### 3. benchmark_dataset.py - 数据集处理
- **功能**: 测试数据的生成和管理
- **特性**:
  - 支持多种数据集格式
  - 随机数据生成功能
  - 多模态数据支持
  - 灵活的数据采样策略

### 4. benchmark_utils.py - 工具函数
- **功能**: 结果处理和格式化工具
- **特性**:
  - PyTorch基准测试格式转换
  - JSON序列化优化
  - 数据清理和标准化

## 快速开始

### 1. 环境准备
```bash
# 使用指定的Docker镜像
docker run -it \
  --name=llm-benchmark \
  -v /data:/data \
  --ipc=host \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/mkfd \
  --device=/dev/dri \
  --shm-size=64G \
  image.sourcefind.cn:5000/dcu/admin/base/vllm:0.8.5-ubuntu22.04-dtk25.04.1-rc5-das1.6-py3.10-20250711 \
  /bin/bash
```

### 2. 启动vLLM服务
```bash
# 修改server.sh中的模型路径，然后执行
bash server.sh
```

### 3. 运行基准测试
```bash
# 修改test.sh中的测试参数，然后执行
bash test.sh
```

### 4. 查看结果
- **汇总结果**: `r1-awq-0705.csv` - 包含所有测试配置的性能指标
- **详细日志**: `log/` 目录 - 每个测试配置的详细日志

## 性能指标说明

### 吞吐量指标
- **Request Throughput**: 每秒处理的请求数 (req/s)
- **Output Throughput**: 每秒生成的token数 (tok/s)  
- **Total Token Throughput**: 每秒处理的总token数 (tok/s)

### 延迟指标
- **TTFT (Time To First Token)**: 从请求到首个token的时间 (ms)
- **TPOT (Time Per Output Token)**: 生成每个token的平均时间 (ms)
- **ITL (Inter-Token Latency)**: token间的延迟 (ms)
- **E2EL (End-to-End Latency)**: 端到端总延迟 (ms)

## 测试配置建议

### 批次大小选择
- **小批次 (1-4)**: 测试低延迟场景
- **中批次 (8-16)**: 测试平衡性能场景  
- **大批次 (32-64)**: 测试高吞吐量场景

### 输入输出长度
- **短文本 (512/512)**: 测试快速响应场景
- **长文本 (1024/1024)**: 测试复杂推理场景
- **自定义长度**: 根据实际应用需求调整

## 注意事项

1. **硬件要求**: 确保有足够的DCU显存支持模型加载
2. **网络配置**: 确保vLLM服务端口可访问
3. **模型路径**: 正确配置模型文件路径
4. **环境变量**: 根据硬件配置调整DCU和NCCL相关环境变量

## 故障排除

### 常见问题
1. **显存不足**: 调整 `--gpu-memory-utilization` 参数
2. **端口冲突**: 修改 `--port` 参数
3. **模型加载失败**: 检查模型路径和权限
4. **通信错误**: 检查NCCL和网络配置

### 日志分析
- 查看 `log/` 目录下的详细日志文件
- 关注错误信息和性能警告
- 使用 `grep` 命令快速定位问题

## 贡献指南

欢迎提交Issue和Pull Request来改进这个工具。请确保：
1. 代码符合项目的编码规范
2. 添加适当的注释和文档
3. 测试新功能的兼容性

## 许可证

本项目采用Apache 2.0许可证，详见LICENSE文件。
