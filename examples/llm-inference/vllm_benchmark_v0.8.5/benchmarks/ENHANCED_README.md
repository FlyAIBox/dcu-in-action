# vLLM 0.8.5 增强版基准测试指南

## 📋 概述

这是一个为大模型推理压测初学者设计的增强版vLLM基准测试工具，在原有功能基础上添加了：

- ✅ **配置文件支持** - 使用YAML文件管理测试参数
- ✅ **可视化展示** - 自动生成图表和HTML报告
- ✅ **详细指标** - 全面的性能分析和统计
- ✅ **初学者友好** - 详细的使用说明和示例

## 🏗️ 代码架构

### 核心模块

| 模块 | 功能 | 说明 |
|------|------|------|
| `enhanced_benchmark_serving.py` | 增强版在线服务测试 | 支持配置文件和可视化 |
| `config_manager.py` | 配置管理 | YAML配置文件解析和参数合并 |
| `visualization.py` | 可视化模块 | 生成图表和HTML报告 |
| `benchmark_configs/` | 配置模板 | 预设的测试配置文件 |

### 原始模块

| 模块 | 功能 |
|------|------|
| `benchmark_serving.py` | 在线服务基准测试 |
| `benchmark_throughput.py` | 离线吞吐量测试 |
| `benchmark_dataset.py` | 数据集处理 |
| `backend_request_func.py` | 后端请求函数 |

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install matplotlib seaborn pyyaml pandas numpy

# 确保vLLM服务正在运行
vllm serve meta-llama/Llama-2-7b-hf --disable-log-requests
```

### 2. 使用配置文件运行测试

```bash
# 使用预设配置运行在线服务测试
python enhanced_benchmark_serving.py --config benchmark_configs/serving_test.yaml

# 使用自定义配置
python enhanced_benchmark_serving.py --config my_config.yaml
```

### 3. 使用命令行参数

```bash
# 基本测试
python enhanced_benchmark_serving.py \
    --model meta-llama/Llama-2-7b-hf \
    --backend vllm \
    --dataset-name random \
    --num-prompts 100 \
    --request-rate 2.0

# 禁用可视化
python enhanced_benchmark_serving.py \
    --config benchmark_configs/serving_test.yaml \
    --no-visualization
```

## 📊 关键性能指标解释

### 吞吐量指标
- **请求吞吐量 (Request Throughput)**: 每秒处理的请求数 (req/s)
- **Token吞吐量 (Token Throughput)**: 每秒生成的token数 (tok/s)
- **总Token吞吐量**: 输入+输出token的总吞吐量

### 延迟指标
- **TTFT (Time To First Token)**: 首个token生成时间，反映模型响应速度
- **TPOT (Time Per Output Token)**: 平均每个输出token的生成时间
- **ITL (Inter-Token Latency)**: token间延迟，反映生成流畅度
- **E2EL (End-to-End Latency)**: 端到端延迟，完整请求处理时间

### 百分位数
- **P50 (中位数)**: 50%的请求延迟低于此值
- **P90**: 90%的请求延迟低于此值
- **P95**: 95%的请求延迟低于此值  
- **P99**: 99%的请求延迟低于此值

## 📁 配置文件详解

### 基本配置结构

```yaml
# 模型配置
model: "meta-llama/Llama-2-7b-hf"
backend: "vllm"
endpoint: "/v1/completions"
host: "localhost"
port: 8000

# 数据集配置
dataset_name: "sharegpt"
dataset_path: "./ShareGPT_V3_unfiltered_cleaned_split.json"
num_prompts: 100
input_len: 1024
output_len: 128

# 测试参数
request_rate: 2.0  # 每秒请求数，inf表示无限制
burstiness: 1.0    # 突发性参数
seed: 0            # 随机种子

# 采样参数
temperature: 1.0
top_p: 1.0
top_k: -1

# 输出配置
save_result: true
result_dir: "./benchmark_results"
enable_visualization: true

# 高级配置
tensor_parallel_size: 1
pipeline_parallel_size: 1
trust_remote_code: false
```

### 预设配置模板

1. **serving_test.yaml** - 在线服务测试
2. **throughput_test.yaml** - 吞吐量测试
3. **vision_test.yaml** - 视觉模型测试

## 📈 可视化功能

### 自动生成的图表

1. **吞吐量指标柱状图** - 直观显示各项吞吐量指标
2. **延迟分布直方图** - TTFT和TPOT的分布情况
3. **百分位数对比图** - 不同百分位数的延迟对比
4. **请求统计饼图** - 成功请求和token分布
5. **配置信息表格** - 测试配置一览

### HTML报告

- 包含所有图表的完整HTML报告
- 详细的配置信息和性能指标
- 便于分享和存档

## 🎯 使用场景和最佳实践

### 1. 性能基线测试

```bash
# 建立性能基线
python enhanced_benchmark_serving.py \
    --config benchmark_configs/serving_test.yaml \
    --num-prompts 1000 \
    --request-rate inf
```

### 2. 负载压力测试

```bash
# 测试不同负载下的性能
for rate in 1 2 5 10; do
    python enhanced_benchmark_serving.py \
        --config benchmark_configs/serving_test.yaml \
        --request-rate $rate \
        --result-dir "./results/rate_$rate"
done
```

### 3. 模型对比测试

```yaml
# config_model_a.yaml
model: "meta-llama/Llama-2-7b-hf"
result_dir: "./comparison/model_a"

# config_model_b.yaml  
model: "meta-llama/Llama-2-13b-hf"
result_dir: "./comparison/model_b"
```

### 4. 数据集影响分析

```bash
# 测试不同数据集的影响
for dataset in "random" "sharegpt" "sonnet"; do
    python enhanced_benchmark_serving.py \
        --dataset-name $dataset \
        --result-dir "./results/$dataset"
done
```

## 🔧 故障排除

### 常见问题

1. **连接失败**
   ```
   错误: 无法连接到服务器
   解决: 确保vLLM服务正在运行，检查host和port配置
   ```

2. **数据集加载失败**
   ```
   错误: 数据集文件不存在
   解决: 检查dataset_path是否正确，下载对应数据集
   ```

3. **可视化失败**
   ```
   错误: 缺少matplotlib依赖
   解决: pip install matplotlib seaborn
   ```

### 调试技巧

1. **使用小数据集测试**: 先用`--num-prompts 10`测试配置
2. **检查服务器日志**: 查看vLLM服务器的输出日志
3. **逐步增加负载**: 从低请求速率开始测试

## 📚 进阶功能

### 1. 自定义数据集

```python
# 创建自定义数据集类
class CustomDataset(BenchmarkDataset):
    def load_data(self):
        # 实现数据加载逻辑
        pass
    
    def sample_requests(self, num_requests, tokenizer):
        # 实现请求采样逻辑
        pass
```

### 2. 扩展可视化

```python
# 添加自定义图表
def create_custom_chart(metrics, config):
    # 实现自定义可视化逻辑
    pass
```

### 3. 结果分析脚本

```python
# 分析多次测试结果
import json
import glob

results = []
for file in glob.glob("./results/*/results.json"):
    with open(file) as f:
        results.append(json.load(f))

# 进行对比分析
```

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个工具！

### 开发环境设置

```bash
git clone <repository>
cd vllm_benchmark_enhanced
pip install -r requirements.txt
```

### 测试

```bash
python -m pytest tests/
```

## 📄 许可证

Apache-2.0 License

---

**💡 提示**: 这个增强版工具旨在帮助初学者更好地理解和使用vLLM基准测试。如果您有任何问题或建议，请随时提出！
