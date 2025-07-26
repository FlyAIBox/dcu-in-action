# 🚀 海光DCU大模型推理性能基准测试框架

这是一个专门为海光DCU优化的vLLM大模型推理性能测试框架，基于官方 [vLLM benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks) 开发，专门面向**大模型推理压测初学者**。

## 🎯 项目目标

本框架旨在为初学者提供完整的大模型推理性能评估解决方案，包括：

### 📊 核心性能指标
* **TTFT (Time To First Token)**: 首次响应时间 - 用户感知的响应速度
* **TPOT (Time Per Output Token)**: 每token生成时间 - 生成速度稳定性
* **ITL (Inter-Token Latency)**: 迭代延迟 - 流式输出流畅度
* **E2EL (End-to-End Latency)**: 端到端延迟 - 总体处理时间
* **吞吐量指标**: 请求吞吐量、输出吞吐量、总token吞吐量
* **并发能力**: 系统能稳定支持的最大并发请求数

### 🔧 技术特性
* **多后端支持**: vLLM、TGI、OpenAI API、TensorRT-LLM等
* **多数据集支持**: ShareGPT、Random、Sonnet、HuggingFace等
* **DCU优化**: 针对海光DCU硬件的专门优化配置
* **并发控制**: 支持请求速率控制和最大并发限制
* **可视化分析**: 丰富的图表和交互式仪表板
* **自动化测试**: 批量测试和结果聚合功能

## 🎓 面向初学者的设计

### 为什么选择这个框架？
1. **零基础友好**: 详细的中文文档和注释
2. **概念解释**: 深入浅出地解释TTFT、TPOT等专业术语
3. **实战导向**: 提供完整的测试流程和最佳实践
4. **可视化结果**: 直观的图表帮助理解性能数据
5. **DCU优化**: 专门针对海光DCU环境优化

### 学习路径
```
基础概念学习 → 环境搭建 → 简单测试 → 并发测试 → 结果分析 → 性能优化
```

## 📚 完整文档体系

本框架提供了完整的文档体系，帮助初学者从零开始掌握大模型推理性能测试：

### 🎓 学习文档
- **[BEGINNER_COMPREHENSIVE_GUIDE.md](BEGINNER_COMPREHENSIVE_GUIDE.md)** - 完整初学者指南
  - 核心概念详解 (TTFT、TPOT、ITL、E2EL等)
  - 系统架构深度解析
  - 性能指标详细说明
  - 使用流程完整指南

### 🔧 技术文档
- **[CODE_STRUCTURE_GUIDE.md](CODE_STRUCTURE_GUIDE.md)** - 代码结构详解
  - 核心模块功能分析
  - 关键算法实现原理
  - 数据流向详细说明
  - 扩展开发指南

### 🛠️ 实战文档
- **[PRACTICAL_EXAMPLES.md](PRACTICAL_EXAMPLES.md)** - 实战示例集合
  - 基础到高级的完整示例
  - 不同场景的测试配置
  - 结果分析和故障排除
  - 最佳实践清单

### 📊 架构图表
- **系统架构图** - 展示整体框架结构和组件关系
- **数据流程图** - 详细说明测试执行的完整流程
- **性能指标图** - 可视化展示各项指标的计算方法

---

## 🚀 快速开始

### 步骤1: 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 启动vLLM服务 (示例)
vllm serve microsoft/DialoGPT-medium \
  --host 0.0.0.0 \
  --port 8000 \
  --swap-space 16 \
  --disable-log-requests
```

### 步骤2: 配置测试参数

编辑 `combos.yaml` 文件：

```yaml
# 基础配置
model: "microsoft/DialoGPT-medium"        # 模型名称
base_url: "http://localhost:8000"         # vLLM服务URL
tokenizer: "microsoft/DialoGPT-medium"    # 分词器路径

# 测试场景 [输入长度, 输出长度]
input_output:
  - [256, 256]    # 短文本场景
  - [1024, 512]   # 长文本场景

# 并发测试 [并发数, 请求总数]
concurrency_prompts:
  - [1, 10]       # 低并发测试
  - [4, 40]       # 中等并发测试
```

### 步骤3: 运行测试

```bash
# 方式1: 批量测试 (推荐)
python3 run_sweep.py

# 方式2: 单次测试
python3 benchmark_serving.py \
  --backend vllm \
  --model microsoft/DialoGPT-medium \
  --dataset-name random \
  --num-prompts 100 \
  --request-rate 10
```

### 步骤4: 分析结果

```bash
# 聚合所有测试结果
python3 aggregate_result.py

# 生成可视化图表
python3 visualize.py
```

---

## 📊 核心功能详解

### 🔧 环境设置

支持多种安装方式：

```bash
# 基础安装
pip install -r requirements.txt

# 开发环境安装 (包含可视化依赖)
pip install -r requirements.txt
pip install plotly seaborn matplotlib
```

### ⚙️ 配置说明

`combos.yaml` 配置文件详解：

* `model`: 要测试的模型名称 (必须与vLLM服务中的模型一致)
* `base_url`: vLLM服务器的基础URL
* `tokenizer`: 分词器路径 (用于精确计算token数量)
* `input_output`: 测试的输入输出长度组合列表
* `concurrency_prompts`: 并发数和请求数的组合列表

---

## 📊 性能指标详解

### 🎯 核心延迟指标

| 指标 | 英文全称 | 含义 | 重要性 | 典型值 |
|------|----------|------|--------|--------|
| **TTFT** | Time To First Token | 首个token生成时间 | 用户感知响应速度 | 50-500ms |
| **TPOT** | Time Per Output Token | 每个token生成时间 | 生成速度稳定性 | 10-100ms |
| **ITL** | Inter-Token Latency | token间延迟 | 流式输出流畅度 | 5-50ms |
| **E2EL** | End-to-End Latency | 端到端总延迟 | 整体用户体验 | 1-30s |

### 🚀 吞吐量指标

| 指标 | 含义 | 计算公式 | 用途 |
|------|------|----------|------|
| **请求吞吐量** | 每秒处理请求数 | completed_requests / duration | 并发处理能力 |
| **Token吞吐量** | 每秒生成token数 | total_tokens / duration | 实际生产能力 |
| **Goodput** | 满足SLA的有效吞吐量 | good_requests / duration | 服务质量评估 |

### 📈 统计分析指标

- **平均值 (Mean)**: 所有样本的算术平均
- **中位数 (Median)**: 50%分位数，更稳定的中心趋势
- **标准差 (Std)**: 数据分散程度的度量
- **百分位数 (P90/P95/P99)**: 性能保证水平的关键指标

---

## 🚀 运行基准测试

### 批量测试 (推荐)

根据 `combos.yaml` 配置运行所有测试组合：

```bash
python3 run_sweep.py
```

结果保存在 `results/` 目录，每个测试用例对应一个 `.json` 文件。

### 单次测试

针对特定场景的定制化测试：

```bash
# 基础测试
python3 benchmark_serving.py \
  --backend vllm \
  --model your_model_name \
  --dataset-name random \
  --num-prompts 100

# 高级测试 (包含SLA评估)
python3 benchmark_serving.py \
  --backend vllm \
  --model your_model_name \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500 \
  --request-rate 15 \
  --max-concurrency 8 \
  --goodput ttft:200 tpot:50 e2el:5000 \
  --save-result
```

---

## 📊 结果聚合与分析

### 聚合测试结果

所有基准测试完成后，运行：

```bash
python3 aggregate_result.py
```

生成汇总文件 `aggregate_results.csv`，包含所有测试的关键指标。

---

## 📈 数据可视化

本项目提供了功能强大的数据可视化工具，帮助您深入分析基准测试结果。

### 🚀 快速开始

生成所有可视化图表：

```bash
python3 visualize.py
```

### 📊 可视化功能

#### 1. 吞吐量分析
```bash
python3 visualize.py --throughput
```
- 并发数与吞吐量关系分析
- 输入长度对性能的影响
- 时间序列趋势分析
- 效率热力图

#### 2. 延迟分析
```bash
python3 visualize.py --latency
```
- TTFT (首个Token时间) 分布分析
- TPOT (每Token时间) vs 并发数
- 延迟组件对比 (TTFT/TPOT/ITL)
- 端到端延迟分析

#### 3. 交互式仪表板
```bash
python3 visualize.py --interactive
```
- 多维度交互式图表
- 实时数据筛选和缩放
- 悬停提示和详细信息
- HTML格式，支持Web浏览

#### 4. 性能报告
```bash
python3 visualize.py --report
```
- 关键性能指标摘要
- 最佳配置推荐
- 性能优化建议
- 详细的数据分析

### 🔧 高级用法

#### 自定义数据源
```bash
python3 visualize.py --csv /path/to/your/data.csv
```

#### 编程接口
```python
from benchmark_visualizer import BenchmarkVisualizer

# 创建可视化器
visualizer = BenchmarkVisualizer("results/aggregate_results.csv")

# 生成特定图表
visualizer.plot_throughput_analysis()
visualizer.plot_latency_analysis()
visualizer.plot_interactive_dashboard()

# 获取性能报告
report = visualizer.generate_performance_report()
print(report)

# 一键生成所有图表
visualizer.generate_all_charts()
```

#### 演示功能
```bash
python3 visualize.py --demo
```
运行完整的演示，包括：
- 基础功能展示
- 自定义分析示例
- 性能洞察分析
- 高级功能演示

### 📁 输出文件

所有生成的文件保存在 `figures/` 目录：

```
figures/
├── throughput_analysis.png      # 吞吐量分析图
├── latency_analysis.png         # 延迟分析图
├── interactive_dashboard.html   # 交互式仪表板
├── performance_report.txt       # 性能分析报告
└── ...
```

---

## 📁 项目结构

```
vllm_benchmark_serving/
├── 🔧 核心文件
│   ├── backend_request_func.py       # 后端请求处理
│   ├── benchmark_serving.py          # 基准测试主程序
│   ├── benchmark_dataset.py          # 数据集处理
│   ├── benchmark_utils.py            # 工具函数
│   ├── aggregate_result.py           # 结果聚合
│   └── combos.yaml                   # 配置文件
├── 📊 可视化工具
│   ├── benchmark_visualizer.py       # 可视化工具类
│   ├── visualize.py                  # 快速启动脚本
│   └── example_usage.py              # 使用示例
├── ⚙️ 配置与依赖
│   ├── requirements.txt              # Python依赖
│   ├── run_sweep.py                  # 批量运行脚本
│   └── README.md                     # 项目文档
├── 📈 输出目录
│   ├── results/                      # 基准测试结果
│   │   ├── aggregate_results.csv     # 聚合结果
│   │   ├── run_1.json               # 单次测试结果
│   │   └── ...
│   └── figures/                      # 可视化图表
│       ├── throughput_analysis.png
│       ├── latency_analysis.png
│       ├── interactive_dashboard.html
│       └── performance_report.txt
```

---

## 📣 注意事项

* 在开始基准测试之前，请确保 vLLM 服务器处于活动状态并在指定的 `base_url` 可访问。
* 您可以在 `combos.yaml` 中自定义提示词、令牌长度和并发范围。

---

如果您希望在 README 中添加示例配置、图表或结果可视化，请告诉我！


vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --host 0.0.0.0 \
  --port 8002