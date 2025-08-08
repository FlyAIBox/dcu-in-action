# DeepSeek 32B 大模型微调实战指南

> 面向医院技术人员的大模型微调完整教程 - 从数据准备到生产部署

## 📋 目录

- [项目概述](#项目概述)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [教程内容](#教程内容)
- [实战案例](#实战案例)
- [常见问题](#常见问题)
- [参考资源](#参考资源)

## 🎯 项目概述

本项目是一个完整的大模型微调实战教程，专为医院技术人员设计。通过循序渐进的方式，从大模型基础概念到实际部署应用，帮助初学者掌握：

- **数据集构建**：如何准备和处理行业特定数据
- **模型微调**：使用LoRA技术高效微调DeepSeek-R1模型
- **推理部署**：基于vLLM的高性能推理服务
- **性能评估**：模型效果评估与压测方法

### 🏥 应用场景

- 眼科诊断辅助系统
- 医疗问答机器人
- 病历智能分析
- 医学知识问答

## 🛠 环境要求

### 硬件配置
- **GPU**: DCU K100-AI (8卡) 或同等算力
- **内存**: 建议 128GB+
- **存储**: 500GB+ 可用空间

### 软件环境
- **操作系统**: Ubuntu 22.04.4
- **Python**: 3.10
- **DCU DTK**: 25.04

### 核心工具链
- **Llamafactory**: 微调框架
- **LoRA**: 高效微调技术
- **DeepSpeed ZeRO-3**: 显存优化
- **Easy Dataset**: 数据集制作工具
- **vLLM 0.8.5**: 高性能推理引擎

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone -b finetuning0808  https://github.com/FlyAIBox/dcu-in-action.git
cd dcu-in-action

# 安装依赖
pip install -r requirements-dtk2504.txt

```

### 2. 数据准备

```bash
# 下载示例数据集
python scripts/download_sample_data.py

# 数据预处理
python scripts/prepare_dataset.py --input data/raw --output data/processed
```

### 3. 开始微调

```bash
# 启动微调训练
python scripts/train.py --config configs/deepseek_lora.yaml
```

## 📚 教程内容

### 第一部分：大模型与微调基础概念

#### 🎯 学习目标
- 理解大模型在医疗领域的应用价值
- 掌握微调的基本概念和必要性
- 了解LoRA和DeepSpeed ZeRO-3核心技术

#### 📖 内容大纲
1. **课程引入**：大模型在眼科的应用前景
   - 医疗AI发展现状
   - 大模型的优势与挑战
   - 实际应用案例分析

2. **微调基本概念**
   - 什么是微调？
   - 为什么要微调？
   - 微调 vs 预训练的区别

3. **核心技术详解**
   - **LoRA (Low-Rank Adaptation)**：高效微调技术
   - **DeepSpeed ZeRO-3**：显存优化策略
   - 技术原理与实现细节

#### 🛠 实践环节
- 环境配置检查
- 工具链安装验证
- 基础概念问答

---

### 第二部分：数据准备与实战环境搭建

#### 🎯 学习目标
- 掌握行业数据集构建方法
- 学会使用Easy Dataset制作数据集
- 完成Llamafactory环境配置

#### 📖 内容大纲
1. **数据准备**
   - 行业特定问题分析
   - 数据收集策略
   - 数据质量评估

2. **工具实战**
   - Easy Dataset工具介绍
   - 眼科案例数据集制作
   - 数据格式标准化

3. **环境配置**
   - Llamafactory安装与配置
   - 依赖管理最佳实践
   - 常见问题排查

#### 🛠 实践环节
```bash
# 数据集制作示例
python tools/easy_dataset.py \
    --input data/medical_qa.json \
    --output data/formatted_dataset \
    --format llamafactory

# 数据质量检查
python tools/validate_dataset.py --dataset data/formatted_dataset
```

---

### 第三部分：DeepSeek-R1 模型微调实战

#### 🎯 学习目标
- 掌握模型微调完整流程
- 学会监控训练关键指标
- 理解模型评估方法

#### 📖 内容大纲
1. **训练与评估**
   - 微调参数配置
   - 训练过程监控
   - 关键指标解读

2. **评估方法**
   - 自动评估指标
   - 人工验证方法
   - 评估结果分析

#### 🛠 实践环节
```bash
# 启动微调训练
llamafactory-cli train \
    --stage sft \
    --model_name deepseek-r1-distill-llama-8b \
    --dataset medical_qa \
    --template deepseek \
    --finetuning_type lora \
    --output_dir saves/deepseek-medical \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --max_samples 1000 \
    --max_grad_norm 1.0 \
    --quantization_bit 4 \
    --loraplus_lr_ratio 16.0 \
    --fp16

# 模型评估
llamafactory-cli eval \
    --model_name deepseek-r1-distill-llama-8b \
    --adapter_name_or_path saves/deepseek-medical \
    --dataset medical_eval \
    --template deepseek \
    --batch_size 4 \
    --max_samples 100
```

---

### 第四部分：微调后模型的推理与部署

#### 🎯 学习目标
- 掌握vLLM高性能推理部署
- 学会进行模型性能测试
- 了解生产环境部署要点

#### 📖 内容大纲
1. **高性能推理**
   - vLLM 0.8.5部署配置
   - 推理服务优化
   - 并发处理策略

2. **性能测试**
   - 推理速度测试
   - 并发压测方法
   - 资源使用监控

#### 🛠 实践环节
```bash
# 启动vLLM推理服务
python -m vllm.entrypoints.openai.api_server \
    --model saves/deepseek-medical \
    --served-model-name deepseek-medical \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.8

# 推理测试
python scripts/test_inference.py \
    --endpoint http://localhost:8000/v1/chat/completions \
    --model deepseek-medical \
    --test-file data/test_cases.json

# 压测
python scripts/benchmark.py \
    --endpoint http://localhost:8000/v1/chat/completions \
    --concurrent-users 10 \
    --duration 300
```

---

### 第五部分：案例分析与总结

#### 🎯 学习目标
- 回顾端到端流程
- 总结核心知识点
- 规划后续学习路径

#### 📖 内容大纲
1. **流程回顾**
   - 从需求到部署的完整流程
   - 关键决策点分析
   - 最佳实践总结

2. **课程总结**
   - 核心技术要点
   - 常见问题解决方案
   - 未来发展趋势

3. **答疑交流**
   - 技术难点讨论
   - 实际应用场景分析
   - 进阶学习建议

## 💡 实战案例

### 案例1：眼科问答系统

**场景描述**：构建一个专业的眼科医疗问答系统

**数据准备**：
```json
{
  "instruction": "患者咨询眼部症状",
  "input": "最近几天眼睛干涩，有异物感，是什么原因？",
  "output": "根据您描述的症状，可能是干眼症。建议：1. 减少用眼时间；2. 使用人工泪液；3. 保持室内湿度；4. 如症状持续，请及时就医。"
}
```

**微调配置**：
```yaml
# configs/ophthalmology_qa.yaml
model_name: deepseek-r1-distill-llama-8b
dataset: ophthalmology_qa
template: deepseek
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
learning_rate: 5e-5
num_train_epochs: 3
max_samples: 2000
```

**效果评估**：
- 专业准确性：85%+
- 回答完整性：90%+
- 安全性检查：100%

### 案例2：病历智能分析

**场景描述**：自动分析眼科病历，提取关键信息

**实现步骤**：
1. 病历数据标准化
2. 关键信息标注
3. 模型微调训练
4. 推理服务部署

## ❓ 常见问题

### Q1: 显存不足怎么办？
**A**:
- 使用DeepSpeed ZeRO-3优化
- 减小batch_size
- 启用梯度检查点
- 使用量化技术

### Q2: 训练速度太慢？
**A**:
- 增加GPU数量
- 优化数据加载
- 使用混合精度训练
- 调整学习率调度

### Q3: 模型效果不理想？
**A**:
- 检查数据质量
- 调整超参数
- 增加训练数据
- 尝试不同的微调策略

### Q4: 如何评估模型安全性？
**A**:
- 设置安全过滤器
- 人工审核样本
- 建立评估基准
- 持续监控输出

## 📖 参考资源

### 官方文档
- [DeepSeek官方文档](https://deepseek.com)
- [LlamaFactory使用指南](https://github.com/hiyouga/LLaMA-Factory)
- [vLLM部署文档](https://docs.vllm.ai)

### 学术论文
- LoRA: Low-Rank Adaptation of Large Language Models
- DeepSpeed: System Optimizations Enable Training Deep Learning Models
- Efficient Large Language Model Training and Inference

### 社区资源
- [Hugging Face Model Hub](https://huggingface.co)
- [医疗AI开发者社区](https://medical-ai.dev)
- [大模型微调最佳实践](https://llm-tuning.guide)

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进本项目！

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

**联系我们**：如有技术问题，请提交Issue或发送邮件至 support@medical-ai.dev