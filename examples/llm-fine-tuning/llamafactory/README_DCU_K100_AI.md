# 🚀 LLaMA Factory × 海光DCU k100-AI: 大模型微调完整指南

## 📋 概述

本指南专门针对**海光DCU k100-AI加速卡**优化，提供基于LLaMA Factory框架的大模型微调完整解决方案。通过本教程，您将学会如何在DCU k100-AI上高效训练和微调大语言模型。

### 🎯 核心特性

- ✅ **DCU专用优化**：针对k100-AI的性能调优配置
- ✅ **完整工具链**：从数据准备到模型部署的全流程
- ✅ **生产就绪**：企业级的稳定性和可扩展性
- ✅ **性能卓越**：充分发挥64GB HBM2E显存优势
- ✅ **易于使用**：零门槛的Web UI界面操作

## 🔧 硬件环境要求

### 📱 DCU配置
- **型号**: 海光DCU k100-AI
- **显存**: 64GB HBM2E
- **计算精度**: 支持FP32、FP16、BF16
- **显存带宽**: ~1.2TB/s

### 💻 系统环境
- **操作系统**: Ubuntu 22.04.4 LTS
- **内核版本**: 5.15.0-94-generic
- **DCU驱动**: ROCK 6.3.8+
- **DTK版本**: 25.04+
- **系统内存**: ≥64GB DDR4
- **存储空间**: ≥200GB SSD

## 🚀 快速开始

### 第一步：环境检查

```bash
# 检查DCU驱动
dcu-smi -L

# 检查DTK版本
cat /opt/dtk/VERSION

# 检查系统信息
uname -a
free -h
```

### 第二步：一键安装

```bash
# 运行自动配置脚本
./scripts/dcu_k100_ai_setup.sh

# 手动激活环境
source ~/.dcurc
conda activate llamafactory-dcu
```

### 第三步：启动Web UI

```bash
# 启动LLaMA Factory Web界面
~/dcu_configs/start_webui.sh
```

访问 http://localhost:7860 开始使用！

## 📊 DCU k100-AI 优化配置

### 🎯 推荐模型配置

| 模型规模 | 批处理大小 | 梯度累积 | LoRA Rank | 学习率 | 预计训练时间 |
|----------|------------|----------|-----------|--------|--------------|
| **Qwen2.5-3B** | 16 | 2 | 32 | 2e-4 | 15-25分钟 |
| **Qwen2.5-7B** | 8 | 4 | 64 | 1e-4 | 35-50分钟 |
| **Qwen2.5-14B** | 4 | 8 | 128 | 5e-5 | 70-90分钟 |
| **Qwen2.5-33B** | 2 | 16 | 256 | 2e-5 | 150-200分钟 |

### ⚡ 性能优化参数

```json
{
    "精度设置": {
        "bf16": true,
        "gradient_checkpointing": true,
        "dataloader_pin_memory": true
    },
    "内存优化": {
        "max_length": 2048,
        "cutoff_len": 2048,
        "preprocessing_num_workers": 8
    },
    "DCU特定": {
        "dataloader_num_workers": 4,
        "ddp_timeout": 3600
    }
}
```

## 🛠️ 使用教程

### 📈 数据准备

1. **使用Easy Dataset生成数据**
   ```bash
   # 参考完整教程文档
   cat doc/LLaMA\ Factory：03-Easy\ Dataset\ 让大模型高效学习领域知识.md
   ```

2. **数据格式要求**
   ```json
   [
       {
           "instruction": "任务描述",
           "input": "用户输入",
           "output": "期望输出"
       }
   ]
   ```

### 🎯 模型微调

1. **Web UI操作**
   - 模型选择：Qwen2.5-3B-Instruct
   - 数据集：上传您的数据集
   - 参数配置：使用推荐的DCU优化参数

2. **命令行训练**
   ```bash
   llamafactory-cli train ~/dcu_configs/qwen2.5_3b_dcu.json
   ```

### 📊 性能监控

```