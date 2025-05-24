# DCU 加速卡实战指南

> 声明：本项目内容整理自海光DCU开发社区(https://developer.sourcefind.cn/)和网络公开资料，仅用于学习和研究目的。

## 项目简介

本项目是一个关于海光 DCU (Data Compute Unit) 加速卡在人工智能和高性能计算领域的实战指南。主要涵盖大模型训练、模型微调、推理加速以及 HPC 科学计算等应用场景。

## 主要特性

- 大模型训练：使用 DCU 加速深度学习模型训练
- 模型微调：针对特定任务的模型优化
- 推理加速：模型部署与推理性能优化
- HPC 计算：科学计算与高性能计算应用

## 环境要求

- 硬件要求：
  - 海光 DCU 加速卡
  - 支持的 CPU 平台
  
- 软件要求：
  - 操作系统：UOS/麒麟/CentOS/Ubuntu
  - DHCC (DCU Heterogeneous Computing Compiler)
  - DPCPP (Data Parallel C++)
  - Python 3.10

## 快速开始

1. 环境配置
```bash
# 安装依赖
./scripts/setup/install_dependencies.sh

# 配置 DCU 环境
source /opt/dtk/env.sh
```

2. 验证安装
```bash
# 运行测试用例
python tests/verify_installation.py
```

## 目录结构

- `docs/`: 详细文档
  - `installation.md`: 安装指南
  - `training.md`: 模型训练教程
  - `fine-tuning.md`: 模型微调指南
  - `inference.md`: 推理部署教程
  - `hpc.md`: HPC 应用指南

- `examples/`: 示例代码
  - `training/`: 模型训练示例
  - `fine-tuning/`: 模型微调示例
  - `inference/`: 推理示例
  - `hpc/`: HPC 计算示例

## 应用场景

### 1. 大模型训练

- LLM (Large Language Model) 训练
- 计算机视觉模型训练
- 分布式训练支持

### 2. 模型微调

- PEFT (Parameter Efficient Fine-Tuning)
- LoRA 适配
- 领域模型优化

### 3. 推理加速

- 模型量化
- 推理性能优化
- 批处理推理

### 4. HPC 应用

- 矩阵计算
- 科学计算
- 并行计算

## 性能优化

- DCU 算子优化
- 内存管理
- 并行计算策略

## 常见问题

请参考 [docs/FAQ.md](docs/FAQ.md) 了解常见问题及解决方案。

## 参考资源

- [海光开发者社区](https://developer.sourcefind.cn/)
- [DCU 编程指南]()
- [DHCC 文档]()

## 贡献指南

欢迎提交 Issue 和 Pull Request 来完善本项目。

## 许可证

本项目采用 Apache 2.0 许可证。

## 免责声明

本项目仅用于学习和研究目的，所有内容均来自海光 DCU 开发社区和网络公开资料。使用本项目时请遵守相关法律法规。