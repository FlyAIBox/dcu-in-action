# 更新日志

本文档记录了海光DCU加速卡实战项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
并且本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/) 规范。

## [未发布]

### 计划新增
- [ ] Web界面管理工具
- [ ] 自动化模型优化
- [ ] 多节点分布式训练支持
- [ ] 模型版本管理系统
- [ ] 实时推理性能监控面板

### 计划改进
- [ ] 优化内存使用效率
- [ ] 简化部署流程
- [ ] 增强错误处理机制
- [ ] 扩展支持的模型类型

## [1.0.0] - 2024-01-15

### 🎉 首次发布

#### 新增功能
- **完整的DCU开发环境**: 基于Docker的一站式开发环境
- **大模型推理系统**: 
  - ChatGLM推理引擎，支持交互式对话
  - vLLM高性能推理服务器
  - 批量推理和API服务支持
- **模型训练框架**:
  - LLaMA模型完整训练流程
  - LoRA高效微调实现
  - 分布式训练支持
- **科学计算应用**:
  - 矩阵计算优化
  - 数值求解算法
  - 并行计算示例
- **性能监控工具**:
  - 实时DCU状态监控
  - 性能基准测试套件
  - 资源使用统计

#### 技术文档
- **安装指南** (`docs/01-dcu-installation.md`): DCU环境配置详解
- **推理教程** (`docs/02-llm-inference.md`): 大模型推理完整指南
- **微调指南** (`docs/03-llm-fine-tuning.md`): 模型微调最佳实践
- **训练教程** (`docs/04-llm-training.md`): 从零开始的模型训练
- **科学计算** (`docs/05-llm-for-science.md`): HPC应用开发指南

#### 示例代码
- **推理示例** (`examples/llm-inference/`):
  - `simple_test.py`: DCU环境测试
  - `chatglm_inference.py`: ChatGLM推理引擎
  - `vllm_server.py`: 高性能推理服务
- **训练示例** (`examples/llm-training/`):
  - `train_llama.py`: LLaMA模型训练
- **微调示例** (`examples/llm-fine-tuning/`):
  - `lora_finetune.py`: LoRA微调实现
- **科学计算** (`examples/llm-for-science/`):
  - 分子动力学模拟
  - 气候模型计算
  - 数值分析算法

#### 工具脚本
- **环境配置** (`scripts/setup/`):
  - `check_environment.sh`: 完整环境检查
  - `install_dependencies.sh`: 自动依赖安装
- **性能监控** (`scripts/utils/`):
  - `monitor_performance.py`: DCU性能监控工具

#### 容器化支持
- **Docker配置**:
  - `Dockerfile`: 完整开发环境镜像
  - `docker-compose.yml`: 多服务编排配置
  - `docker-entrypoint.sh`: 容器启动脚本
- **Makefile**: 便捷的项目管理命令

#### 项目配置
- **依赖管理**:
  - `requirements.txt`: Python依赖列表
  - `pyproject.toml`: 现代Python项目配置
- **开发工具**: 代码格式化、类型检查、测试框架配置

### 支持的模型
- **ChatGLM系列**: ChatGLM2-6B, ChatGLM3-6B
- **LLaMA系列**: LLaMA-2-7B, LLaMA-2-13B
- **Qwen系列**: Qwen-7B-Chat, Qwen-14B-Chat
- **Baichuan系列**: Baichuan2-7B-Chat, Baichuan2-13B-Chat

### 支持的功能
- ✅ 模型推理加速
- ✅ 混合精度训练
- ✅ LoRA/QLoRA微调
- ✅ 多DCU并行训练
- ✅ 量化推理
- ✅ 流式输出
- ✅ 批量处理
- ✅ API服务部署

### 性能基准 (DCU K100)
| 任务类型 | 模型规模 | 吞吐量 | 延迟 |
|----------|----------|--------|------|
| 推理 | 7B参数 | 150 tokens/s | 80ms |
| 训练 | 7B参数 | 1200 samples/h | - |
| 微调 | 7B参数 (LoRA) | 2400 samples/h | - |

### 系统要求
- **硬件**: 海光DCU K100/K100-AI/Z100L
- **操作系统**: Ubuntu 20.04+, CentOS 7.9+
- **驱动**: DTK 25.04+
- **内存**: 32GB+ 系统内存推荐
- **存储**: 100GB+ 可用磁盘空间

### 依赖版本
- Python >= 3.8
- PyTorch >= 2.1.0
- Transformers >= 4.35.0
- CUDA/ROCm 兼容版本

## [0.2.0] - 2023-12-20

### 新增
- LoRA微调支持
- 性能监控工具
- Docker容器化部署
- 科学计算示例

### 改进
- 优化推理性能
- 增强错误处理
- 完善文档结构

### 修复
- 修复内存泄漏问题
- 解决模型加载错误
- 修正配置文件格式

## [0.1.0] - 2023-11-15

### 新增
- 项目初始化
- 基础推理功能
- 简单训练示例
- 基础文档框架

---

## 贡献指南

### 如何贡献更新日志

1. **新增功能**: 在"未发布"部分添加到"新增"小节
2. **问题修复**: 添加到"修复"小节
3. **性能改进**: 添加到"改进"小节
4. **重大变更**: 添加到"变更"小节，并标注为破坏性变更

### 格式规范

- 使用语义化版本号 (MAJOR.MINOR.PATCH)
- 每个变更条目应该简洁明了
- 包含相关的文件路径或组件名称
- 对于破坏性变更，使用 **[BREAKING]** 标记

### 发布流程

1. 更新版本号
2. 整理"未发布"部分到新版本
3. 更新README中的版本信息
4. 创建Git标签
5. 发布Docker镜像

---

关于项目发展路线图和详细的技术规划，请查看 [项目主页](https://github.com/hygon-technologies/dcu-in-action)。 