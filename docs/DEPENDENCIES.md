# 📦 DCU-in-Action 依赖管理指南

## 概述

本项目提供了多层次的依赖管理方案，根据不同的使用场景和环境需求，提供最优化的依赖配置。

## 📋 依赖文件说明

### 1. requirements.txt (标准依赖)
- **用途**: 生产环境推荐的标准依赖配置
- **特点**: 包含核心功能，体积适中，安装快速
- **适用场景**: 
  - 生产环境部署
  - 大模型训练和推理
  - 一般开发使用
- **包含功能**:
  - PyTorch生态系统 (torch, transformers, tokenizers)
  - DCU专用加速库 (lmslim, flash-attn, vllm, deepspeed)
  - Web服务框架 (fastapi, uvicorn, gradio)
  - 科学计算基础 (numpy, scipy, pandas)
  - 配置管理和监控工具

### 2. requirements-full.txt (完整依赖)
- **用途**: 包含所有功能模块的完整依赖
- **特点**: 功能齐全，支持所有特性
- **适用场景**:
  - 开发环境
  - 研究和实验
  - 需要完整功能的场景
- **额外包含**:
  - 多媒体处理 (pillow, librosa, opencv)
  - 数据库支持 (redis, sqlalchemy, pymongo)
  - 云服务集成 (boto3, aws-cli)
  - 开发调试工具 (jupyter, ipython)
  - 性能分析工具
  - 分布式计算框架 (ray)

### 3. common/docker/requirements.txt (容器环境)
- **用途**: Docker容器化部署优化配置
- **特点**: 精简但功能完整，适合容器环境
- **适用场景**:
  - Docker部署
  - Kubernetes集群
  - 云原生环境
- **优化特性**:
  - 移除开发专用工具
  - 保留生产必需功能
  - 优化镜像大小

## 🛠️ 安装方式

### 方式1: 智能安装脚本（推荐）

```bash
# 检查环境
./scripts/install_requirements.sh --check

# 标准安装（推荐）
./scripts/install_requirements.sh --mode standard

# 完整安装
./scripts/install_requirements.sh --mode full

# 最小安装
./scripts/install_requirements.sh --mode minimal

# Docker环境安装
./scripts/install_requirements.sh --mode docker
```

#### 安装脚本功能
- ✅ 自动环境检查 (Python版本、系统资源、DCU设备)
- ✅ 智能依赖管理 (根据模式选择合适的依赖)
- ✅ DCU特定包检测和安装
- ✅ 安装后验证
- ✅ 详细的日志输出和错误处理

### 方式2: 手动安装

```bash
# 标准依赖
pip install -r requirements.txt

# 完整依赖
pip install -r requirements-full.txt

# Docker依赖
pip install -r common/docker/requirements.txt
```

## 🔧 DCU特定依赖

### DCU优化包列表
以下包需要从海光官方下载页面获取：

| 包名 | 版本 | 说明 |
|------|------|------|
| torch | 2.4.1+das.opt2.dtk2504 | DCU优化的PyTorch |
| lmslim | 0.2.1+das.dtk2504 | 大模型推理优化 |
| flash-attn | 2.6.1+das.opt4.dtk2504 | 高效注意力机制 |
| vllm | 0.6.2+das.opt3.dtk2504 | 高性能推理引擎 |
| deepspeed | 0.14.2+das.opt2.dtk2504 | 分布式训练框架 |
| triton | 3.0.0 | 高性能计算内核 |

### 安装DCU包

```bash
# 下载官方wheel文件后
pip install torch-2.4.1+das.opt2.dtk2504-cp310-cp310-manylinux_2_28_x86_64.whl
pip install lmslim-0.2.1+das.dtk2504-cp310-cp310-manylinux_2_28_x86_64.whl
# ... 其他包
```

### 自动检测安装
智能安装脚本会自动检测当前目录中的DCU wheel文件并安装：

```bash
# 将wheel文件放到项目根目录
ls *.whl
# 运行安装脚本，会自动检测并安装
./scripts/install_requirements.sh --mode standard
```

## 📊 版本管理

### 版本选择原则
1. **基于实际环境**: 所有版本号基于生产环境pip list验证
2. **兼容性优先**: 确保包之间的版本兼容性
3. **性能优化**: 选择性能最优的版本组合
4. **稳定性保证**: 选择经过测试验证的稳定版本

### 版本更新策略
- 定期检查依赖包更新
- 在测试环境验证新版本兼容性
- 逐步升级，确保向后兼容
- 记录版本变更和测试结果

## 🚨 常见问题

### Q1: DCU环境检测失败
**问题**: 未检测到DCU设备文件
**解决方案**:
```bash
# 检查设备文件
ls -la /dev/kfd /dev/dri

# 检查驱动状态
lsmod | grep amdgpu

# 重新安装驱动或在模拟模式下运行
```

### Q2: 依赖包冲突
**问题**: 包版本冲突或安装失败
**解决方案**:
```bash
# 清理pip缓存
pip cache purge

# 强制重新安装
pip install --upgrade --force-reinstall torch transformers

# 使用虚拟环境
python -m venv new_env
source new_env/bin/activate
```

### Q3: 网络安装问题
**问题**: 下载速度慢或连接失败
**解决方案**:
```bash
# 使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 配置永久镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q4: 内存不足
**问题**: 安装过程中内存不足
**解决方案**:
```bash
# 增加swap空间
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 或选择最小安装模式
./scripts/install_requirements.sh --mode minimal
```

## 📈 性能优化

### 安装优化
- 使用本地缓存减少重复下载
- 并行安装兼容的包
- 预下载DCU特定包到本地

### 运行时优化
- 合理设置环境变量
- 配置DCU设备可见性
- 优化内存和显存使用

## 🔄 升级指南

### 从旧版本升级
```bash
# 备份当前环境
pip freeze > current_requirements.txt

# 卸载旧版本
pip uninstall -r current_requirements.txt -y

# 安装新版本
./scripts/install_requirements.sh --mode standard

# 验证升级结果
./scripts/install_requirements.sh --check
```

### 增量更新
```bash
# 更新特定包
pip install --upgrade transformers torch

# 更新所有包到兼容版本
pip install --upgrade -r requirements.txt
```

## 🌟 最佳实践

1. **环境隔离**: 始终使用虚拟环境
2. **版本锁定**: 生产环境使用精确版本号
3. **分层安装**: 根据需求选择合适的依赖文件
4. **定期检查**: 定期验证环境状态
5. **文档更新**: 记录环境变更和配置

## 📞 技术支持

- 📖 查看详细文档: [README.md](../README.md)
- 🐛 报告问题: [GitHub Issues](https://github.com/your-org/dcu-in-action/issues)
- 💬 技术讨论: [GitHub Discussions](https://github.com/your-org/dcu-in-action/discussions)
- 🌐 官方资源: [海光DCU开发者社区](https://developer.sourcefind.cn/)

---

**💡 提示**: 建议首次使用时先运行环境检查，确保系统满足基本要求后再进行依赖安装。 