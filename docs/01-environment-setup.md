# 🚀 DCU环境搭建指南

本指南将帮助您从零开始搭建完整的海光DCU开发环境，包括硬件检查、驱动安装、开发环境配置等。

## 📋 环境要求

### 硬件要求

| 组件 | 最低要求 | 推荐配置 | 说明 |
|------|----------|----------|------|
| **DCU设备** | Z100/K100 | Z100L/K100-AI | 海光DCU加速卡 |
| **CPU** | 8核心 | 16核心+ | x86_64架构 |
| **内存** | 32GB | 128GB+ | DDR4/DDR5 |
| **存储** | 500GB SSD | 2TB+ NVMe | 高速存储 |
| **网络** | 千兆网卡 | 万兆网卡 | 分布式训练需要 |

### 软件要求

| 软件 | 版本要求 | 推荐版本 | 用途 |
|------|----------|----------|------|
| **操作系统** | Ubuntu 20.04+ | Ubuntu 22.04 LTS | 主操作系统 |
| **Python** | 3.8+ | 3.10+ | 开发语言 |
| **DCU Runtime** | 5.0+ | 最新版本 | DCU驱动 |
| **Docker** | 20.10+ | 最新版本 | 容器化 |

## 🔧 安装步骤

### Step 1: 系统准备

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y \
    curl \
    wget \
    git \
    vim \
    htop \
    build-essential \
    cmake \
    pkg-config

# 检查系统信息
uname -a
lscpu
free -h
df -h
```

### Step 2: DCU硬件检查

```bash
# 检查DCU设备
lspci | grep -i display

# 检查DCU设备节点
ls -la /dev/kfd /dev/dri/

# 检查DCU驱动状态
dmesg | grep -i dcu
```

### Step 3: DCU驱动安装

#### 方法1: 使用官方安装包

```bash
# 下载DCU驱动包（请从官方获取最新版本）
wget https://developer.sourcefind.cn/downloads/dcu-runtime-5.0.tar.gz

# 解压安装包
tar -xzf dcu-runtime-5.0.tar.gz
cd dcu-runtime-5.0

# 安装驱动
sudo ./install.sh

# 重启系统
sudo reboot
```

#### 方法2: 使用Docker镜像（推荐）

```bash
# 拉取官方DCU镜像
docker pull image.sourcefind.cn:5000/dcu/admin/base/pytorch:2.4.1-ubuntu22.04-dtk25.04-py3.10

# 运行容器测试
docker run --rm --device=/dev/kfd --device=/dev/dri \
    image.sourcefind.cn:5000/dcu/admin/base/pytorch:2.4.1-ubuntu22.04-dtk25.04-py3.10\
    python -c "import torch; print(torch.cuda.is_available())"
```

### Step 4: Python环境配置

```bash
# 安装Python 3.10
sudo apt install -y python3.10 python3.10-venv python3.10-dev

# 创建虚拟环境
python3.10 -m venv dcu-env
source dcu-env/bin/activate

# 升级pip
pip install --upgrade pip setuptools wheel
```

### Step 5: 项目安装

```bash
# 克隆项目
git clone https://github.com/your-org/dcu-in-action.git
cd dcu-in-action

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .

# 配置环境变量
echo 'export DCU_VISIBLE_DEVICES=0,1,2,3' >> ~/.bashrc
echo 'export PYTHONPATH=$PWD:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

## ✅ 环境验证

### 基础验证

```bash
# 运行环境检查脚本
./common/setup/check_environment.sh

# 验证DCU设备
python -c "
from common.dcu import device_manager
print('DCU设备信息:')
print(device_manager.get_device_info())
"

# 验证PyTorch DCU支持
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'DCU可用: {torch.cuda.is_available()}')
print(f'DCU设备数量: {torch.cuda.device_count()}')
"
```

### 性能测试

```bash
# 运行基础性能测试
cd examples/benchmarks
python dcu_benchmark.py

# 运行矩阵计算测试
python matrix_benchmark.py --size 1024

# 运行内存测试
python memory_benchmark.py
```

## 🐳 Docker环境（推荐）

### 构建开发镜像

```bash
# 构建镜像
docker build -t dcu-in-action:dev .

# 运行开发容器
docker run -it --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    -v $(pwd):/workspace \
    -p 8888:8888 \
    -p 6006:6006 \
    dcu-in-action:dev
```

### 使用Docker Compose

```bash
# 启动完整开发环境
docker-compose up -d

# 进入开发容器
docker-compose exec dcu-dev bash

# 查看服务状态
docker-compose ps
```

## 🔧 常见问题

### Q1: DCU设备未检测到

**问题**: `torch.cuda.is_available()` 返回 `False`

**解决方案**:
```bash
# 检查设备权限
ls -la /dev/kfd /dev/dri/

# 添加用户到render组
sudo usermod -a -G render $USER

# 重新登录或重启
```

### Q2: 内存不足错误

**问题**: 训练时出现 `CUDA out of memory`

**解决方案**:
```bash
# 检查DCU内存使用
python -c "
import torch
print(f'DCU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'已用内存: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB')
"

# 使用内存优化工具
python common/dcu/memory_optimizer.py --optimize
```

### Q3: 性能不佳

**问题**: 训练速度慢于预期

**解决方案**:
```bash
# 运行性能分析
python common/dcu/performance_profiler.py --profile

# 检查DCU利用率
watch -n 1 'python -c "
import torch
for i in range(torch.cuda.device_count()):
    print(f\"DCU {i}: {torch.cuda.utilization(i)}%\")
"'
```

## 📊 性能基准

### 硬件性能

| 测试项目 | Z100 | K100 | K100-AI |
|----------|------|------|---------|
| **计算性能** | 32 TFLOPS | 45 TFLOPS | 60 TFLOPS |
| **内存带宽** | 1.6 TB/s | 2.0 TB/s | 2.4 TB/s |
| **显存容量** | 32GB | 32GB | 64GB |

### 软件性能

| 框架 | 版本 | 性能提升 | 兼容性 |
|------|------|----------|--------|
| **PyTorch** | 2.0+ | 3-5x | ✅ 完全兼容 |
| **Transformers** | 4.30+ | 2-4x | ✅ 完全兼容 |
| **vLLM** | 0.2.0+ | 5-10x | ✅ 完全兼容 |

## 🚀 下一步

环境搭建完成后，您可以：

1. **学习基础教程**: [第一个DCU程序](02-first-dcu-program.md)
2. **运行训练示例**: [大模型训练实战](02-model-training.md)
3. **尝试模型微调**: [高效模型微调](03-model-finetuning.md)
4. **部署推理服务**: [推理服务部署](04-model-inference.md)

## 📞 获取帮助

如果在环境搭建过程中遇到问题：

- 📚 查看 [FAQ文档](../faq.md)
- 🐛 提交 [GitHub Issue](https://github.com/your-org/dcu-in-action/issues)
- 💬 加入 [技术交流群](https://discord.gg/dcu-in-action)
- 📧 发送邮件到 [support@dcu-in-action.org](mailto:support@dcu-in-action.org) 