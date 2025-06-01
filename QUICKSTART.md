# 🚀 DCU-in-Action 快速开始指南

## 📋 环境要求

- Python 3.8+
- 海光DCU驱动 5.0+ (可选，可以运行模拟模式)
- Ubuntu 20.04+ 或其他Linux发行版

## ⚡ 快速安装

### 1. 克隆项目
```bash
git clone https://github.com/your-org/dcu-in-action.git
cd dcu-in-action
```

### 2. 创建虚拟环境
```bash
# 使用 conda
conda create -n dcu_env python=3.10
conda activate dcu_env

# 或使用 venv
python -m venv dcu_env
source dcu_env/bin/activate  # Linux/Mac
# dcu_env\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
# 安装核心依赖
pip install -r requirements.txt

# 或安装完整依赖
pip install -r common/docker/requirements.txt
```

## 🎯 快速验证

### 运行基础测试
```bash
# 测试 DCU 管理器
python examples/basic/test_dcu_manager.py
```

如果看到类似以下输出，说明安装成功：
```
==================================================
DCU Manager Basic Test
==================================================
DCU Available: True
Device Count: 2
...
==================================================
Test completed successfully!
==================================================
```

### 检查生成的配置文件
```bash
cat test_config.yaml
```

## 🛠️ 核心功能

### 1. DCU 设备管理
```python
from common.dcu import DCUManager

# 初始化设备管理器
dcu = DCUManager()

# 检查设备可用性
print(f"DCU Available: {dcu.is_available()}")
print(f"Device Count: {dcu.get_device_count()}")

# 获取设备信息
devices = dcu.get_all_devices_info()
for device in devices:
    print(f"Device: {device.name}")
    print(f"Memory: {device.memory_total} MB")
```

### 2. 配置管理
```python
from common.utils import ConfigManager

# 创建配置管理器
config = ConfigManager()

# 设置配置
config.set('dcu.device_id', 0)
config.set('training.batch_size', 32)

# 获取配置
device_id = config.get('dcu.device_id')
batch_size = config.get('training.batch_size')

# 保存配置
config.save_config('my_config.yaml')
```

### 3. 性能监控
```python
from common.dcu import DCUManager

dcu = DCUManager()

# 开始监控
dcu.start_monitoring(interval=1.0)

# 获取性能摘要
summary = dcu.get_performance_summary()
print(summary)

# 停止监控
dcu.stop_monitoring()
```

## 📁 项目结构

```
dcu-in-action/
├── common/                  # 核心工具库
│   ├── dcu/                # DCU设备管理
│   ├── utils/              # 通用工具
│   └── llm/                # 大模型工具
├── examples/               # 实战示例
│   ├── basic/              # 基础示例
│   ├── training/           # 训练示例
│   └── inference/          # 推理示例
├── docs/                   # 文档
└── configs/                # 配置文件
```

## 🚨 故障排除

### 1. PyTorch 不可用
如果看到 "PyTorch 不可用，启用模拟模式" 的警告，这是正常的。在没有 DCU 环境的机器上，项目会自动启用模拟模式进行测试。

### 2. 导入错误
如果遇到导入错误，请确保：
- Python 路径正确
- 依赖包已安装
- 在项目根目录运行脚本

### 3. 权限问题
如果遇到权限问题，确保对项目目录有读写权限：
```bash
chmod -R 755 dcu-in-action/
```

## 📖 下一步

- 查看 [完整文档](README.md)
- 浏览 [示例目录](examples/)
- 了解 [大模型训练](docs/tutorials/02-model-training.md)
- 学习 [模型微调](docs/tutorials/03-model-finetuning.md)

## 🤝 获取帮助

- 查看 [问题解答](docs/faq.md)
- 提交 [Issue](https://github.com/your-org/dcu-in-action/issues)
- 参与 [讨论](https://github.com/your-org/dcu-in-action/discussions)

---

**⭐ 如果这个项目对您有帮助，请给个Star支持！⭐** 