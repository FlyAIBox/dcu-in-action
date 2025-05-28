# 🚀 海光DCU加速卡实战 - 快速开始指南

欢迎使用海光DCU加速卡实战项目！本指南将帮助您快速上手，从环境配置到运行第一个大模型应用。

## 📋 前置要求

### 硬件要求
- **DCU设备**: 海光K100、K100-AI、Z100L等
- **内存**: 推荐32GB以上系统内存
- **存储**: 至少100GB可用磁盘空间
- **网络**: 稳定的互联网连接（用于下载模型）

### 软件要求
- **操作系统**: Ubuntu 20.04+ 或 CentOS 7.9+
- **DCU驱动**: DTK 25.04+
- **Docker**: 20.10+ (可选，推荐)
- **Python**: 3.8+ (如果不使用Docker)

## 🛠️ 安装方式

我们提供两种安装方式，推荐使用Docker方式以获得最佳体验。

### 方式1: Docker安装 (推荐)

#### 1.1 克隆项目
```bash
git clone https://github.com/hygon-technologies/dcu-in-action.git
cd dcu-in-action
```

#### 1.2 初始化环境
```bash
# 使用Makefile快速设置
make setup

# 或手动创建目录
mkdir -p models datasets outputs logs
chmod +x scripts/setup/*.sh docker-entrypoint.sh
```

#### 1.3 构建Docker镜像
```bash
# 快速构建
make build

# 或使用docker命令
docker build -t dcu-in-action:latest .
```

#### 1.4 启动开发环境
```bash
# 启动主容器
make run

# 或启动所有服务
make run-all
```

#### 1.5 进入容器
```bash
# 进入开发容器
make shell

# 或直接使用docker命令
docker exec -it dcu-dev-main bash
```

### 方式2: 本地安装

#### 2.1 克隆项目
```bash
git clone https://github.com/hygon-technologies/dcu-in-action.git
cd dcu-in-action
```

#### 2.2 检查DCU环境
```bash
# 运行环境检查脚本
bash scripts/setup/check_environment.sh
```

#### 2.3 安装依赖
```bash
# 自动安装脚本
bash scripts/setup/install_dependencies.sh

# 或使用Makefile
make install

# 或手动安装
pip install -r requirements.txt
```

## 🏃‍♂️ 快速验证

### 1. 基础环境测试
```bash
# 测试DCU环境
python examples/llm-inference/simple_test.py

# 或使用Makefile
make test-dcu
```

期望输出：
```
🔄 开始DCU环境测试...
✅ DCU设备检测成功
✅ PyTorch DCU支持正常
✅ 基础计算测试通过
✅ 混合精度测试通过
📊 测试报告已保存
```

### 2. DCU性能监控
```bash
# 启动性能监控
python scripts/utils/monitor_performance.py monitor

# 或使用Makefile
make monitor-local
```

### 3. 模型推理测试
```bash
# ChatGLM推理测试
python examples/llm-inference/chatglm_inference.py --mode chat

# 或基准测试
python examples/llm-inference/chatglm_inference.py --mode benchmark
```

## 🎯 核心功能演示

### 1. 大模型推理

#### 1.1 简单推理测试
```bash
# 单次推理
python examples/llm-inference/chatglm_inference.py \
    --mode test \
    --prompt "介绍一下海光DCU的技术优势"
```

#### 1.2 交互式对话
```bash
# 启动聊天模式
python examples/llm-inference/chatglm_inference.py --mode chat
```

#### 1.3 高性能推理服务
```bash
# 启动vLLM推理服务
python examples/llm-inference/vllm_server.py \
    --mode server \
    --model "Qwen/Qwen-7B-Chat" \
    --host 0.0.0.0 \
    --port 8000
```

然后访问 http://localhost:8000/docs 查看API文档

### 2. 模型训练

#### 2.1 创建训练数据
```bash
# 创建示例数据
python examples/llm-fine-tuning/lora_finetune.py --create_sample_data
```

#### 2.2 LoRA微调
```bash
# 启动LoRA微调
python examples/llm-fine-tuning/lora_finetune.py \
    --dataset_path ./data/sample_data.json \
    --output_dir ./outputs/lora-finetuned \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4

# 或使用Makefile
make train-lora
```

#### 2.3 完整模型训练
```bash
# LLaMA模型训练
python examples/llm-training/train_llama.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --dataset_name "wikitext" \
    --output_dir ./outputs/llama-trained \
    --num_train_epochs 1

# 或使用Makefile
make train-llama
```

### 3. 科学计算

#### 3.1 矩阵计算性能测试
```python
# 启动Python环境
python

# 执行矩阵运算测试
import torch
import time

# 大规模矩阵乘法
size = 4096
a = torch.randn(size, size, device='cuda')
b = torch.randn(size, size, device='cuda')

start_time = time.time()
c = torch.mm(a, b)
torch.cuda.synchronize()
end_time = time.time()

print(f"矩阵乘法性能: {(2 * size**3) / (end_time - start_time) / 1e12:.2f} TFLOPS")
```

#### 3.2 科学计算应用
```bash
# 查看科学计算示例
python examples/llm-for-science/molecular_dynamics.py
python examples/llm-for-science/climate_simulation.py
```

## 🖥️ 服务访问

项目启动后，可以通过以下方式访问各种服务：

| 服务 | 访问地址 | 描述 |
|------|----------|------|
| Jupyter Lab | http://localhost:8888 | 交互式开发环境 |
| 推理API | http://localhost:8000 | FastAPI推理服务 |
| Gradio界面 | http://localhost:7860 | Web界面演示 |
| TensorBoard | http://localhost:6006 | 训练可视化 |
| 监控面板 | http://localhost:9090 | 性能监控 |

### 启动Jupyter Lab
```bash
# Docker方式
make jupyter

# 本地方式
make jupyter-local
```

### 启动推理服务
```bash
# Docker方式
make inference

# 本地方式
make inference-local
```

## 🔧 常用命令

### Docker相关
```bash
# 查看帮助
make help

# 构建镜像
make build

# 启动服务
make run
make run-all

# 停止服务
make stop

# 重启服务
make restart

# 进入容器
make shell

# 查看日志
make logs
```

### 开发相关
```bash
# 环境检查
make check

# 安装依赖
make install

# 代码格式化
make format

# 运行测试
make test

# 性能基准
make benchmark
```

### 应用相关
```bash
# 启动Jupyter
make jupyter

# 启动推理
make inference

# 启动训练
make train

# 启动监控
make monitor
```

## 📊 性能调优

### 1. 内存优化
```python
# 启用混合精度
model = model.half()  # FP16

# 梯度检查点
model.gradient_checkpointing_enable()

# 清理显存
torch.cuda.empty_cache()
```

### 2. 批处理优化
```python
# 动态批处理
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 梯度累积
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. 分布式训练
```bash
# 多DCU训练
torchrun --nproc_per_node=4 examples/llm-training/train_llama.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --output_dir ./outputs/distributed-training
```

## 🐛 故障排查

### 常见问题及解决方案

#### 1. DCU设备不可用
```bash
# 检查驱动
hy-smi

# 检查权限
ls -la /dev/dri/

# 重新加载模块
sudo modprobe amdgpu
```

#### 2. 内存不足错误
```bash
# 减少批次大小
--per_device_train_batch_size 2

# 启用梯度检查点
--gradient_checkpointing True

# 使用量化
--use_4bit True
```

#### 3. 模型下载失败
```bash
# 设置镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 手动下载
huggingface-cli download model_name

# 使用本地模型
--model_name_or_path /path/to/local/model
```

#### 4. Docker相关问题
```bash
# 检查设备映射
docker run --device /dev/dri:/dev/dri -it dcu-in-action:latest

# 查看容器日志
make logs

# 重建镜像
make clean-all
make build
```

## 📚 学习资源

### 文档资源
- [项目文档](docs/): 详细的技术文档
- [API文档](http://localhost:8000/docs): 推理服务API
- [官方文档](https://developer.sourcefind.cn/): 海光DCU开发文档

### 示例代码
- `examples/llm-inference/`: 推理示例
- `examples/llm-training/`: 训练示例  
- `examples/llm-fine-tuning/`: 微调示例
- `examples/llm-for-science/`: 科学计算示例

### 社区支持
- [GitHub Issues](https://github.com/hygon-technologies/dcu-in-action/issues)
- [开发者社区](https://developer.sourcefind.cn/)
- [技术论坛](https://bbs.sourcefind.cn/)

## 🚀 下一步

完成快速开始后，建议您：

1. **深入学习**: 阅读[详细文档](docs/)了解更多高级功能
2. **实践项目**: 使用自己的数据尝试训练和推理
3. **性能优化**: 根据应用需求调优模型和系统参数
4. **社区参与**: 分享经验，参与开源社区建设

## ❓ 获取帮助

如果遇到问题：

1. 首先查看[故障排查](#故障排查)部分
2. 搜索[GitHub Issues](https://github.com/hygon-technologies/dcu-in-action/issues)
3. 查阅[官方文档](https://developer.sourcefind.cn/)
4. 在社区论坛提问

---

**祝您使用愉快！🎉**

如果本项目对您有帮助，请给我们一个⭐星标支持！ 