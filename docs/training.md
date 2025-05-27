# DCU 大模型训练指南

> 本文档整理自海光DCU开发社区和网络公开资料

## 概述

本指南介绍如何使用海光 DCU 加速卡进行大模型训练，包括环境准备、框架选择、训练策略和性能优化等内容。

## 支持的训练框架

### PyTorch
- **版本支持**：PyTorch 1.10+ (推荐 2.0+)
- **特性**：完整的 DCU 适配，支持分布式训练
- **适用场景**：大语言模型、计算机视觉模型

### TensorFlow
- **版本支持**：TensorFlow 2.8+
- **特性**：支持 Keras 高级 API
- **适用场景**：传统深度学习模型

### PaddlePaddle
- **版本支持**：PaddlePaddle 2.4+
- **特性**：国产框架，DCU 原生支持
- **适用场景**：NLP、CV、推荐系统

### OneFlow
- **版本支持**：OneFlow 0.8+
- **特性**：高性能分布式训练
- **适用场景**：大规模模型训练

## 环境准备

### 1. 基础环境

```bash
# 设置 DCU 可见性
export HIP_VISIBLE_DEVICES=0,1,2,3

# 设置内存分配策略
export HIP_FORCE_DEV_KERNARG=1

# 优化内存使用
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128
```

### 2. 安装训练框架

```bash
# 安装 PyTorch (从光源镜像仓库)
pip install torch-*.whl torchvision-*.whl

# 安装训练依赖
pip install transformers datasets accelerate
pip install deepspeed apex
```

### 3. 验证环境

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"DCU available: {torch.cuda.is_available()}")
print(f"DCU count: {torch.cuda.device_count()}")

# 测试 DCU 计算
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).cuda()
    y = torch.mm(x, x.t())
    print("DCU computation test passed!")
```

## 大语言模型训练

### 1. 预训练

#### 单卡训练示例

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

# 模型和分词器
model_name = "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 移动到 DCU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 数据准备
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=1000,
    fp16=True,  # 使用混合精度
    dataloader_num_workers=4,
)

# 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# 开始训练
trainer.train()
```

#### 多卡分布式训练

```bash
# 使用 torchrun 启动分布式训练
torchrun --nproc_per_node=4 train_llm.py \
    --model_name_or_path facebook/opt-6.7b \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 3 \
    --output_dir ./output \
    --overwrite_output_dir \
    --fp16 \
    --gradient_checkpointing \
    --dataloader_num_workers 4
```

#### DeepSpeed 集成

```python
# deepspeed_config.json
{
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 2,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-4,
            "warmup_num_steps": 1000
        }
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "activation_checkpointing": {
        "partition_activations": false,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": false,
        "number_checkpoints": null,
        "synchronize_checkpoint_boundary": false,
        "profile": false
    }
}
```

```bash
# 使用 DeepSpeed 启动训练
deepspeed --num_gpus=4 train_llm_deepspeed.py \
    --deepspeed deepspeed_config.json \
    --model_name_or_path facebook/opt-13b \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --output_dir ./output \
    --num_train_epochs 3 \
    --fp16
```

### 2. 指令微调

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import torch

# 指令数据格式
instruction_data = [
    {
        "instruction": "请解释什么是人工智能",
        "input": "",
        "output": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统..."
    },
    # 更多指令数据...
]

# 数据预处理
def format_instruction(example):
    if example["input"]:
        prompt = f"### 指令:\n{example['instruction']}\n\n### 输入:\n{example['input']}\n\n### 回答:\n"
    else:
        prompt = f"### 指令:\n{example['instruction']}\n\n### 回答:\n"
    
    return {
        "text": prompt + example["output"] + tokenizer.eos_token
    }

# 创建数据集
dataset = Dataset.from_list(instruction_data)
dataset = dataset.map(format_instruction)

# 分词
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 训练配置
training_args = TrainingArguments(
    output_dir="./instruction_tuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    logging_steps=50,
    save_steps=500,
    fp16=True,
    learning_rate=2e-5,
    weight_decay=0.01,
)

# 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

## 计算机视觉模型训练

### 1. 图像分类

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 数据加载
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

# 模型定义
import torchvision.models as models
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 有 10 个类别

# 移动到 DCU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# 训练循环
def train_epoch(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 99:
            print(f'Batch {batch_idx+1}, Loss: {running_loss/100:.3f}, Acc: {100.*correct/total:.3f}%')
            running_loss = 0.0

# 训练
num_epochs = 200
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    train_epoch(model, trainloader, criterion, optimizer, device)
    scheduler.step()
```

### 2. 目标检测

```python
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
import torchvision.transforms as T

# 数据变换
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# 自定义数据集（示例）
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # 加载图像和标注文件
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, idx):
        # 加载图像
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        
        # 加载标注
        ann_path = os.path.join(self.root, "annotations", self.annotations[idx])
        # 解析标注文件，获取边界框和标签
        boxes = []  # [[x1, y1, x2, y2], ...]
        labels = []  # [1, 2, 3, ...]
        
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target

    def __len__(self):
        return len(self.imgs)

# 模型定义
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 类 + 背景
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# 移动到 DCU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 数据加载
dataset = CustomDataset('path/to/dataset', get_transform(train=True))
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# 优化器
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    lr_scheduler.step()
    print(f'Epoch {epoch+1}, Loss: {losses.item():.4f}')
```

## 性能优化策略

### 1. 内存优化

```python
# 梯度检查点
import torch.utils.checkpoint as checkpoint

class OptimizedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # 使用梯度检查点减少内存使用
        return checkpoint.checkpoint(self.model, x)

# 混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
model = model.cuda()

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### 2. 数据加载优化

```python
# 优化数据加载
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,  # 增加工作进程
    pin_memory=True,  # 固定内存
    persistent_workers=True,  # 持久化工作进程
    prefetch_factor=2,  # 预取因子
)
```

### 3. 模型并行

```python
# 数据并行
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# 分布式数据并行
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    setup(rank, world_size)
    
    model = YourModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # 训练代码...
    
    cleanup()
```

## 监控和调试

### 1. 训练监控

```python
import wandb
from torch.utils.tensorboard import SummaryWriter

# Weights & Biases
wandb.init(project="dcu-training")

# TensorBoard
writer = SummaryWriter('runs/experiment_1')

# 训练循环中记录指标
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # ... 训练代码 ...
        
        # 记录损失
        wandb.log({"loss": loss.item(), "epoch": epoch})
        writer.add_scalar('Loss/Train', loss.item(), epoch * len(train_loader) + batch_idx)
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch * len(train_loader) + batch_idx)

writer.close()
```

### 2. 性能分析

```python
# DCU 性能分析
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, (data, target) in enumerate(train_loader):
        if step >= (1 + 1 + 3) * 2:
            break
        
        # 训练步骤
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        prof.step()
```

## 常见问题和解决方案

### 1. 内存不足

**问题**：训练时出现 OOM (Out of Memory) 错误

**解决方案**：
- 减小批次大小
- 使用梯度累积
- 启用梯度检查点
- 使用混合精度训练

```python
# 梯度累积示例
accumulation_steps = 4
optimizer.zero_grad()

for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 2. 训练速度慢

**问题**：训练速度不理想

**解决方案**：
- 优化数据加载管道
- 使用更大的批次大小
- 启用混合精度训练
- 检查 DCU 利用率

```bash
# 监控 DCU 使用情况
watch -n 1 rocm-smi
```

### 3. 精度问题

**问题**：模型精度不收敛

**解决方案**：
- 检查学习率设置
- 验证数据预处理
- 使用学习率调度器
- 检查损失函数

## 最佳实践

1. **环境配置**
   - 使用容器化部署确保环境一致性
   - 设置合适的环境变量
   - 定期更新 DTK 和框架版本

2. **数据管理**
   - 使用高效的数据格式（如 HDF5、Parquet）
   - 实现数据预处理缓存
   - 合理设置数据加载参数

3. **模型设计**
   - 选择适合 DCU 架构的模型
   - 合理使用检查点和恢复机制
   - 实现模型版本管理

4. **训练策略**
   - 使用渐进式训练策略
   - 实现早停机制
   - 定期验证和测试

5. **监控调试**
   - 设置完善的日志记录
   - 使用可视化工具监控训练过程
   - 定期进行性能分析

## 相关资源

- [DCU 安装指南](installation.md)
- [模型微调指南](fine-tuning.md)
- [推理部署指南](inference.md)
- [HPC 应用指南](hpc.md)
- [海光开发者社区](https://developer.hpccube.com/)
- [光源镜像仓库](https://sourcefind.cn/)
