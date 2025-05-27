# DCU 模型微调指南

> 本文档整理自海光DCU开发社区和网络公开资料

## 概述

本指南介绍如何使用海光 DCU 加速卡进行模型微调，包括参数高效微调（PEFT）、LoRA、QLoRA 等先进技术，以及针对不同应用场景的微调策略。

## 微调方法概览

### 1. 全参数微调（Full Fine-tuning）
- **特点**：更新模型所有参数
- **优势**：效果最好，适应性强
- **劣势**：显存需求大，训练时间长
- **适用场景**：数据充足，计算资源充裕

### 2. 参数高效微调（PEFT）
- **特点**：只更新少量参数
- **优势**：显存需求小，训练快速
- **劣势**：效果可能略逊于全参数微调
- **适用场景**：资源受限，快速适配

### 3. 指令微调（Instruction Tuning）
- **特点**：使用指令-回答格式数据
- **优势**：提升模型遵循指令能力
- **适用场景**：对话系统，任务助手

## LoRA 微调

### 1. LoRA 原理

LoRA（Low-Rank Adaptation）通过低秩矩阵分解来近似全参数微调的效果：

```
W = W₀ + ΔW = W₀ + BA
```

其中：
- W₀：预训练权重（冻结）
- B：低秩矩阵（r×d）
- A：低秩矩阵（k×r）
- r：秩，通常远小于原始维度

### 2. LoRA 实现示例

```python
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer

# 加载基础模型
model_name = "facebook/opt-6.7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # 秩
    lora_alpha=32,  # 缩放参数
    lora_dropout=0.1,  # Dropout
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 目标模块
    bias="none",
)

# 应用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 数据准备
def format_instruction(example):
    return {
        "text": f"### 指令:\n{example['instruction']}\n\n### 回答:\n{example['output']}{tokenizer.eos_token}"
    }

# 训练参数
training_args = TrainingArguments(
    output_dir="./lora_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    fp16=True,
    learning_rate=2e-4,
    weight_decay=0.01,
    remove_unused_columns=False,
)

# 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# 保存 LoRA 权重
model.save_pretrained("./lora_weights")
```

### 3. LoRA 推理

```python
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(base_model, "./lora_weights")

# 推理
def generate_response(instruction, max_length=512):
    prompt = f"### 指令:\n{instruction}\n\n### 回答:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### 回答:\n")[1]

# 测试
response = generate_response("请解释什么是机器学习")
print(response)
```

## QLoRA 微调

### 1. QLoRA 特点

QLoRA（Quantized LoRA）结合了量化和 LoRA 技术：
- 使用 4-bit 量化存储预训练权重
- 在 16-bit 精度下进行 LoRA 微调
- 大幅减少显存使用

### 2. QLoRA 实现

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-13b",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 准备模型进行 k-bit 训练
model = prepare_model_for_kbit_training(model)

# LoRA 配置
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用 LoRA
model = get_peft_model(model, lora_config)

# 训练配置
training_args = TrainingArguments(
    output_dir="./qlora_model",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    fp16=True,
    learning_rate=2e-4,
    weight_decay=0.01,
    optim="paged_adamw_32bit",  # 使用分页优化器
)
```

## LLaMA-Factory 微调

### 1. 环境准备

```bash
# 克隆 LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 安装依赖
pip install -e .

# 安装额外依赖
pip install bitsandbytes transformers_stream_generator tiktoken
```

### 2. 数据准备

```json
// dataset_info.json
{
  "custom_dataset": {
    "file_name": "custom_data.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

```json
// custom_data.json
[
  {
    "instruction": "请解释什么是深度学习",
    "input": "",
    "output": "深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的复杂模式..."
  },
  {
    "instruction": "如何优化神经网络的性能？",
    "input": "",
    "output": "优化神经网络性能可以从以下几个方面入手：1. 调整网络架构..."
  }
]
```

### 3. 训练配置

```yaml
# train_config.yaml
model_name_or_path: facebook/opt-6.7b
dataset: custom_dataset
template: default
finetuning_type: lora
lora_target: q_proj,v_proj,k_proj,o_proj
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.1
output_dir: ./saves/opt-6.7b-lora
num_train_epochs: 3.0
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
lr_scheduler_type: cosine
learning_rate: 2.0e-4
max_grad_norm: 1.0
logging_steps: 10
save_steps: 500
warmup_steps: 100
fp16: true
```

### 4. 启动训练

```bash
# 命令行训练
llamafactory-cli train train_config.yaml

# 或使用 Python API
python -m llamafactory.train \
    --model_name_or_path facebook/opt-6.7b \
    --dataset custom_dataset \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj,k_proj,o_proj \
    --output_dir ./saves/opt-6.7b-lora \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 2e-4 \
    --num_train_epochs 3.0 \
    --fp16
```

### 5. Web UI 训练

```bash
# 启动 Web 界面
llamafactory-cli webui

# 或
python -m llamafactory.webui
```

## 多模态模型微调

### 1. 视觉语言模型微调

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import LoraConfig, get_peft_model
import torch

# 加载多模态模型
model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
processor = LlavaNextProcessor.from_pretrained(model_name)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA 配置（针对语言模型部分）
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用 LoRA
model = get_peft_model(model, lora_config)

# 数据预处理
def preprocess_multimodal_data(examples):
    texts = []
    images = []
    
    for example in examples:
        # 构建对话格式
        conversation = f"USER: <image>\n{example['question']}\nASSISTANT: {example['answer']}"
        texts.append(conversation)
        images.append(example['image'])
    
    # 处理输入
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    return inputs

# 训练循环
def train_multimodal_model(model, dataloader, optimizer, num_epochs):
    model.train()
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            
            outputs = model(**inputs)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### 2. 语音模型微调

```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from peft import LoraConfig, get_peft_model
import torch
import torchaudio

# 加载语音模型
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none"
)

# 应用 LoRA
model = get_peft_model(model, lora_config)

# 数据预处理
def preprocess_audio_data(batch):
    # 加载音频文件
    audio_arrays = []
    for audio_path in batch["audio_path"]:
        audio_array, sampling_rate = torchaudio.load(audio_path)
        # 重采样到 16kHz
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            audio_array = resampler(audio_array)
        audio_arrays.append(audio_array.squeeze().numpy())
    
    # 处理输入
    inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    
    # 处理标签
    with processor.as_target_processor():
        labels = processor(batch["transcription"], return_tensors="pt", padding=True)
    
    inputs["labels"] = labels["input_ids"]
    return inputs
```

## 领域特定微调

### 1. 医疗领域微调

```python
# 医疗数据格式
medical_data = [
    {
        "instruction": "根据症状诊断疾病",
        "input": "患者主诉：发热3天，咳嗽，胸痛",
        "output": "根据症状，可能的诊断包括：1. 肺炎 2. 支气管炎 3. 胸膜炎。建议进行胸部X光检查..."
    },
    {
        "instruction": "解释医学术语",
        "input": "什么是心房颤动？",
        "output": "心房颤动是一种常见的心律失常，特征是心房快速、不规则的电活动..."
    }
]

# 特殊的数据预处理
def format_medical_instruction(example):
    prompt = f"作为一名医生，请回答以下问题：\n\n问题：{example['input']}\n\n回答："
    return {
        "text": prompt + example["output"] + tokenizer.eos_token
    }
```

### 2. 法律领域微调

```python
# 法律数据格式
legal_data = [
    {
        "instruction": "法律咨询",
        "input": "合同违约的法律后果是什么？",
        "output": "合同违约的法律后果主要包括：1. 继续履行 2. 采取补救措施 3. 赔偿损失..."
    },
    {
        "instruction": "法条解释",
        "input": "请解释《民法典》第一百四十三条",
        "output": "《民法典》第一百四十三条规定了民事法律行为的有效要件..."
    }
]

# 法律领域特定配置
legal_training_args = TrainingArguments(
    output_dir="./legal_model",
    num_train_epochs=5,  # 法律领域可能需要更多轮次
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,  # 较低的学习率
    warmup_steps=200,
    weight_decay=0.01,
    fp16=True,
)
```

### 3. 金融领域微调

```python
# 金融数据格式
finance_data = [
    {
        "instruction": "投资建议",
        "input": "如何评估一只股票的投资价值？",
        "output": "评估股票投资价值可以从以下几个方面：1. 基本面分析 2. 技术面分析 3. 估值分析..."
    },
    {
        "instruction": "风险评估",
        "input": "什么是系统性风险？",
        "output": "系统性风险是指影响整个市场或经济系统的风险，无法通过分散投资来消除..."
    }
]
```

## 微调评估

### 1. 自动评估指标

```python
from datasets import load_metric
import numpy as np

# BLEU 评估
bleu_metric = load_metric("bleu")

def compute_bleu(predictions, references):
    # 分词
    predictions = [pred.split() for pred in predictions]
    references = [[ref.split()] for ref in references]
    
    result = bleu_metric.compute(predictions=predictions, references=references)
    return result

# ROUGE 评估
rouge_metric = load_metric("rouge")

def compute_rouge(predictions, references):
    result = rouge_metric.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )
    return {
        "rouge1": result["rouge1"].mid.fmeasure,
        "rouge2": result["rouge2"].mid.fmeasure,
        "rougeL": result["rougeL"].mid.fmeasure,
    }

# 困惑度评估
def compute_perplexity(model, tokenizer, texts):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()
```

### 2. 人工评估

```python
# 生成评估样本
def generate_evaluation_samples(model, tokenizer, test_instructions, num_samples=100):
    model.eval()
    results = []
    
    for instruction in test_instructions[:num_samples]:
        prompt = f"### 指令:\n{instruction}\n\n### 回答:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("### 回答:\n")[1]
        
        results.append({
            "instruction": instruction,
            "response": response
        })
    
    return results

# 保存评估结果
import json

evaluation_results = generate_evaluation_samples(model, tokenizer, test_instructions)
with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
```

## 微调优化技巧

### 1. 学习率调度

```python
from transformers import get_cosine_schedule_with_warmup

# 余弦退火学习率
num_training_steps = len(train_dataloader) * num_epochs
num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# 训练循环中使用
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # ... 训练代码 ...
        scheduler.step()
```

### 2. 早停机制

```python
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience

# 使用早停
early_stopping = EarlyStopping(patience=3)

for epoch in range(num_epochs):
    # 训练
    train_loss = train_epoch(model, train_dataloader)
    
    # 验证
    val_loss = validate_epoch(model, val_dataloader)
    
    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

### 3. 梯度裁剪

```python
from torch.nn.utils import clip_grad_norm_

# 训练循环中添加梯度裁剪
for batch in train_dataloader:
    optimizer.zero_grad()
    
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    
    # 梯度裁剪
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
```

## 常见问题和解决方案

### 1. 显存不足

**解决方案**：
- 使用 QLoRA 进行 4-bit 量化
- 减小批次大小，增加梯度累积步数
- 使用梯度检查点
- 启用 DeepSpeed ZeRO

```python
# DeepSpeed ZeRO 配置
deepspeed_config = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        },
        "offload_param": {
            "device": "cpu"
        }
    },
    "fp16": {
        "enabled": True
    }
}
```

### 2. 训练不稳定

**解决方案**：
- 降低学习率
- 增加 warmup 步数
- 使用梯度裁剪
- 检查数据质量

### 3. 过拟合

**解决方案**：
- 增加 dropout
- 使用权重衰减
- 早停机制
- 数据增强

## 最佳实践

1. **数据质量**
   - 确保数据格式一致
   - 去除低质量样本
   - 平衡不同类别的数据

2. **超参数调优**
   - 从较小的学习率开始
   - 逐步调整 LoRA 秩和 alpha
   - 使用验证集选择最佳参数

3. **训练监控**
   - 监控训练和验证损失
   - 定期生成样本检查质量
   - 使用 TensorBoard 可视化

4. **模型保存**
   - 定期保存检查点
   - 保存最佳模型
   - 记录训练配置

## 相关资源

- [DCU 安装指南](installation.md)
- [模型训练指南](training.md)
- [推理部署指南](inference.md)
- [HPC 应用指南](hpc.md)
- [LLaMA-Factory 官方文档](https://github.com/hiyouga/LLaMA-Factory)
- [PEFT 官方文档](https://github.com/huggingface/peft)
- [海光开发者社区](https://developer.hpccube.com/)
