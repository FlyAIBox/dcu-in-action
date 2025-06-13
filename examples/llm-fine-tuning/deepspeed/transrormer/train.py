import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os
import json # 用于加载 DeepSpeed 配置

# --- 1. 环境与日志设置 ---
# 设置一些环境变量，优化分布式训练（可选但推荐）
os.environ["TOKENIZERS_PARALLELISM"] = "false" # 避免 Hugging Face Tokenizers 库的并行化警告
# os.environ["NCCL_P2P_DISABLE"] = "1" # 对于某些硬件配置或调试时可能有用，通常不需要禁用点对点通信
# os.environ["NCCL_IB_DISABLE"] = "0" # 确保 InfiniBand 开启，如果硬件支持，对多卡高速互联至关重要
# os.environ["NCCL_DEBUG"] = "INFO" # 打印 NCCL (NVIDIA Collective Communications Library) 调试信息，用于问题排查

# --- 2. DeepSpeed 配置定义 ---
# DeepSpeed 配置定义为 Python 字典，之后会转换为 JSON 字符串并保存到文件
deepspeed_config = {
    # 全局训练批次大小设置
    # 'auto': DeepSpeed 会根据 'train_micro_batch_size_per_gpu'、
    # 'gradient_accumulation_steps' 和 GPU 数量自动计算全局训练批次大小。
    # 这是模型在一个优化步骤中处理的样本总数。
    "train_batch_size": "auto",

    # 每张 GPU 上的微批次大小
    # 'auto': DeepSpeed 会根据 'gradient_accumulation_steps' 和 'train_batch_size' 自动优化。
    # 也可以在 TrainingArguments 中通过 per_device_train_batch_size 显式设置。
    # 这是实际放入 GPU 显存的最小批次单位。
    "train_micro_batch_size_per_gpu": "auto",

    # 梯度累积步数
    # 'auto': DeepSpeed 会自动管理。
    # 在执行一次优化器步骤（即更新模型权重）之前，累积梯度的步数。
    # 通过在多次微批次前向/后向传播后才执行一次权重更新，可以模拟更大的总批次大小，
    # 同时降低单次 GPU 显存需求。
    "gradient_accumulation_steps": "auto",

    # 优化器配置
    "optimizer": {
        "type": "AdamW", # 优化器类型，AdamW 是 Adam 优化器的变体，适合 Transformer 模型
        "params": {
            "lr": 2e-5,        # 初始学习率，大型语言模型微调常见起始点
            "betas": [0.9, 0.95], # Adam 优化器的动量参数，分别对应一阶矩和二阶矩估计
            "eps": 1e-8,       # 防止除零错误的小值
            "weight_decay": 0.01 # 权重衰减（L2 正则化）系数，防止模型过拟合
        }
    },

    # 学习率调度器配置
    "scheduler": {
        "type": "WarmupLR", # 调度器类型，WarmupLR 表示学习率会先从0逐渐预热到最大值
        "params": {
            "warmup_min_lr": 0,    # 预热阶段的起始学习率
            "warmup_max_lr": 2e-5, # 预热阶段的最终学习率（通常是最大学习率）
            "warmup_num_steps": 100 # 预热阶段持续的步数
        }
    },

    # 混合精度训练配置
    "fp16": {
        "enabled": True, # 启用混合精度训练（FP16）。这是现代 LLM 训练标配，显著减少显存并加速计算。
        "auto_cast": False # 通常与 PyTorch 的 torch.cuda.amp.autocast 配合使用，这里 DeepSpeed 自身管理。
    },

    # DeepSpeed ZeRO (Zero Redundancy Optimizer) 优化策略
    "zero_optimization": {
        "stage": 2, # ZeRO 阶段。
                    # Stage 1: 只对优化器状态（optimizer states）进行分片。
                    # Stage 2: 对优化器状态和梯度（gradients）进行分片。每张 GPU 负责存储其分配到的优化器状态和梯度。
                    # Stage 3: 对优化器状态、梯度和模型参数（parameters）进行分片。
                    # 建议：对于 8张64GB GPU 的配置，Stage 2 通常是最佳选择，平衡了内存效率和通信效率。
        "offload_optimizer": { # 将优化器状态卸载到 CPU
            "device": "cpu", # 将优化器状态存储在 CPU 内存中，显著节省 GPU 显存。
            "pin_memory": True # 启用锁页内存，加速 CPU 和 GPU 之间的数据传输。
        },
        "offload_param": { # 将模型参数卸载到 CPU （在 Stage 2 下通常不必要，但此处保留）
            "device": "cpu", # 在 Stage 3 下，模型参数会卸载到 CPU。在 Stage 2 下，此配置通常不发挥作用。
            "pin_memory": True # 启用锁页内存。
        },
        "overlap_comm": True, # 启用通信和计算重叠，隐藏分布式通信的延迟。
        "contiguous_gradients": True, # 将梯度存储在连续内存中，优化内存访问和通信效率。
        "sub_group_size": 1e9, # 主要用于 ZeRO-Stage3。定义参数分片时子组大小。
        "reduce_bucket_size": 5e8, # 用于梯度规约（gradient reduction）的桶大小。
        "stage3_prefetch_bucket_size": 5e8, # 主要用于 ZeRO-Stage3。预取桶的大小。
        "stage3_param_persistence_threshold": 1e4, # 主要用于 ZeRO-Stage3。小于此阈值的参数不会被卸载。
        "stage3_max_live_parameters": 1e9, # 主要用于 ZeRO-Stage3。允许的最大实时参数量。
        "stage3_max_reuse_distance": 1e9, # 主要用于 ZeRO-Stage3。参数重用的最大距离。
        "stage3_gather_fp16_weights_on_model_save": True # 主要用于 ZeRO-Stage3。保存模型时是否将 FP16 权重收集到一起。
    },

    "gradient_clipping": 1.0, # 梯度裁剪阈值，防止训练过程中梯度爆炸。
    "wall_clock_breakdown": False, # 是否打印详细的时间分析。调试时可设为 True。
    "checkpoint": {
        "tag": "DeepSeek-R1-Distill-Qwen-32B_fine-tuned" # DeepSpeed 检查点的标签或前缀。
    }
}

# 将 DeepSpeed 配置保存到 ds_config.json 文件
deepspeed_config_path = "ds_config.json"
with open(deepspeed_config_path, "w") as f:
    json.dump(deepspeed_config, f, indent=4)
print(f"DeepSpeed configuration saved to {deepspeed_config_path}")


# --- 3. 模型与分词器加载 ---
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# 加载模型：不进行 4-bit 量化，使用 bfloat16 精度（A100通常支持）
# device_map="auto" 让 Hugging Face Accelerate 自动将模型分发到所有可用 GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # 模型加载精度，对于A100建议使用bfloat16以获得更好数值稳定性
    device_map="auto", # 自动将模型权重分发到所有可用GPU
    trust_remote_code=True, # 允许加载自定义模型代码（如 Qwen 架构）
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# 对于某些模型（如 Qwen），需要手动设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # 对于因果语言模型，通常将 padding 放在右侧，避免影响注意力机制


# --- 4. 配置 LoRA 适配器 ---
# 强烈推荐开启梯度检查点，它能显著减少显存使用，同时对计算速度影响较小
model.gradient_checkpointing_enable()

# LoRA (Low-Rank Adaptation) 配置
lora_config = LoraConfig(
    r = 16, # LoRA 的秩（rank）。决定了适配器矩阵的维度和表达能力，通常 8, 16, 32。
    lora_alpha = 32, # LoRA 的缩放因子，通常设置为 r 的两倍。
    # target_modules 需要根据 DeepSeek-R1-Distill-Qwen-32B 模型的实际架构来确定，
    # 通常是注意力机制中的线性投影层 (e.g., Q, K, V, O 投影) 以及 MLP 中的线性层。
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout = 0.05, # LoRA 层的 dropout 概率，用于防止过拟合。
    bias = "none", # 决定是否微调偏置项。 "none" 表示不微调偏置项。
    task_type = "CAUSAL_LM", # 任务类型，这里是因果语言模型（即生成文本）。
)
model = get_peft_model(model, lora_config)

# 打印可训练参数量，方便验证 LoRA 配置是否生效
model.print_trainable_parameters()


# --- 5. 准备数据集 ---
# 定义 Prompt 模板，用于将原始数据集格式化为模型训练所需的输入
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# 从 Hugging Face Datasets Hub 加载 Dolly 2.0 数据集
dataset = load_dataset("databricks/dolly-v2-15k", split="train")

# 定义格式化函数，将原始数据转换为模型训练所需的 Prompt-Response 格式
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["context"] # Dolly 2.0 数据集中对应 'input' 的字段是 'context'
    outputs      = examples["response"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        if input_text: # 如果有输入上下文，则使用包含输入的模板
            text = alpaca_prompt.format(instruction, input_text, output) + tokenizer.eos_token
        else: # 如果没有输入上下文，则使用只有指令和响应的模板
            text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}" + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }

# 应用格式化函数到数据集，将原始数据转换为文本格式
dataset = dataset.map(formatting_prompts_func, batched = True,)

# Tokenize 数据集：将文本转换为模型可理解的 token ID
def tokenize_function(examples):
    # truncation=True: 截断超过 max_length 的序列
    # max_length=2048: 设置最大序列长度。DeepSeek-R1-Distill-Qwen-32B 通常支持 2048 或 4096。
    #                  根据您的数据平均长度和显存情况调整。
    # padding="max_length": 将所有序列填充到 max_length。
    return tokenizer(examples["text"], truncation=True, max_length=2048, padding="max_length")

# 对格式化后的数据集进行 tokenization，并移除原始文本列
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["instruction", "context", "response", "category", "text"])


# --- 6. 配置训练参数 (transformers.TrainingArguments) ---
training_args = TrainingArguments(
    output_dir="./fine_tuned_model", # 训练输出目录，保存检查点和日志
    num_train_epochs=3, # 训练轮数。通常 3-5 轮是常见起点。
    per_device_train_batch_size=4, #!!! 每张 GPU 上的微批次大小。对于 8x64G 卡，可以尝试 4-8，甚至更大。
                                   #    需要根据实际显存使用率和性能来调整。
    gradient_accumulation_steps=2, #!!! 梯度累积步数。在执行一次优化器更新前，累积梯度的次数。
                                   #    用于模拟更大的全局批次大小，例如：
                                   #    全局批次大小 = num_gpus * per_device_train_batch_size * gradient_accumulation_steps
                                   #    如果您有 8 张 GPU，每卡 batch_size=4，累积步数=2，则全局批次大小 = 8 * 4 * 2 = 64。
    learning_rate=2e-5, # 学习率。这是权重更新的步长。
    weight_decay=0.01, # 权重衰减（L2 正则化）系数，有助于防止过拟合。
    fp16=True, # 启用 FP16 混合精度训练。如果模型加载时使用的是 bfloat16，建议这里使用 bf16=True。
               # 对于 A100，bfloat16 更好，但 fp16 兼容性更广。
    logging_steps=10, # 每隔多少步记录一次训练日志（包括损失、学习率等）。
    save_steps=500, # 每隔多少步保存一次模型检查点。
    save_total_limit=2, # 最多保存的检查点数量。旧的检查点会被删除。
    overwrite_output_dir=True, # 如果输出目录已存在，则覆盖其内容。
    deepspeed=deepspeed_config_path, #!!! 指向 DeepSpeed 配置文件的路径。
    report_to="none", # 不使用任何训练报告工具。可选值有 "tensorboard", "wandb" 等。
    # 额外分布式训练参数，推荐用于多卡训练以提高效率和稳定性
    dataloader_drop_last=True, # 确保分布式训练的每个 GPU 上的批次大小一致，避免数据不均。
    ddp_find_unused_parameters=False, # 优化分布式训练效率，避免在分布式数据并行中寻找未使用的参数。
    group_by_length=True, # 动态批次填充。将长度相近的样本分到同一个批次，减少 padding 造成的计算浪费。
    logging_first_step=True, # 记录第一步的训练日志。
    optim="adamw_torch", # 明确使用 PyTorch 默认的 AdamW 优化器实现。
    lr_scheduler_type="cosine", # 学习率调度器类型，余弦衰减（cosine decay）是常见的选择。
    warmup_steps=100, # 学习率预热步数。在训练初期逐渐增加学习率，帮助模型稳定。
)

# --- 7. 创建 Trainer 并开始训练 ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset, # 使用 Tokenized 后的数据集进行训练
    tokenizer=tokenizer, # 必须传递 tokenizer，用于保存模型和生成文本
)

print(f"Starting training on {torch.cuda.device_count()} GPUs...")
trainer.train()

# --- 8. 保存微调后的模型 ---
# 保存 LoRA 适配器权重。这些权重是微调后的增量部分，体积很小。
# 加载时，将 LoRA 权重与原始大模型结合即可使用。
trainer.model.save_pretrained("./final_lora_model")
tokenizer.save_pretrained("./final_lora_model")

print("LoRA adapters and tokenizer saved to ./final_lora_model")

# 如果您希望保存合并 LoRA 权重到原始大模型后的完整模型，可以取消注释以下代码：
# 警告：合并后的模型会很大 (32B 半精度模型约占 64GB 磁盘空间)，
# 且需要足够大的 CPU 内存来完成合并过程。
# model_to_save = trainer.model.base_model.merge_and_unload()
# model_to_save.save_pretrained("./merged_full_model")
# tokenizer.save_pretrained("./merged_full_model")
# print("Merged full model and tokenizer saved to ./merged_full_model")