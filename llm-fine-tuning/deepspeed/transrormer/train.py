import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    HfArgumentParser,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os
import json
from dataclasses import dataclass, field
from typing import Optional
import bitsandbytes as bnb

# --- 1. 环境与日志设置 ---
# 设置一些环境变量，优化分布式训练（可选但推荐）
os.environ["TOKENIZERS_PARALLELISM"] = "false" # 避免 Hugging Face Tokenizers 库的并行化警告

# --- 关键修改：针对 ROCm 内存优化和多 GPU 调试 ---
# PYTORCH_HIP_ALLOC_CONF：解决 HIP OOM 错误中提到的内存碎片化问题。
# "expandable_segments:True" 允许 PyTorch 的 HIP 后端使用可扩展的内存段，提高内存利用率。
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:512,expandable_segments:True"

# NCCL_IB_DISABLE：确保 InfiniBand 开启（如果您的硬件支持），对多卡高速互联至关重要。
# 设置为 "0" 明确启用 InfiniBand。
os.environ["NCCL_IB_DISABLE"] = "0"

# NCCL_DEBUG：打印 NCCL (NVIDIA Collective Communications Library) 调试信息。
# "INFO" 级别会提供关于分布式通信的详细日志，有助于问题排查。
os.environ["NCCL_DEBUG"] = "INFO"
# os.environ["NCCL_P2P_DISABLE"] = "1" # 通常不需要禁用点对点通信，除非有特定问题

# HSA_ENABLE_SDMA：确保 AMD GPU 的 SDMA 功能启用，对多卡高速互联至关重要。
# 设置为 "0" 明确禁用 SDMA。
os.environ["HSA_ENABLE_SDMA"] = "0"


# --- 2. 参数定义 ---
@dataclass
class ModelArguments:
    """
    与模型和 LoRA 配置相关的参数
    """
    model_name_or_path: str = field(
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        metadata={"help": "预训练模型的路径或 Hugging Face Hub 上的模型标识符"}
    )
    load_in_4bit: bool = field(
        default=True,
        metadata={"help": "是否使用4bit量化加载模型"}
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "是否使用8bit量化加载模型"}
    )
    lora_r: int = field(default=16, metadata={"help": "LoRA 的秩 (rank)"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA 的 alpha 缩放因子"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA 层的 dropout 概率"})
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "要应用 LoRA 的模块列表"}
    )

@dataclass
class DataArguments:
    """
    与数据处理相关的参数
    """
    dataset_name: str = field(
        default="databricks/databricks-dolly-15k", 
        metadata={"help": "要使用的数据集名称 (通过 datasets 库)"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Tokenization 后的最大总输入序列长度"}
    )

def find_all_linear_names(model):
    """找出所有需要量化的线性层名称"""
    cls = bnb.nn.Linear4bit if model_args.load_in_4bit else bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[-1])
    return list(lora_module_names)

def main():
    # --- 3. 解析命令行参数 ---
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- 5. 模型与分词器加载 ---
    print(f"Loading model: {model_args.model_name_or_path}")
    
    # 量化配置
    quantization_config = None
    if model_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=quantization_config,
        trust_remote_code=True,
        use_cache=False,  # 训练时禁用 KV cache 以节省显存
    )
    
    # 准备量化训练
    if model_args.load_in_4bit or model_args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 6. 配置 LoRA 适配器 ---
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # 自动发现需要量化的模块
    if model_args.load_in_4bit or model_args.load_in_8bit:
        model_args.lora_target_modules = find_all_linear_names(model)
    
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=model_args.lora_target_modules,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 7. 准备数据集 ---
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
    
    print(f"Loading dataset: {data_args.dataset_name}")
    dataset = load_dataset(data_args.dataset_name, split="train")

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["context"]
        outputs      = examples["response"]
        texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            if input_text:
                text = alpaca_prompt.format(instruction, input_text, output) + tokenizer.eos_token
            else:
                text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}" + tokenizer.eos_token
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=data_args.max_seq_length, 
            padding="max_length"
        )

    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["instruction", "context", "response", "category", "text"]
    )

    # --- 8. 创建 Trainer 并开始训练 ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    print(f"Starting training on {training_args.world_size} GPUs...")
    trainer.train()

    # --- 9. 保存微调后的模型 ---
    print(f"Saving final LoRA model to {training_args.output_dir}")
    trainer.model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print("Training complete.")

if __name__ == "__main__":
    main()