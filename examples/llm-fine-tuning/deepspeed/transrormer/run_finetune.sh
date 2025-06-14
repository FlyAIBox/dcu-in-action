#!/bin/bash

# 设置环境变量以优化 ROCm/HIP 性能
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512,expandable_segments:True
export HSA_ENABLE_SDMA=0

# DeepSpeed 启动器
deepspeed --num_gpus 8 \
    --master_port 29500 \
    train.py \
    --deepspeed ds_config_zero2.json \
    \
    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
    --dataset_name "databricks/databricks-dolly-15k" \
    --max_seq_length 1024 \
    \
    --quantize_bits 4 \
    \
    --lora_r 16 \
    --lora_alpha 32 \
    \
    --output_dir "./fine_tuned_model" \
    --overwrite_output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --logging_first_step True \
    \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    \
    --dataloader_drop_last True \
    --ddp_find_unused_parameters False \
    \
    --lr_scheduler_type "cosine" \
    --warmup_steps 100 \
    --report_to "none" \
    --bf16 True