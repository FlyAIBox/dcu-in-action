#!/bin/bash
# Qwen3-32B 电力领域专用训练脚本
# 海光DCU k100-AI (64GB HBM2E) x 8

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║      Qwen3-32B 电力领域专业模型 8卡DCU微调训练             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

# 检查DCU环境
echo -e "\n${YELLOW}检查DCU环境...${NC}"
if ! command -v hy-smi &> /dev/null; then
    echo -e "${RED}错误: 未找到hy-smi命令，请确保DCU驱动已正确安装${NC}"
    exit 1
fi

# 显示DCU信息
echo -e "\n${YELLOW}DCU设备信息:${NC}"
hy-smi -L

# 检查可用DCU数量
DCU_COUNT=$(hy-smi -L | grep -c "DCU")
if [ "$DCU_COUNT" -lt 8 ]; then
    echo -e "${RED}警告: 检测到的DCU数量($DCU_COUNT)少于8个${NC}"
    echo -e "${YELLOW}是否继续使用$DCU_COUNT个DCU进行训练? (y/n)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
    WORLD_SIZE=$DCU_COUNT
else
    WORLD_SIZE=8
fi

# 设置DCU环境变量
echo -e "\n${YELLOW}设置DCU环境变量...${NC}"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 设置分布式训练环境变量
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=$WORLD_SIZE
export RANK=0

# 设置Python环境
echo -e "\n${YELLOW}激活Python环境...${NC}"
if [ -d "$HOME/miniconda3/envs/llamafactory-dcu" ]; then
    source $HOME/miniconda3/bin/activate llamafactory-dcu
elif [ -d "$HOME/anaconda3/envs/llamafactory-dcu" ]; then
    source $HOME/anaconda3/bin/activate llamafactory-dcu
else
    echo -e "${RED}错误: 未找到llamafactory-dcu环境，请先运行安装脚本${NC}"
    exit 1
fi

# 创建必要的目录
echo -e "\n${YELLOW}创建工作目录...${NC}"
mkdir -p saves/qwen3-32b-power-domain-8dcu
mkdir -p logs/qwen3-32b-power-domain-8dcu
mkdir -p data
mkdir -p results/power_eval

# 检查电力领域数据集
echo -e "\n${YELLOW}检查电力领域数据集...${NC}"
if [ ! -f "data/power_domain.json" ]; then
    echo -e "${YELLOW}未找到电力领域数据集，正在生成...${NC}"
    
    # 检查示例数据
    if [ -f "data/power_domain_samples.json" ]; then
        echo -e "${GREEN}找到示例数据，将基于示例数据生成完整数据集${NC}"
        python scripts/prepare_power_domain_dataset.py \
            --output_dir data \
            --dataset_name power_domain \
            --num_calc_problems 200 \
            --include_samples
    else
        echo -e "${YELLOW}生成新的电力领域数据集...${NC}"
        python scripts/prepare_power_domain_dataset.py \
            --output_dir data \
            --dataset_name power_domain \
            --num_calc_problems 300
    fi
fi

# 显示数据集统计
if [ -f "data/power_domain.json" ]; then
    SAMPLE_COUNT=$(python -c "import json; print(len(json.load(open('data/power_domain.json'))))")
    echo -e "${GREEN}电力领域数据集包含 $SAMPLE_COUNT 条样本${NC}"
fi

# 检查模型缓存目录
MODEL_CACHE_DIR="/root/AI-BOX/models"
if [ ! -d "$MODEL_CACHE_DIR" ]; then
    echo -e "${YELLOW}创建模型缓存目录: $MODEL_CACHE_DIR${NC}"
    mkdir -p $MODEL_CACHE_DIR
fi

# 显示训练配置
echo -e "\n${GREEN}电力领域微调配置:${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "模型: Qwen/Qwen2.5-32B-Instruct"
echo -e "领域: 电力系统专业知识"
echo -e "微调方法: LoRA (rank=384)"
echo -e "DCU数量: $WORLD_SIZE"
echo -e "批处理大小: 1 per device"
echo -e "梯度累积: 8 steps"
echo -e "有效批处理大小: $((1 * WORLD_SIZE * 8))"
echo -e "学习率: 5e-5 (专业领域优化)"
echo -e "训练轮数: 5"
echo -e "精度: BF16"
echo -e "DeepSpeed: Stage 3 (Zero Redundancy Optimizer)"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"

# 显示应用场景
echo -e "\n${GREEN}训练完成后可应用于:${NC}"
echo -e "• 电网调度决策支持"
echo -e "• 设备故障智能诊断"
echo -e "• 电力系统分析计算"
echo -e "• 技术文档自动理解"
echo -e "• 运维知识问答系统"

# 询问是否继续
echo -e "\n${YELLOW}是否开始电力领域专业模型训练? (y/n)${NC}"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo -e "${RED}训练已取消${NC}"
    exit 0
fi

# 启动分布式训练
echo -e "\n${GREEN}启动电力领域DeepSpeed分布式训练...${NC}"
echo -e "${YELLOW}提示: 训练日志将保存在 logs/qwen3-32b-power-domain-8dcu/${NC}"
echo -e "${YELLOW}提示: 使用 Ctrl+C 可以中断训练${NC}"
echo -e "${YELLOW}提示: 可以在另一个终端运行 ./scripts/monitor_dcu_training.sh 监控训练${NC}"

# 记录开始时间
START_TIME=$(date +%s)

# 使用torchrun启动分布式训练
torchrun \
    --nnodes=1 \
    --nproc_per_node=$WORLD_SIZE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $(which llamafactory-cli) train \
    configs/qwen3_32b_power_domain_8dcu.yaml \
    2>&1 | tee logs/qwen3-32b-power-domain-8dcu/train.log

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

# 检查训练结果
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}电力领域模型训练完成!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}训练时长: ${HOURS}小时${MINUTES}分钟${NC}"
    echo -e "${GREEN}模型保存在: saves/qwen3-32b-power-domain-8dcu/${NC}"
    echo -e "${GREEN}日志保存在: logs/qwen3-32b-power-domain-8dcu/${NC}"
    
    # 显示最终checkpoint
    LATEST_CHECKPOINT=$(ls -t saves/qwen3-32b-power-domain-8dcu/checkpoint-* 2>/dev/null | head -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo -e "${GREEN}最新checkpoint: $LATEST_CHECKPOINT${NC}"
    fi
    
    # 显示模型评估命令
    echo -e "\n${GREEN}下一步操作建议:${NC}"
    echo -e "${BLUE}1. 评估模型性能:${NC}"
    echo -e "   llamafactory-cli eval configs/eval_power_domain.yaml"
    
    echo -e "\n${BLUE}2. 合并LoRA权重:${NC}"
    echo -e "   cat > configs/merge_power_domain.yaml << EOF"
    echo -e "model_name_or_path: Qwen/Qwen2.5-32B-Instruct"
    echo -e "adapter_name_or_path: saves/qwen3-32b-power-domain-8dcu"
    echo -e "export_dir: models/qwen3-32b-power-domain"
    echo -e "export_size: 32"
    echo -e "export_device: auto"
    echo -e "export_legacy_format: false"
    echo -e "EOF"
    echo -e "   llamafactory-cli export configs/merge_power_domain.yaml"
    
    echo -e "\n${BLUE}3. 启动电力领域助手服务:${NC}"
    echo -e "   python scripts/inference_server.py \\"
    echo -e "       --model_path models/qwen3-32b-power-domain \\"
    echo -e "       --device dcu \\"
    echo -e "       --system_prompt \"你是一个电力系统专家，精通电力系统分析、设备运维和故障诊断。\" \\"
    echo -e "       --port 8001"
    
    echo -e "\n${BLUE}4. 测试电力领域问答:${NC}"
    echo -e "   curl -X POST http://localhost:8001/v1/chat/completions \\"
    echo -e "       -H \"Content-Type: application/json\" \\"
    echo -e "       -d '{"
    echo -e "         \"messages\": ["
    echo -e "           {\"role\": \"user\", \"content\": \"什么是电力系统的N-1准则？\"}"
    echo -e "         ]"
    echo -e "       }'"
    
else
    echo -e "\n${RED}训练过程中出现错误，请检查日志文件${NC}"
    echo -e "${RED}日志位置: logs/qwen3-32b-power-domain-8dcu/train.log${NC}"
    exit 1
fi

echo -e "\n${GREEN}电力领域专业模型训练脚本执行完成!${NC}" 