#!/bin/bash
# DCU k100-AI 优化配置脚本
# 针对海光DCU k100-AI加速卡的LLaMA Factory环境配置

set -e

echo "🚀 开始配置海光DCU k100-AI环境..."

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查系统环境
check_environment() {
    log_info "检查系统环境..."
    
    # 检查操作系统
    if [[ ! -f /etc/os-release ]]; then
        log_error "无法识别操作系统"
        return 1
    fi
    
    source /etc/os-release
    if [[ "$ID" != "ubuntu" ]]; then
        log_warning "推荐使用Ubuntu 22.04，当前系统: $PRETTY_NAME"
    fi
    
    # 检查内核版本
    KERNEL_VERSION=$(uname -r)
    log_info "内核版本: $KERNEL_VERSION"
    
    # 检查DCU驱动
    if command -v dcu-smi &> /dev/null; then
        log_success "DCU驱动已安装"
        dcu-smi -L
    else
        log_warning "DCU驱动未安装，请先安装ROCK驱动"
        return 1
    fi
    
    # 检查DTK
    if [[ -d "/opt/dtk" ]]; then
        log_success "DTK已安装"
        DTK_VERSION=$(cat /opt/dtk/VERSION 2>/dev/null || echo "未知版本")
        log_info "DTK版本: $DTK_VERSION"
    else
        log_warning "DTK未安装，请先安装DTK工具包"
        return 1
    fi
    
    return 0
}

# 设置DCU环境变量
setup_dcu_environment() {
    log_info "配置DCU环境变量..."
    
    # 创建环境配置文件
    cat > ~/.dcurc << 'EOF'
# 海光DCU k100-AI 环境配置
export HIP_VISIBLE_DEVICES=0
export ROCM_PATH=/opt/dtk
export HIP_PLATFORM=hcc
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$ROCM_PATH/python:$PYTHONPATH

# DCU性能优化
export HIP_LAUNCH_BLOCKING=0
export HIP_FORCE_DEV_KERNARG=1
export MIOPEN_FIND_ENFORCE=1
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_cache

# 内存优化
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072

echo "🚀 DCU k100-AI 环境已加载"
EOF

    # 添加到bashrc
    if ! grep -q "source ~/.dcurc" ~/.bashrc; then
        echo "source ~/.dcurc" >> ~/.bashrc
        log_success "环境变量已添加到 ~/.bashrc"
    fi
    
    # 立即加载环境变量
    source ~/.dcurc
}

# 创建DCU优化配置文件
create_dcu_configs() {
    log_info "创建DCU优化配置文件..."
    
    # 创建配置目录
    CONFIG_DIR="$HOME/dcu_configs"
    mkdir -p "$CONFIG_DIR"
    
    # DCU训练配置模板
    cat > "$CONFIG_DIR/qwen2.5_3b_dcu.json" << 'EOF'
{
    "model_name": "qwen2.5-3b-instruct",
    "stage": "sft",
    "do_train": true,
    "finetuning_type": "lora",
    "lora_target": "q_proj,k_proj,v_proj,o_proj",
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "dataset": "financial_reports",
    "dataset_dir": "data",
    "template": "qwen",
    "cutoff_len": 2048,
    "max_samples": 10000,
    "overwrite_cache": true,
    "preprocessing_num_workers": 8,
    "output_dir": "saves/qwen2.5-3b-dcu",
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 500,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 100,
    "num_train_epochs": 8,
    "max_grad_norm": 1.0,
    "bf16": true,
    "dataloader_pin_memory": true,
    "dataloader_num_workers": 4,
    "gradient_checkpointing": true,
    "ddp_timeout": 3600,
    "include_num_input_tokens_seen": true,
    "val_size": 0.1,
    "eval_strategy": "steps",
    "save_strategy": "steps",
    "load_best_model_at_end": true,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": false,
    "save_total_limit": 5
}
EOF

    # 启动脚本
    cat > "$CONFIG_DIR/start_webui.sh" << 'EOF'
#!/bin/bash
# DCU LLaMA Factory Web UI 启动脚本

# 激活环境
source ~/.dcurc
if command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate llamafactory-dcu 2>/dev/null || echo "注意：conda环境llamafactory-dcu未找到"
fi

# 设置DCU环境变量
export HIP_VISIBLE_DEVICES=0
export USE_MODELSCOPE_HUB=1

# 检查LLaMA Factory是否安装
if ! command -v llamafactory-cli &> /dev/null; then
    echo "❌ LLaMA Factory未安装，请先运行安装脚本"
    exit 1
fi

# 启动Web UI
echo "🚀 启动LLaMA Factory Web UI..."
echo "访问地址: http://localhost:7860"
cd ~/LLaMA-Factory 2>/dev/null || echo "警告：LLaMA-Factory目录不存在，从当前目录启动"
llamafactory-cli webui --host 0.0.0.0 --port 7860
EOF

    chmod +x "$CONFIG_DIR/start_webui.sh"
    
    # 性能监控脚本
    cat > "$CONFIG_DIR/monitor_dcu.sh" << 'EOF'
#!/bin/bash
# DCU性能监控脚本

echo "🔍 DCU k100-AI 性能监控"
echo "=========================="

# 检查dcu-smi是否可用
if ! command -v dcu-smi &> /dev/null; then
    echo "❌ dcu-smi命令未找到，请检查DCU驱动安装"
    exit 1
fi

while true; do
    clear
    echo "时间: $(date)"
    echo "------------------------"
    
    # DCU使用情况
    echo "📊 DCU使用情况:"
    if dcu-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null; then
        dcu-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | \
        awk -F',' '{printf "DCU %s: %s | 显存: %s/%s MB | 利用率: %s%% | 温度: %s°C\n", $1, $2, $3, $4, $5, $6}'
    else
        echo "简化模式显示:"
        dcu-smi -L 2>/dev/null || echo "无法获取DCU信息"
    fi
    
    echo ""
    
    # 系统资源
    echo "💻 系统资源:"
    if command -v top &> /dev/null; then
        CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//' 2>/dev/null || echo "N/A")
        echo "CPU使用率: $CPU_USAGE"
    fi
    
    if command -v free &> /dev/null; then
        MEMORY_INFO=$(free -h | grep Mem | awk '{printf "已用: %s / 总计: %s", $3, $2}' 2>/dev/null || echo "N/A")
        echo "内存使用: $MEMORY_INFO"
    fi
    
    echo ""
    echo "按 Ctrl+C 退出监控"
    echo "=========================="
    
    sleep 2
done
EOF

    chmod +x "$CONFIG_DIR/monitor_dcu.sh"
    
    log_success "DCU配置文件已创建在 $CONFIG_DIR"
}

# 创建使用说明
create_usage_guide() {
    log_info "创建使用说明..."
    
    cat > "$HOME/DCU_K100_AI_GUIDE.md" << 'EOF'
# 海光DCU k100-AI LLaMA Factory 使用指南

## 🚀 快速开始

### 1. 启动Web UI
```bash
~/dcu_configs/start_webui.sh
```
访问: http://localhost:7860

### 2. 监控DCU性能
```bash
~/dcu_configs/monitor_dcu.sh
```

### 3. 环境激活
```bash
source ~/.dcurc
conda activate llamafactory-dcu  # 如果使用conda环境
```

## 📊 推荐配置

### Qwen2.5-3B模型
- 批处理大小: 4-8
- 梯度累积: 4-8
- 学习率: 2e-4
- LoRA rank: 32
- 精度: bf16

### 显存优化
- gradient_checkpointing: true
- dataloader_pin_memory: true
- 最大序列长度: 2048

## 🔧 常用命令

### 训练模型
```bash
llamafactory-cli train ~/dcu_configs/qwen2.5_3b_dcu.json
```

### 推理测试
```bash
llamafactory-cli chat \
    --model_name qwen2.5-3b-instruct \
    --checkpoint_dir saves/qwen2.5-3b-dcu
```

## ⚠️ 注意事项

1. 确保DCU驱动版本 >= 6.3.8
2. 使用bf16精度以获得最佳性能
3. 监控显存使用，避免OOM
4. 定期清理缓存目录

## 🆘 故障排除

### 显存不足
- 减小batch_size
- 启用gradient_checkpointing
- 减少max_length

### 训练慢
- 检查DataLoader参数
- 启用混合精度
- 优化数据预处理

## 📞 支持

- DCU官方文档: https://developer.hygon.cn
- LLaMA Factory: https://github.com/hiyouga/LLaMA-Factory
- 问题反馈: 请提交Issue到项目仓库
EOF

    log_success "使用指南已创建: $HOME/DCU_K100_AI_GUIDE.md"
}

# 显示帮助信息
show_help() {
    cat << 'EOF'
DCU k100-AI LLaMA Factory 配置脚本

用法: ./dcu_k100_ai_setup.sh [选项]

选项:
  --check-only    仅检查环境，不进行配置
  --config-only   仅创建配置文件，不检查环境
  --help, -h      显示此帮助信息

示例:
  ./dcu_k100_ai_setup.sh              # 完整安装配置
  ./dcu_k100_ai_setup.sh --check-only # 仅检查环境
  ./dcu_k100_ai_setup.sh --config-only # 仅创建配置

EOF
}

# 主函数
main() {
    echo "🎯 海光DCU k100-AI LLaMA Factory 环境配置"
    echo "=========================================="
    
    # 解析命令行参数
    case "${1:-}" in
        --help|-h)
            show_help
            exit 0
            ;;
        --check-only)
            if check_environment; then
                log_success "✅ 环境检查通过"
            else
                log_error "❌ 环境检查失败"
                exit 1
            fi
            exit 0
            ;;
        --config-only)
            setup_dcu_environment
            create_dcu_configs
            create_usage_guide
            log_success "✅ 配置文件创建完成"
            exit 0
            ;;
        "")
            # 默认：完整配置
            log_info "开始完整环境配置..."
            
            if check_environment; then
                log_success "✅ 环境检查通过"
            else
                log_error "❌ 环境检查失败，请先安装DCU驱动和DTK"
                echo ""
                echo "📋 安装指南："
                echo "1. DCU驱动(ROCK): https://developer.hygon.cn"
                echo "2. DTK工具包: https://developer.hygon.cn" 
                exit 1
            fi
            
            setup_dcu_environment
            create_dcu_configs
            create_usage_guide
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
    
    echo ""
    log_success "🎉 DCU k100-AI环境配置完成！"
    echo ""
    echo "📋 下一步操作:"
    echo "1. 重新登录以加载环境变量，或运行: source ~/.bashrc"
    echo "2. 启动Web UI: ~/dcu_configs/start_webui.sh"
    echo "3. 查看使用指南: cat ~/DCU_K100_AI_GUIDE.md"
    echo "4. 监控DCU性能: ~/dcu_configs/monitor_dcu.sh"
    echo ""
    echo "🚀 享受DCU k100-AI的强大性能吧！"
}

# 运行主函数
main "$@" 