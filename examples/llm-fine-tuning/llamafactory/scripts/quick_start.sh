#!/bin/bash
# 大模型微调实战指南 - 快速开始脚本
# 一键完成环境配置、数据处理、模型训练的完整流程

set -e

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

# 显示帮助信息
show_help() {
    cat << EOF
大模型微调实战指南 - 快速开始脚本

用法: $0 [选项] <场景>

场景:
  customer-service    客服场景微调
  code-generation     代码生成场景微调  
  financial-qa        金融问答场景微调
  custom             自定义场景微调

选项:
  -h, --help         显示帮助信息
  -e, --env-only     仅安装环境，不进行训练
  -d, --data-only    仅处理数据，不进行训练
  -t, --train-only   仅进行训练（假设环境和数据已准备好）
  -s, --skip-deps    跳过依赖安装
  --dry-run          仅验证配置，不执行实际操作

示例:
  $0 customer-service              # 完整运行客服场景微调
  $0 -e customer-service          # 仅安装环境
  $0 -d code-generation           # 仅处理代码生成数据
  $0 --dry-run financial-qa       # 验证金融问答配置

EOF
}

# 解析命令行参数
ENV_ONLY=false
DATA_ONLY=false
TRAIN_ONLY=false
SKIP_DEPS=false
DRY_RUN=false
SCENARIO=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -e|--env-only)
            ENV_ONLY=true
            shift
            ;;
        -d|--data-only)
            DATA_ONLY=true
            shift
            ;;
        -t|--train-only)
            TRAIN_ONLY=true
            shift
            ;;
        -s|--skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        customer-service|code-generation|financial-qa|custom)
            SCENARIO=$1
            shift
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 验证参数
if [[ -z "$SCENARIO" ]]; then
    log_error "请指定场景类型"
    show_help
    exit 1
fi

log_info "开始执行大模型微调流程..."
log_info "场景: $SCENARIO"
log_info "当前目录: $(pwd)"

# 检查项目结构
check_project_structure() {
    log_info "检查项目结构..."
    
    required_dirs=("scripts/llamafactory" "examples/configs" "examples/datasets")
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log_error "缺少目录: $dir"
            log_info "请确保在项目根目录下运行此脚本"
            exit 1
        fi
    done
    
    log_success "项目结构检查通过"
}

# 环境安装
install_environment() {
    if [[ "$SKIP_DEPS" == "true" ]]; then
        log_info "跳过环境安装"
        return
    fi
    
    log_info "开始安装环境..."
    
    if [[ -f "scripts/llamafactory/install_llamafactory.sh" ]]; then
        bash scripts/llamafactory/install_llamafactory.sh
        log_success "环境安装完成"
    else
        log_error "安装脚本不存在: scripts/llamafactory/install_llamafactory.sh"
        exit 1
    fi
}

# 激活环境
activate_environment() {
    if [[ -f "llamafactory_env/bin/activate" ]]; then
        source llamafactory_env/bin/activate
        log_info "已激活虚拟环境"
    else
        log_warning "虚拟环境不存在，使用系统Python"
    fi
}

# 准备示例数据
prepare_sample_data() {
    log_info "准备示例数据..."
    
    case $SCENARIO in
        customer-service)
            if [[ ! -f "examples/datasets/customer_service_sample.json" ]]; then
                log_error "客服示例数据不存在"
                exit 1
            fi
            DATA_FILE="examples/datasets/customer_service_sample.json"
            ;;
        code-generation)
            if [[ ! -f "examples/datasets/code_generation_sample.json" ]]; then
                log_error "代码生成示例数据不存在"
                exit 1
            fi
            DATA_FILE="examples/datasets/code_generation_sample.json"
            ;;
        financial-qa)
            log_warning "金融问答数据需要您提供，请将数据放置在 examples/datasets/financial_qa.json"
            DATA_FILE="examples/datasets/financial_qa.json"
            if [[ ! -f "$DATA_FILE" ]]; then
                log_info "创建示例金融问答数据..."
                cat > "$DATA_FILE" << 'EOF'
[
  {
    "instruction": "解释什么是股票分红",
    "input": "",
    "output": "股票分红是指上市公司将部分利润以现金或股票的形式分配给股东的行为。分红通常包括现金分红和股票分红两种形式。现金分红是直接向股东账户发放现金，股票分红是向股东免费发放新股票。分红比例和时间由公司董事会决定，需要股东大会批准。"
  }
]
EOF
            fi
            ;;
        custom)
            log_info "自定义场景，请确保您的数据文件格式正确"
            DATA_FILE="examples/datasets/custom_data.json"
            ;;
    esac
    
    log_success "数据准备完成: $DATA_FILE"
}

# 获取配置文件
get_config_file() {
    case $SCENARIO in
        customer-service)
            CONFIG_FILE="examples/configs/customer_service_config.yaml"
            ;;
        code-generation)
            CONFIG_FILE="examples/configs/code_generation_config.yaml"
            ;;
        financial-qa)
            CONFIG_FILE="examples/configs/financial_qa_config.yaml"
            ;;
        custom)
            CONFIG_FILE="examples/configs/custom_config.yaml"
            if [[ ! -f "$CONFIG_FILE" ]]; then
                log_info "创建自定义配置文件..."
                cp examples/configs/customer_service_config.yaml "$CONFIG_FILE"
                log_info "请编辑 $CONFIG_FILE 以适配您的需求"
            fi
            ;;
    esac
    
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "配置文件不存在: $CONFIG_FILE"
        exit 1
    fi
    
    log_info "使用配置文件: $CONFIG_FILE"
}

# 验证配置
validate_config() {
    log_info "验证训练配置..."
    
    if [[ -f "scripts/llamafactory/train_model.py" ]]; then
        python scripts/llamafactory/train_model.py --config "$CONFIG_FILE" --dry_run
        if [[ $? -eq 0 ]]; then
            log_success "配置验证通过"
        else
            log_error "配置验证失败"
            exit 1
        fi
    else
        log_warning "训练脚本不存在，跳过验证"
    fi
}

# 开始训练
start_training() {
    log_info "开始模型训练..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "干运行模式，跳过实际训练"
        return
    fi
    
    # 创建输出目录
    OUTPUT_DIR="./saves/${SCENARIO}-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$OUTPUT_DIR"
    
    # 更新配置文件中的输出目录
    if command -v sed &> /dev/null; then
        sed -i.bak "s|output_dir:.*|output_dir: \"$OUTPUT_DIR\"|" "$CONFIG_FILE"
    fi
    
    log_info "训练输出目录: $OUTPUT_DIR"
    
    # 执行训练
    python scripts/llamafactory/train_model.py --config "$CONFIG_FILE"
    
    if [[ $? -eq 0 ]]; then
        log_success "训练完成！"
        log_info "训练结果保存在: $OUTPUT_DIR"
        
        # 显示后续步骤
        show_next_steps "$OUTPUT_DIR"
    else
        log_error "训练失败"
        exit 1
    fi
}

# 显示后续步骤
show_next_steps() {
    local output_dir=$1
    
    cat << EOF

🎉 训练完成！后续步骤：

1. 合并LoRA权重：
   llamafactory-cli export \\
     --model_name_or_path <base_model_path> \\
     --adapter_name_or_path $output_dir \\
     --template <template_name> \\
     --finetuning_type lora \\
     --export_dir ./merged_model

2. 启动推理服务：
   python scripts/llamafactory/inference_server.py \\
     --model_path ./merged_model \\
     --host 0.0.0.0 \\
     --port 8000

3. 评估模型性能：
   python scripts/llamafactory/evaluate_model.py \\
     --model_path ./merged_model \\
     --test_data examples/datasets/test_data.json

4. 查看训练日志：
   tensorboard --logdir $output_dir/logs

📚 更多信息请参考文档：
   - 理论篇：docs/llm-fine-tuning-theory.md
   - 实战篇：docs/llamafactory-practical-guide.md

EOF
}

# 主流程
main() {
    log_info "=========================================="
    log_info "  大模型微调实战指南 - 快速开始"
    log_info "=========================================="
    
    # 检查项目结构
    check_project_structure
    
    # 获取配置文件
    get_config_file
    
    # 环境安装
    if [[ "$DATA_ONLY" != "true" && "$TRAIN_ONLY" != "true" ]]; then
        install_environment
    fi
    
    # 激活环境
    activate_environment
    
    # 准备数据
    if [[ "$ENV_ONLY" != "true" && "$TRAIN_ONLY" != "true" ]]; then
        prepare_sample_data
    fi
    
    # 验证配置
    if [[ "$ENV_ONLY" != "true" && "$DATA_ONLY" != "true" ]]; then
        validate_config
    fi
    
    # 开始训练
    if [[ "$ENV_ONLY" != "true" && "$DATA_ONLY" != "true" ]]; then
        start_training
    fi
    
    log_success "所有步骤完成！"
}

# 错误处理
trap 'log_error "脚本执行过程中出现错误，退出码: $?"' ERR

# 运行主流程
main 