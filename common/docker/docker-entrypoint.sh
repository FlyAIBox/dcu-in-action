#!/bin/bash

# Docker容器入口脚本
# 用于启动DCU开发环境的各种服务

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# 显示欢迎信息
show_welcome() {
    cat << 'EOF'
================================================================
          🚀 海光DCU加速卡实战环境 🚀
================================================================
   
   环境信息:
   - 基础镜像: DCU PyTorch 2.1.0
   - Python: 3.10
   - DTK: 25.04
   - 项目目录: /workspace/dcu-in-action
   
   可用服务:
   - Jupyter Lab: http://localhost:8888
   - FastAPI服务: http://localhost:8000
   - Gradio界面: http://localhost:7860
   
   快速开始:
   - 检查环境: python examples/llm-inference/simple_test.py
   - 监控DCU: python scripts/utils/monitor_performance.py monitor
   - 启动推理: python examples/llm-inference/vllm_server.py
   
================================================================
EOF
}

# 检查DCU环境
check_dcu_env() {
    log_info "检查DCU环境..."
    
    # 检查DCU设备
    if command -v hy-smi >/dev/null 2>&1; then
        log_success "hy-smi 可用"
        hy-smi || log_warning "无法获取DCU信息，可能需要设备权限"
    else
        log_warning "hy-smi 不可用"
    fi
    
    # 检查PyTorch DCU支持
    python -c "
import torch
print('PyTorch版本:', torch.__version__)
print('DCU可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('DCU数量:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'DCU {i}:', torch.cuda.get_device_name(i))
else:
    print('注意: DCU不可用，可能需要映射设备或设置权限')
"
}

# 设置权限
setup_permissions() {
    log_info "设置文件权限..."
    
    # 确保脚本可执行
    find /workspace/dcu-in-action/scripts -name "*.sh" -exec chmod +x {} \;
    find /workspace/dcu-in-action/examples -name "*.py" -exec chmod +x {} \;
    
    log_success "权限设置完成"
}

# 启动Jupyter服务
start_jupyter() {
    if [ "$START_JUPYTER" = "true" ]; then
        log_info "启动Jupyter Lab..."
        cd /workspace/dcu-in-action
        nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
            --NotebookApp.token='' --NotebookApp.password='' \
            --notebook-dir=/workspace/dcu-in-action > /tmp/jupyter.log 2>&1 &
        log_success "Jupyter Lab 已启动，访问: http://localhost:8888"
    fi
}

# 启动监控服务
start_monitoring() {
    if [ "$START_MONITOR" = "true" ]; then
        log_info "启动DCU监控服务..."
        cd /workspace/dcu-in-action
        nohup python scripts/utils/monitor_performance.py monitor -i 5 -j > /tmp/monitor.log 2>&1 &
        log_success "DCU监控服务已启动"
    fi
}

# 启动推理服务
start_inference_server() {
    if [ "$START_INFERENCE" = "true" ] && [ -n "$MODEL_NAME" ]; then
        log_info "启动推理服务: $MODEL_NAME"
        cd /workspace/dcu-in-action
        nohup python examples/llm-inference/vllm_server.py \
            --mode server --model "$MODEL_NAME" \
            --host 0.0.0.0 --port 8000 > /tmp/inference.log 2>&1 &
        log_success "推理服务已启动，访问: http://localhost:8000"
    fi
}

# 运行环境检查
run_env_check() {
    if [ "$RUN_ENV_CHECK" = "true" ]; then
        log_info "运行环境检查..."
        cd /workspace/dcu-in-action
        bash scripts/setup/check_dcu_environment.sh
    fi
}

# 主函数
main() {
    show_welcome
    
    # 检查DCU环境
    check_dcu_env
    
    # 设置权限
    setup_permissions
    
    # 运行环境检查
    run_env_check
    
    # 启动服务
    start_jupyter
    start_monitoring
    start_inference_server
    
    # 根据传入的参数执行不同操作
    case "${1:-interactive}" in
        "jupyter")
            log_info "启动Jupyter模式..."
            cd /workspace/dcu-in-action
            exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
                --NotebookApp.token='' --NotebookApp.password='' \
                --notebook-dir=/workspace/dcu-in-action
            ;;
        "bash")
            log_info "启动Bash模式..."
            cd /workspace/dcu-in-action
            exec /bin/bash
            ;;
        "train")
            log_info "启动训练模式..."
            cd /workspace/dcu-in-action
            if [ -n "$2" ]; then
                exec python "$2" "${@:3}"
            else
                log_error "请指定训练脚本路径"
                exit 1
            fi
            ;;
        "inference")
            log_info "启动推理模式..."
            cd /workspace/dcu-in-action
            MODEL_NAME=${2:-"Qwen/Qwen-7B-Chat"}
            exec python examples/llm-inference/vllm_server.py \
                --mode server --model "$MODEL_NAME" --host 0.0.0.0 --port 8000
            ;;
        "monitor")
            log_info "启动监控模式..."
            cd /workspace/dcu-in-action
            exec python scripts/utils/monitor_performance.py monitor
            ;;
        "interactive"|*)
            log_info "启动交互模式..."
            cd /workspace/dcu-in-action
            
            # 显示可用命令
            echo ""
            log_info "可用命令:"
            echo "  python examples/llm-inference/simple_test.py     # 环境测试"
            echo "  python examples/llm-inference/chatglm_inference.py --mode chat  # 聊天测试"
            echo "  python scripts/utils/monitor_performance.py monitor  # DCU监控"
            echo "  jupyter lab --ip=0.0.0.0 --port=8888 --allow-root   # 启动Jupyter"
            echo ""
            
            # 进入bash
            exec /bin/bash
            ;;
    esac
}

# 信号处理
trap 'log_info "正在停止服务..."; pkill -f jupyter; pkill -f python; exit 0' SIGTERM SIGINT

# 执行主函数
main "$@" 