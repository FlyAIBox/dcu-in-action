#!/bin/bash

# 海光DCU K100-AI大模型推理环境自动配置脚本
# 版本: v2.0
# 作者: DCU-in-Action Team

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

# 检查是否为root用户
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_warning "建议不要使用root用户运行此脚本"
        read -p "是否继续? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# 检查系统要求
check_system_requirements() {
    log_info "检查系统要求..."
    
    # 检查操作系统
    if [[ ! -f /etc/os-release ]]; then
        log_error "无法确定操作系统类型"
        exit 1
    fi
    
    source /etc/os-release
    if [[ "$ID" != "ubuntu" ]] || [[ ! "$VERSION_ID" =~ ^22\.04 ]]; then
        log_warning "推荐使用Ubuntu 22.04，当前系统: $PRETTY_NAME"
    fi
    
    # 检查内核版本
    KERNEL_VERSION=$(uname -r)
    log_info "内核版本: $KERNEL_VERSION"
    
    # 检查DCU设备
    if ! command -v rocm-smi &> /dev/null; then
        log_error "rocm-smi未找到，请先安装DCU驱动"
        exit 1
    fi
    
    # 检查DCU设备数量
    DCU_COUNT=$(rocm-smi --showid | grep -c "GPU\[" || echo "0")
    log_info "检测到 $DCU_COUNT 个DCU设备"
    
    if [[ $DCU_COUNT -eq 0 ]]; then
        log_error "未检测到DCU设备"
        exit 1
    fi
    
    log_success "系统要求检查完成"
}

# 安装系统依赖
install_system_dependencies() {
    log_info "安装系统依赖..."
    
    sudo apt update
    
    # 基础工具
    sudo apt install -y \
        build-essential \
        cmake \
        git \
        git-lfs \
        curl \
        wget \
        jq \
        htop \
        iotop \
        nethogs \
        numactl \
        hwloc \
        python3-dev \
        python3-pip \
        python3-venv
    
    # 网络工具
    sudo apt install -y \
        net-tools \
        iperf3 \
        tcpdump \
        wireshark-common
    
    log_success "系统依赖安装完成"
}

# 安装Miniconda
install_miniconda() {
    log_info "检查Miniconda安装..."
    
    if command -v conda &> /dev/null; then
        log_success "Miniconda已安装"
        return
    fi
    
    log_info "安装Miniconda..."
    cd /tmp
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    
    # 添加到PATH
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    # 初始化conda
    ~/miniconda3/bin/conda init bash
    
    log_success "Miniconda安装完成"
}

# 创建Python虚拟环境
create_python_environments() {
    log_info "创建Python虚拟环境..."
    
    # 确保conda可用
    source ~/.bashrc
    
    # vLLM环境
    log_info "创建vLLM环境..."
    conda create -n vllm python=3.10 -y
    source ~/miniconda3/bin/activate vllm
    
    # 安装PyTorch for ROCm
    pip install torch==2.4.0+rocm6.0 torchvision==0.19.0+rocm6.0 \
        --index-url https://download.pytorch.org/whl/rocm6.0
    
    # 安装vLLM
    pip install vllm==0.6.2
    
    # 安装其他依赖
    pip install \
        transformers==4.45.0 \
        accelerate==0.27.0 \
        datasets==2.19.0 \
        openai==1.3.0 \
        aiohttp==3.9.0 \
        fastapi==0.104.0 \
        uvicorn==0.24.0 \
        pydantic==2.5.0 \
        numpy==1.24.3 \
        pandas==2.1.0 \
        matplotlib==3.7.2 \
        seaborn==0.12.2 \
        psutil==5.9.6 \
        py3nvml==0.2.7
    
    conda deactivate
    
    # SGLang环境
    log_info "创建SGLang环境..."
    conda create -n sglang python=3.10 -y
    source ~/miniconda3/bin/activate sglang
    
    pip install torch==2.4.0+rocm6.0 --index-url https://download.pytorch.org/whl/rocm6.0
    pip install sglang[all]==0.3.0
    pip install transformers==4.45.0 accelerate==0.27.0
    
    conda deactivate
    
    # Xinference环境
    log_info "创建Xinference环境..."
    conda create -n xinference python=3.10 -y
    source ~/miniconda3/bin/activate xinference
    
    pip install "xinference[all]==0.15.0"
    pip install transformers==4.45.0 accelerate==0.27.0
    
    conda deactivate
    
    # 通用测试环境
    log_info "创建通用测试环境..."
    conda create -n benchmark python=3.10 -y
    source ~/miniconda3/bin/activate benchmark
    
    pip install \
        requests==2.31.0 \
        aiohttp==3.9.0 \
        asyncio==3.4.3 \
        pandas==2.1.0 \
        matplotlib==3.7.2 \
        seaborn==0.12.2 \
        plotly==5.17.0 \
        jupyter==1.0.0 \
        notebook==7.0.0 \
        psutil==5.9.6 \
        pyyaml==6.0.1 \
        tqdm==4.66.0 \
        click==8.1.7
    
    conda deactivate
    
    log_success "Python虚拟环境创建完成"
}

# 创建目录结构
create_directory_structure() {
    log_info "创建目录结构..."
    
    BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    
    mkdir -p "$BASE_DIR"/{scripts,benchmark,tools,configs,results,models,logs}
    mkdir -p "$BASE_DIR"/results/{vllm,sglang,xinference}
    mkdir -p "$BASE_DIR"/logs/{vllm,sglang,xinference,system}
    
    log_success "目录结构创建完成"
}

# 下载模型（可选）
download_models() {
    log_info "是否下载测试模型？"
    read -p "下载DeepSeek-7B和Qwen-7B模型用于测试? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "跳过模型下载"
        return
    fi
    
    log_info "下载测试模型..."
    
    BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    MODEL_DIR="$BASE_DIR/models"
    
    cd "$MODEL_DIR"
    
    # 配置git-lfs
    git lfs install
    
    # 下载DeepSeek-7B
    if [[ ! -d "deepseek-llm-7b-base" ]]; then
        log_info "下载DeepSeek-LLM-7B-Base..."
        git clone https://huggingface.co/deepseek-ai/deepseek-llm-7b-base
    fi
    
    # 下载Qwen-7B
    if [[ ! -d "Qwen-7B" ]]; then
        log_info "下载Qwen-7B..."
        git clone https://huggingface.co/Qwen/Qwen-7B
    fi
    
    log_success "模型下载完成"
}

# 配置环境变量
configure_environment() {
    log_info "配置环境变量..."
    
    # 创建环境配置文件
    cat > ~/.dcu_inference_env << EOF
# DCU推理环境配置
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROCM_PATH=/opt/rocm
export PATH=\$ROCM_PATH/bin:\$PATH
export LD_LIBRARY_PATH=\$ROCM_PATH/lib:\$LD_LIBRARY_PATH

# 优化设置
export VLLM_ATTENTION_BACKEND=FLASHINFER
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0

# 模型路径
export MODEL_BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
export RESULTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/results"
export LOGS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/logs"
EOF
    
    # 添加到bashrc
    if ! grep -q "source ~/.dcu_inference_env" ~/.bashrc; then
        echo "source ~/.dcu_inference_env" >> ~/.bashrc
    fi
    
    log_success "环境变量配置完成"
}

# 验证安装
verify_installation() {
    log_info "验证安装..."
    
    # 验证conda环境
    source ~/.bashrc
    
    # 检查vLLM
    log_info "检查vLLM环境..."
    source ~/miniconda3/bin/activate vllm
    python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" || log_error "vLLM验证失败"
    conda deactivate
    
    # 检查SGLang
    log_info "检查SGLang环境..."
    source ~/miniconda3/bin/activate sglang
    python -c "import sglang; print('SGLang验证成功')" || log_error "SGLang验证失败"
    conda deactivate
    
    # 检查Xinference
    log_info "检查Xinference环境..."
    source ~/miniconda3/bin/activate xinference
    python -c "import xinference; print('Xinference验证成功')" || log_error "Xinference验证失败"
    conda deactivate
    
    # 检查DCU状态
    log_info "检查DCU状态..."
    rocm-smi --showid
    
    log_success "安装验证完成"
}

# 主函数
main() {
    log_info "开始海光DCU K100-AI大模型推理环境配置..."
    
    check_root
    check_system_requirements
    install_system_dependencies
    install_miniconda
    create_python_environments
    create_directory_structure
    download_models
    configure_environment
    verify_installation
    
    log_success "环境配置完成！"
    log_info "请运行 'source ~/.bashrc' 或重新打开终端以加载环境变量"
    log_info "然后可以运行测试脚本: ./scripts/check_dcu_status.sh"
}

# 执行主函数
main "$@" 