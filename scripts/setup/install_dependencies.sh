#!/bin/bash

# 海光DCU开发环境依赖安装脚本
# 自动安装和配置DCU开发所需的软件包和Python库

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置参数
PYTHON_VERSION="3.10"
PYTORCH_VERSION="2.1.0"
DTK_VERSION="25.04"
INSTALL_OPTIONAL=false
USE_CONDA=false
MIRROR_SITE="default"

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

# 帮助信息
show_help() {
    cat << EOF
海光DCU开发环境依赖安装脚本

用法: $0 [选项]

选项:
    -h, --help              显示此帮助信息
    -p, --python VERSION    指定Python版本 (默认: 3.10)
    -t, --torch VERSION     指定PyTorch版本 (默认: 2.1.0)
    -d, --dtk VERSION       指定DTK版本 (默认: 25.04)
    -o, --optional          安装可选组件 (vLLM, Jupyter等)
    -c, --conda             使用Conda环境管理
    -m, --mirror SITE       使用镜像源 (tsinghua, aliyun, ustc)
    --dry-run              仅显示将要执行的命令，不实际安装

示例:
    $0                      # 基础安装
    $0 -o                   # 包含可选组件
    $0 -c -m tsinghua       # 使用Conda和清华镜像
    $0 --dry-run            # 预览安装过程

EOF
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -p|--python)
                PYTHON_VERSION="$2"
                shift 2
                ;;
            -t|--torch)
                PYTORCH_VERSION="$2"
                shift 2
                ;;
            -d|--dtk)
                DTK_VERSION="$2"
                shift 2
                ;;
            -o|--optional)
                INSTALL_OPTIONAL=true
                shift
                ;;
            -c|--conda)
                USE_CONDA=true
                shift
                ;;
            -m|--mirror)
                MIRROR_SITE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 执行命令 (支持dry-run)
execute_cmd() {
    local cmd="$1"
    local description="$2"
    
    if [ "$description" ]; then
        log_info "$description"
    fi
    
    if [ "$DRY_RUN" = "true" ]; then
        echo "    [DRY-RUN] $cmd"
    else
        eval "$cmd"
    fi
}

# 检测操作系统
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
        VER=$VERSION_ID
    else
        log_error "无法检测操作系统"
        exit 1
    fi
    
    log_info "检测到操作系统: $OS $VER"
}

# 设置镜像源
setup_mirrors() {
    if [ "$MIRROR_SITE" = "default" ]; then
        return
    fi
    
    log_info "配置镜像源: $MIRROR_SITE"
    
    case $MIRROR_SITE in
        tsinghua)
            PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
            CONDA_CHANNELS="-c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free"
            ;;
        aliyun)
            PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"
            CONDA_CHANNELS="-c https://mirrors.aliyun.com/anaconda/pkgs/main -c https://mirrors.aliyun.com/anaconda/pkgs/free"
            ;;
        ustc)
            PIP_INDEX_URL="https://pypi.mirrors.ustc.edu.cn/simple"
            CONDA_CHANNELS="-c https://mirrors.ustc.edu.cn/anaconda/pkgs/main -c https://mirrors.ustc.edu.cn/anaconda/pkgs/free"
            ;;
        *)
            log_warning "未知镜像源: $MIRROR_SITE，使用默认源"
            ;;
    esac
    
    if [ "$PIP_INDEX_URL" ]; then
        mkdir -p ~/.pip
        cat > ~/.pip/pip.conf << EOF
[global]
index-url = $PIP_INDEX_URL
trusted-host = $(echo $PIP_INDEX_URL | cut -d'/' -f3)
EOF
    fi
}

# 安装系统依赖
install_system_deps() {
    log_info "安装系统依赖包..."
    
    case $OS in
        *Ubuntu*|*Debian*)
            execute_cmd "sudo apt update" "更新包管理器"
            execute_cmd "sudo apt install -y build-essential cmake git wget curl" "安装基础构建工具"
            execute_cmd "sudo apt install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv" "安装Python"
            execute_cmd "sudo apt install -y python3-pip" "安装pip"
            execute_cmd "sudo apt install -y libffi-dev libssl-dev libnuma-dev" "安装开发库"
            
            # 检查并安装Docker
            if ! command -v docker > /dev/null 2>&1; then
                log_info "安装Docker..."
                execute_cmd "curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh" "安装Docker"
                execute_cmd "sudo usermod -aG docker \$USER" "添加用户到docker组"
            fi
            ;;
            
        *CentOS*|*Red\ Hat*)
            execute_cmd "sudo yum update -y" "更新包管理器"
            execute_cmd "sudo yum groupinstall -y 'Development Tools'" "安装开发工具组"
            execute_cmd "sudo yum install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-devel" "安装Python"
            execute_cmd "sudo yum install -y python3-pip cmake git wget curl" "安装基础工具"
            ;;
            
        *)
            log_warning "未知操作系统，跳过系统依赖安装"
            ;;
    esac
}

# 安装DTK开发工具包
install_dtk() {
    log_info "安装DTK开发工具包..."
    
    # 检查是否已安装
    if command -v dtk-config > /dev/null 2>&1; then
        local current_version=$(dtk-config --version 2>/dev/null || echo "未知")
        log_info "DTK已安装，版本: $current_version"
        return
    fi
    
    # 下载并安装DTK
    DTK_URL="https://developer.sourcefind.cn/downloads/dtk-${DTK_VERSION}.tar.gz"
    
    execute_cmd "wget -O dtk-${DTK_VERSION}.tar.gz ${DTK_URL}" "下载DTK"
    execute_cmd "tar -xzf dtk-${DTK_VERSION}.tar.gz" "解压DTK"
    execute_cmd "cd dtk-${DTK_VERSION} && sudo ./install.sh" "安装DTK"
    execute_cmd "rm -rf dtk-${DTK_VERSION}*" "清理临时文件"
    
    # 设置环境变量
    cat >> ~/.bashrc << 'EOF'
# DTK环境变量
export DTK_ROOT=/opt/dtk
export PATH=$DTK_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$DTK_ROOT/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$DTK_ROOT/python:$PYTHONPATH
EOF
    
    log_success "DTK安装完成，请运行 'source ~/.bashrc' 生效环境变量"
}

# 设置Python环境
setup_python_env() {
    log_info "设置Python环境..."
    
    if [ "$USE_CONDA" = "true" ]; then
        setup_conda_env
    else
        setup_venv
    fi
}

# 设置Conda环境
setup_conda_env() {
    log_info "设置Conda环境..."
    
    # 检查Conda是否已安装
    if ! command -v conda > /dev/null 2>&1; then
        log_info "安装Miniconda..."
        execute_cmd "wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" "下载Miniconda"
        execute_cmd "bash miniconda.sh -b -p \$HOME/miniconda3" "安装Miniconda"
        execute_cmd "rm miniconda.sh" "清理安装文件"
        execute_cmd "source \$HOME/miniconda3/bin/activate" "激活Conda"
    fi
    
    # 创建虚拟环境
    ENV_NAME="dcu-dev"
    execute_cmd "conda create -n $ENV_NAME python=${PYTHON_VERSION} -y $CONDA_CHANNELS" "创建Conda环境"
    execute_cmd "conda activate $ENV_NAME" "激活环境"
    
    log_success "Conda环境 '$ENV_NAME' 创建完成"
}

# 设置Python虚拟环境
setup_venv() {
    log_info "设置Python虚拟环境..."
    
    ENV_DIR="$HOME/dcu-dev-env"
    execute_cmd "python${PYTHON_VERSION} -m venv $ENV_DIR" "创建虚拟环境"
    execute_cmd "source $ENV_DIR/bin/activate" "激活虚拟环境"
    execute_cmd "pip install --upgrade pip setuptools wheel" "升级pip工具"
    
    # 添加激活脚本到bashrc
    echo "# DCU开发环境" >> ~/.bashrc
    echo "alias activate-dcu='source $ENV_DIR/bin/activate'" >> ~/.bashrc
    
    log_success "Python虚拟环境创建完成，使用 'activate-dcu' 命令激活"
}

# 安装PyTorch和相关库
install_pytorch() {
    log_info "安装PyTorch和相关深度学习库..."
    
    # PyTorch ROCm版本
    TORCH_INDEX_URL="https://download.pytorch.org/whl/rocm5.6"
    
    execute_cmd "pip install torch==$PYTORCH_VERSION torchvision torchaudio --index-url $TORCH_INDEX_URL" "安装PyTorch"
    execute_cmd "pip install transformers>=4.35.0" "安装Transformers"
    execute_cmd "pip install accelerate>=0.24.0" "安装Accelerate"
    execute_cmd "pip install datasets>=2.14.0" "安装Datasets"
    execute_cmd "pip install tokenizers>=0.14.0" "安装Tokenizers"
    
    log_success "PyTorch安装完成"
}

# 安装科学计算库
install_scientific_libs() {
    log_info "安装科学计算库..."
    
    execute_cmd "pip install numpy scipy matplotlib" "安装基础科学计算库"
    execute_cmd "pip install pandas jupyter notebook" "安装数据分析工具"
    execute_cmd "pip install scikit-learn seaborn plotly" "安装机器学习和可视化库"
    execute_cmd "pip install numba cupy-cuda11x" "安装GPU加速库"
    
    log_success "科学计算库安装完成"
}

# 安装开发工具
install_dev_tools() {
    log_info "安装开发工具..."
    
    execute_cmd "pip install black flake8 isort mypy" "安装代码格式化工具"
    execute_cmd "pip install pytest pytest-cov" "安装测试框架"
    execute_cmd "pip install pre-commit" "安装Git钩子"
    execute_cmd "pip install ipykernel ipywidgets" "安装Jupyter扩展"
    
    log_success "开发工具安装完成"
}

# 安装可选组件
install_optional_components() {
    if [ "$INSTALL_OPTIONAL" != "true" ]; then
        return
    fi
    
    log_info "安装可选组件..."
    
    # vLLM推理引擎
    execute_cmd "pip install vllm" "安装vLLM"
    
    # Web框架
    execute_cmd "pip install fastapi uvicorn gradio streamlit" "安装Web框架"
    
    # 向量数据库和检索
    execute_cmd "pip install faiss-gpu sentence-transformers" "安装向量检索库"
    
    # 监控和调试
    execute_cmd "pip install wandb tensorboard" "安装实验跟踪工具"
    execute_cmd "pip install psutil gpustat" "安装系统监控工具"
    
    log_success "可选组件安装完成"
}

# 配置Jupyter
setup_jupyter() {
    log_info "配置Jupyter环境..."
    
    # 生成Jupyter配置
    execute_cmd "jupyter notebook --generate-config" "生成Jupyter配置"
    
    # 安装内核
    execute_cmd "python -m ipykernel install --user --name dcu-dev --display-name 'DCU Development'" "安装Jupyter内核"
    
    # 安装扩展
    execute_cmd "pip install jupyter_contrib_nbextensions" "安装Jupyter扩展"
    execute_cmd "jupyter contrib nbextension install --user" "启用扩展"
    
    log_success "Jupyter配置完成"
}

# 验证安装
verify_installation() {
    log_info "验证安装..."
    
    # 创建测试脚本
    cat > /tmp/dcu_test.py << 'EOF'
import sys
import torch
import numpy as np
import transformers

print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"Transformers版本: {transformers.__version__}")
print(f"NumPy版本: {np.__version__}")

print(f"DCU可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"DCU数量: {torch.cuda.device_count()}")
    print(f"DCU名称: {torch.cuda.get_device_name(0)}")
    
    # 简单计算测试
    a = torch.randn(100, 100, device='cuda')
    b = torch.randn(100, 100, device='cuda')
    c = torch.mm(a, b)
    print("DCU矩阵计算测试: 通过")
else:
    print("DCU不可用，请检查驱动和配置")
EOF
    
    execute_cmd "python /tmp/dcu_test.py" "运行验证测试"
    execute_cmd "rm /tmp/dcu_test.py" "清理测试文件"
}

# 生成使用指南
generate_usage_guide() {
    log_info "生成使用指南..."
    
    GUIDE_FILE="DCU_Development_Guide.md"
    
    cat > "$GUIDE_FILE" << 'EOF'
# 海光DCU开发环境使用指南

## 环境激活

### 虚拟环境
```bash
# 激活DCU开发环境
activate-dcu

# 或者
source ~/dcu-dev-env/bin/activate
```

### Conda环境
```bash
conda activate dcu-dev
```

## 基础使用

### 1. 验证环境
```bash
python -c "import torch; print(f'DCU可用: {torch.cuda.is_available()}')"
```

### 2. 运行示例
```bash
cd examples/llm-inference
python simple_test.py
```

### 3. 启动Jupyter
```bash
jupyter notebook
```

## 常用命令

### 检查DCU状态
```bash
hy-smi
```

### 监控资源使用
```bash
watch -n 1 hy-smi
```

### 安装额外包
```bash
pip install package_name
```

## 故障排除

### DCU不可用
1. 检查驱动: `hy-smi`
2. 检查权限: `ls -la /dev/dri/`
3. 重启系统

### 内存不足
1. 减少批次大小
2. 使用混合精度
3. 启用梯度检查点

### 包冲突
1. 重新创建环境
2. 使用pip freeze检查版本
3. 降级冲突包

## 更多资源

- 官方文档: https://developer.sourcefind.cn/
- 社区论坛: https://bbs.sourcefind.cn/
- GitHub: https://github.com/hygon-technologies/
EOF
    
    log_success "使用指南已生成: $GUIDE_FILE"
}

# 主安装流程
main() {
    cat << 'EOF'
================================================================
                海光DCU开发环境安装程序
================================================================
EOF
    
    log_info "开始安装DCU开发环境..."
    log_info "配置参数:"
    log_info "  Python版本: $PYTHON_VERSION"
    log_info "  PyTorch版本: $PYTORCH_VERSION"
    log_info "  DTK版本: $DTK_VERSION"
    log_info "  安装可选组件: $INSTALL_OPTIONAL"
    log_info "  使用Conda: $USE_CONDA"
    log_info "  镜像源: $MIRROR_SITE"
    
    if [ "$DRY_RUN" = "true" ]; then
        log_warning "这是预览模式，不会实际安装任何软件"
    fi
    
    echo ""
    read -p "是否继续安装? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "安装已取消"
        exit 0
    fi
    
    # 执行安装步骤
    detect_os
    setup_mirrors
    install_system_deps
    install_dtk
    setup_python_env
    install_pytorch
    install_scientific_libs
    install_dev_tools
    install_optional_components
    setup_jupyter
    verify_installation
    generate_usage_guide
    
    log_success "🎉 DCU开发环境安装完成!"
    log_info "请运行以下命令生效环境变量:"
    echo "  source ~/.bashrc"
    log_info "然后激活开发环境:"
    if [ "$USE_CONDA" = "true" ]; then
        echo "  conda activate dcu-dev"
    else
        echo "  activate-dcu"
    fi
    log_info "查看使用指南: cat DCU_Development_Guide.md"
}

# 检查权限
check_permissions() {
    if [ "$EUID" -eq 0 ]; then
        log_error "请不要以root权限运行此脚本"
        exit 1
    fi
}

# 程序入口
parse_args "$@"
check_permissions
main 