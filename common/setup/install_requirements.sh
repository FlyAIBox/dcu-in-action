#!/bin/bash

# DCU-in-Action 智能依赖安装脚本
# 支持不同场景的依赖安装和环境检查

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
DCU-in-Action 依赖安装脚本

用法: $0 [选项]

选项:
    -m, --mode MODE         安装模式 (minimal|standard|full|docker)
    -c, --check             仅检查环境，不安装依赖
    -f, --force             强制重新安装所有依赖
    -v, --verbose           详细输出
    -h, --help              显示此帮助信息

安装模式说明:
    minimal     最小依赖安装 (仅核心功能)
    standard    标准安装 (推荐，包含大部分功能)
    full        完整安装 (所有功能模块)
    docker      Docker环境安装

示例:
    $0 --mode standard                # 标准安装
    $0 --check                       # 仅检查环境
    $0 --mode full --verbose         # 完整安装并显示详细信息

EOF
}

# 检查Python环境
check_python() {
    log_info "检查Python环境..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        return 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    log_info "Python版本: $PYTHON_VERSION"
    
    # 检查Python版本 (>= 3.8)
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        log_success "Python版本检查通过"
    else
        log_error "Python版本过低，需要3.8+，当前版本: $PYTHON_VERSION"
        return 1
    fi
    
    return 0
}

# 检查pip环境
check_pip() {
    log_info "检查pip环境..."
    
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 未安装"
        return 1
    fi
    
    PIP_VERSION=$(pip3 --version | cut -d' ' -f2)
    log_info "pip版本: $PIP_VERSION"
    
    log_success "pip环境检查通过"
    return 0
}

# 检查DCU环境（可选）
check_dcu() {
    log_info "检查DCU环境..."
    
    # 检查DCU设备文件
    if [ -e /dev/kfd ] && [ -e /dev/dri ]; then
        log_success "检测到DCU设备文件"
        DCU_AVAILABLE=true
    else
        log_warning "未检测到DCU设备文件，将在模拟模式下运行"
        DCU_AVAILABLE=false
    fi
    
    # 检查ROCm环境
    if command -v rocm-smi &> /dev/null; then
        log_success "检测到ROCm环境"
        if [ "$VERBOSE" = true ]; then
            rocm-smi --showdriverversion
        fi
    else
        log_warning "未检测到ROCm环境"
    fi
    
    return 0
}

# 检查系统资源
check_system() {
    log_info "检查系统资源..."
    
    # 检查内存
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    log_info "系统内存: ${TOTAL_MEM}GB"
    
    if [ "$TOTAL_MEM" -lt 8 ]; then
        log_warning "内存不足8GB，某些功能可能受限"
    else
        log_success "内存检查通过"
    fi
    
    # 检查磁盘空间
    DISK_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    log_info "可用磁盘空间: ${DISK_SPACE}GB"
    
    if [ "$DISK_SPACE" -lt 10 ]; then
        log_warning "磁盘空间不足10GB，可能影响依赖安装"
    else
        log_success "磁盘空间检查通过"
    fi
    
    return 0
}

# 环境检查主函数
check_environment() {
    log_info "开始环境检查..."
    
    local checks_passed=0
    local total_checks=4
    
    check_python && ((checks_passed++)) || true
    check_pip && ((checks_passed++)) || true
    check_dcu && ((checks_passed++)) || true
    check_system && ((checks_passed++)) || true
    
    log_info "环境检查完成: $checks_passed/$total_checks 项检查通过"
    
    if [ "$checks_passed" -ge 2 ]; then
        log_success "环境检查基本满足要求"
        return 0
    else
        log_error "环境检查失败，请解决上述问题后重试"
        return 1
    fi
}

# 安装DCU特定依赖
install_dcu_packages() {
    log_info "安装DCU特定依赖包..."
    
    # DCU特定包的下载URL（示例）
    DCU_PACKAGES=(
        "torch==2.4.1+das.opt2.dtk2504"
        "lmslim==0.2.1+das.dtk2504"
        "flash-attn==2.6.1+das.opt4.dtk2504"
        "vllm==0.6.2+das.opt3.dtk2504"
        "deepspeed==0.14.2+das.opt2.dtk2504"
    )
    
    log_warning "DCU特定包需要从官方下载页面获取:"
    log_warning "https://das.sourcefind.cn:55011/portal/#/home"
    log_warning "请确保已下载对应的wheel文件到当前目录"
    
    # 检查是否有DCU相关的wheel文件
    DCU_WHEELS=$(find . -name "*das*.whl" 2>/dev/null | wc -l)
    if [ "$DCU_WHEELS" -gt 0 ]; then
        log_info "检测到 $DCU_WHEELS 个DCU wheel文件"
        pip3 install *.whl || log_warning "部分DCU包安装失败"
    else
        log_warning "未检测到DCU wheel文件，跳过DCU特定包安装"
    fi
}

# 根据模式安装依赖
install_dependencies() {
    local mode=$1
    
    log_info "开始安装依赖 (模式: $mode)..."
    
    # 升级pip
    log_info "升级pip..."
    pip3 install --upgrade pip
    
    case $mode in
        "standard")
            log_info "安装标准依赖..."
            pip3 install -r requirements.txt
            install_dcu_packages
            ;;
        "full")
            log_info "安装完整依赖..."
            pip3 install -r requirements-full.txt
            install_dcu_packages
            ;;
        *)
            log_error "未知的安装模式: $mode"
            return 1
            ;;
    esac
    
    log_success "依赖安装完成"
}

# 安装后验证
post_install_verification() {
    log_info "进行安装后验证..."
    
    # 检查关键包是否安装成功
    CRITICAL_PACKAGES=("torch" "transformers" "numpy" "fastapi")
    
    for package in "${CRITICAL_PACKAGES[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            log_success "$package 安装成功"
        else
            log_warning "$package 安装可能有问题"
        fi
    done
    
    # 运行基础测试
    if [ -f "examples/basic/test_dcu_manager.py" ]; then
        log_info "运行基础功能测试..."
        if python3 examples/basic/test_dcu_manager.py; then
            log_success "基础功能测试通过"
        else
            log_warning "基础功能测试失败"
        fi
    fi
}

# 主函数
main() {
    local mode="standard"
    local check_only=false
    local force_install=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--mode)
                mode="$2"
                shift 2
                ;;
            -c|--check)
                check_only=true
                shift
                ;;
            -f|--force)
                force_install=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 显示欢迎信息
    echo "========================================"
    echo "DCU-in-Action 依赖安装脚本"
    echo "========================================"
    
    # 环境检查
    if ! check_environment; then
        log_error "环境检查失败，退出安装"
        exit 1
    fi
    
    # 如果仅检查环境，则退出
    if [ "$check_only" = true ]; then
        log_success "环境检查完成"
        exit 0
    fi
    
    # 确认安装模式
    log_info "安装模式: $mode"
    if [ "$mode" = "full" ]; then
        log_warning "完整安装将下载大量依赖包，可能需要较长时间"
        read -p "是否继续? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "安装已取消"
            exit 0
        fi
    fi
    
    # 安装依赖
    if install_dependencies "$mode"; then
        post_install_verification
        log_success "所有安装步骤完成!"
        echo
        echo "接下来可以："
        echo "1. 运行 'python examples/basic/test_dcu_manager.py' 测试基础功能"
        echo "2. 查看 'examples/' 目录了解更多使用方法"
        echo "3. 访问项目文档获取详细指南"
    else
        log_error "安装过程中出现错误"
        exit 1
    fi
}

# 脚本入口
main "$@" 