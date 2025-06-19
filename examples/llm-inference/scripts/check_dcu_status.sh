#!/bin/bash

# 海光DCU K100-AI硬件状态检查脚本
# 版本: v2.0

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

log_header() {
    echo -e "${CYAN}===== $1 =====${NC}"
}

# 检查ROCm安装
check_rocm_installation() {
    log_header "ROCm环境检查"
    
    if ! command -v rocm-smi &> /dev/null; then
        log_error "rocm-smi未找到，请安装DCU驱动"
        return 1
    fi
    
    log_success "rocm-smi已安装"
    
    # 检查ROCm版本
    if command -v rocminfo &> /dev/null; then
        ROCM_VERSION=$(rocminfo | grep "Agent" | head -1 | awk '{print $2}' || echo "未知")
        log_info "ROCm版本: $ROCM_VERSION"
    fi
    
    return 0
}

# 检查DCU设备信息
check_dcu_devices() {
    log_header "DCU设备信息"
    
    # 设备数量
    DCU_COUNT=$(rocm-smi --showid 2>/dev/null | grep -c "GPU\[" || echo "0")
    log_info "检测到 $DCU_COUNT 个DCU设备"
    
    if [[ $DCU_COUNT -eq 0 ]]; then
        log_error "未检测到DCU设备"
        return 1
    fi
    
    # 详细设备信息
    log_info "设备详细信息:"
    rocm-smi --showid --showproductname --showvbios
    
    return 0
}

# 检查设备状态
check_device_status() {
    log_header "设备状态检查"
    
    # 温度检查
    log_info "设备温度:"
    rocm-smi --showtemp
    
    # 功耗检查
    log_info "设备功耗:"
    rocm-smi --showpower
    
    # 内存信息
    log_info "显存使用情况:"
    rocm-smi --showuse --showmeminfo
    
    # 时钟频率
    log_info "时钟频率:"
    rocm-smi --showclocks
    
    return 0
}

# 检查PCIe信息
check_pcie_info() {
    log_header "PCIe信息检查"
    
    # 查找DCU设备的PCIe信息
    DCU_DEVICES=$(lspci | grep -i "display\|vga\|3d" | grep -i "advanced micro devices" || echo "")
    
    if [[ -z "$DCU_DEVICES" ]]; then
        log_warning "未在lspci中找到DCU设备"
        return 1
    fi
    
    log_info "PCIe设备列表:"
    echo "$DCU_DEVICES"
    
    # 检查PCIe带宽
    log_info "PCIe链路状态:"
    for device in $(lspci | grep -i "display\|vga\|3d" | grep -i "advanced micro devices" | cut -d' ' -f1); do
        echo "设备 $device:"
        lspci -vvv -s $device | grep -E "LnkCap|LnkSta" | head -2
    done
    
    return 0
}

# 检查系统资源
check_system_resources() {
    log_header "系统资源检查"
    
    # CPU信息
    log_info "CPU信息:"
    echo "CPU核心数: $(nproc)"
    echo "CPU型号: $(lscpu | grep "Model name" | awk -F: '{print $2}' | xargs)"
    
    # 内存信息
    log_info "内存信息:"
    free -h
    
    # NUMA拓扑
    if command -v numactl &> /dev/null; then
        log_info "NUMA拓扑:"
        numactl --hardware | head -10
    fi
    
    # 磁盘空间
    log_info "磁盘空间:"
    df -h / /tmp
    
    return 0
}

# 检查网络配置
check_network() {
    log_header "网络配置检查"
    
    # 网络接口
    log_info "网络接口:"
    ip addr show | grep -E "^[0-9]+:|inet " | head -10
    
    # 网络连通性测试
    log_info "网络连通性测试:"
    if ping -c 3 8.8.8.8 &> /dev/null; then
        log_success "外网连接正常"
    else
        log_warning "外网连接异常"
    fi
    
    return 0
}

# 检查Python环境
check_python_environments() {
    log_header "Python环境检查"
    
    # 检查conda
    if command -v conda &> /dev/null; then
        log_success "Conda已安装"
        log_info "Conda版本: $(conda --version)"
        
        # 列出环境
        log_info "Conda环境列表:"
        conda env list
    else
        log_warning "Conda未安装"
    fi
    
    # 检查Python版本
    log_info "系统Python版本: $(python3 --version)"
    
    return 0
}

# 性能基准测试
run_basic_benchmark() {
    log_header "基础性能测试"
    
    if ! command -v rocm-smi &> /dev/null; then
        log_error "无法运行性能测试，rocm-smi未找到"
        return 1
    fi
    
    log_info "运行基础GPU计算测试..."
    
    # 创建临时测试脚本
    cat > /tmp/gpu_test.py << 'EOF'
import time
try:
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"可用GPU数量: {device_count}")
        
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # 简单计算测试
        device = torch.device('cuda:0')
        print(f"测试设备: {device}")
        
        # 矩阵乘法测试
        size = 1024
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        start_time = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end_time = time.time()
        
        print(f"矩阵乘法 ({size}x{size}) 耗时: {end_time - start_time:.4f} 秒")
        print("基础GPU测试完成")
    else:
        print("CUDA/ROCm不可用")
        
except ImportError:
    print("PyTorch未安装，跳过GPU测试")
except Exception as e:
    print(f"GPU测试出错: {e}")
EOF
    
    # 尝试在不同环境中运行
    for env in vllm sglang xinference; do
        if conda env list | grep -q $env; then
            log_info "在 $env 环境中测试..."
            source ~/miniconda3/bin/activate $env 2>/dev/null && python /tmp/gpu_test.py && conda deactivate || log_warning "$env 环境测试失败"
        fi
    done
    
    # 清理
    rm -f /tmp/gpu_test.py
    
    return 0
}

# 生成系统报告
generate_system_report() {
    log_header "生成系统报告"
    
    REPORT_FILE="/tmp/dcu_system_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "===== 海光DCU K100-AI系统报告 ====="
        echo "生成时间: $(date)"
        echo "主机名: $(hostname)"
        echo "操作系统: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
        echo "内核版本: $(uname -r)"
        echo ""
        
        echo "===== DCU设备信息 ====="
        rocm-smi --showid --showproductname --showvbios 2>/dev/null || echo "无法获取DCU信息"
        echo ""
        
        echo "===== 硬件配置 ====="
        echo "CPU: $(lscpu | grep "Model name" | awk -F: '{print $2}' | xargs)"
        echo "CPU核心数: $(nproc)"
        echo "内存: $(free -h | grep Mem | awk '{print $2}')"
        echo ""
        
        echo "===== 当前设备状态 ====="
        rocm-smi --showtemp --showpower --showuse 2>/dev/null || echo "无法获取设备状态"
        echo ""
        
        echo "===== 软件环境 ====="
        echo "Python: $(python3 --version 2>/dev/null || echo "未安装")"
        echo "Conda: $(conda --version 2>/dev/null || echo "未安装")"
        echo ""
        
        if command -v conda &> /dev/null; then
            echo "Conda环境:"
            conda env list 2>/dev/null || echo "无法获取环境列表"
        fi
        
    } > "$REPORT_FILE"
    
    log_success "系统报告已生成: $REPORT_FILE"
    
    return 0
}

# 主函数
main() {
    echo -e "${CYAN}"
    echo "================================================================="
    echo "          海光DCU K100-AI硬件状态检查工具 v2.0"
    echo "================================================================="
    echo -e "${NC}"
    
    local exit_code=0
    
    # 依次执行检查
    check_rocm_installation || exit_code=1
    echo
    
    check_dcu_devices || exit_code=1
    echo
    
    check_device_status
    echo
    
    check_pcie_info
    echo
    
    check_system_resources
    echo
    
    check_network
    echo
    
    check_python_environments
    echo
    
    # 询问是否运行性能测试
    read -p "是否运行基础性能测试? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_basic_benchmark
        echo
    fi
    
    # 询问是否生成报告
    read -p "是否生成系统报告? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        generate_system_report
    fi
    
    echo -e "${CYAN}=================================================================${NC}"
    if [[ $exit_code -eq 0 ]]; then
        log_success "所有检查完成，系统状态正常"
    else
        log_warning "检查完成，但发现一些问题，请查看上述输出"
    fi
    echo -e "${CYAN}=================================================================${NC}"
    
    return $exit_code
}

# 执行主函数
main "$@" 