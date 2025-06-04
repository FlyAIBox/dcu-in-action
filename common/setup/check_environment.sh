#!/bin/bash

# 海光DCU环境检查脚本
# 检查硬件、驱动、软件环境是否正确配置

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

# 检查结果统计
TOTAL_CHECKS=0
PASSED_CHECKS=0

check_item() {
    local description="$1"
    local command="$2"
    local required="$3"  # true or false
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    log_info "检查: $description"
    
    if eval "$command" > /dev/null 2>&1; then
        log_success "$description - 通过"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        if [ "$required" = "true" ]; then
            log_error "$description - 失败 (必需)"
        else
            log_warning "$description - 失败 (可选)"
        fi
        return 1
    fi
}

print_header() {
    echo "================================================================"
    echo "           海光DCU环境检查工具 v1.0"
    echo "================================================================"
    echo ""
}

print_summary() {
    echo ""
    echo "================================================================"
    echo "                    检查结果总结"
    echo "================================================================"
    echo "总检查项目: $TOTAL_CHECKS"
    echo "通过项目: $PASSED_CHECKS"
    echo "通过率: $((PASSED_CHECKS * 100 / TOTAL_CHECKS))%"
    echo ""
    
    if [ $PASSED_CHECKS -eq $TOTAL_CHECKS ]; then
        log_success "🎉 所有检查项目都通过了！DCU环境配置正确。"
    else
        log_warning "⚠️  部分检查项目未通过，请根据上述提示进行修复。"
    fi
    echo "================================================================"
}

# 1. 硬件检查
check_hardware() {
    log_info "开始硬件检查..."
    
    # 检查DCU设备
    check_item "DCU设备存在" "lspci | grep -i 'DCU\|Hygon\|Haiguang\|Advanced Micro Devices'" true
    
    # 检查内存
    check_item "系统内存充足 (>= 16GB)" "[ \$(free -g | awk '/^Mem:/{print \$2}') -ge 16 ]" true
    
    # 检查磁盘空间
    check_item "磁盘空间充足 (>= 100GB)" "[ \$(df -BG / | tail -1 | awk '{print \$4}' | sed 's/G//') -ge 100 ]" true
    
    echo ""
}

# 2. 驱动检查
check_drivers() {
    log_info "开始驱动检查..."
    
    # 检查DCU驱动
    check_item "DCU驱动已安装" "command -v hy-smi" true
    
    # 检查驱动版本
    if command -v hy-smi > /dev/null 2>&1; then
        DRIVER_VERSION=$(hy-smi --version 2>/dev/null | head -1 || echo "未知")
        log_info "DCU驱动版本: $DRIVER_VERSION"
    fi
    
    # 检查内核模块
    check_item "DCU内核模块已加载" "lsmod | grep -E 'amdgpu|dcu|hycu'" false
    
    echo ""
}

# 3. 软件环境检查
check_software() {
    log_info "开始软件环境检查..."
    
    # 检查Python
    check_item "Python 3.8+ 已安装" "python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'" true
    
    if command -v python3 > /dev/null 2>&1; then
        PYTHON_VERSION=$(python3 --version 2>&1)
        log_info "Python版本: $PYTHON_VERSION"
    fi
    
    # 检查pip
    check_item "pip 已安装" "command -v pip3" true
    
    # 检查DTK
    check_item "DTK 已安装" "command -v hy-smi" true
    
    # if command -v hy-smi > /dev/null 2>&1; then
    #     DTK_VERSION=$(dtk-config --version 2>/dev/null || echo "未知")
    #     log_info "DTK版本: $DTK_VERSION"
    # fi

    if command -v hy-smi > /dev/null 2>&1; then # 检查DTK环境是否可能已激活
        DTK_VERSION="未知" # 默认值

        # 1. 检查 ROCM_PATH 是否已设置
        if [ -z "$ROCM_PATH" ]; then
            DTK_VERSION="未知 (ROCM_PATH 环境变量未设置)"
        # 2. 检查版本文件是否存在且可读
        elif [ ! -r "$ROCM_PATH/.dtk_version" ]; then
            DTK_VERSION="未知 (版本文件 '$ROCM_PATH/.dtk_version' 不存在或不可读)"
        else
            # 3. 从文件中提取以 "DTK-" 开头的行
            # 使用 grep 查找以 "DTK-" 开头的行，并用 head -n 1确保只取第一行（以防万一有多行匹配）
            version_line_from_file=$(grep '^DTK-' "$ROCM_PATH/.dtk_version" | head -n 1)

            if [ -n "$version_line_from_file" ]; then
                DTK_VERSION="$version_line_from_file" # 整行 "DTK-25.04"
                # 如果您只需要 "25.04" 部分，可以取消下面一行的注释并注释掉上面一行：
                # DTK_VERSION="${version_line_from_file#DTK-}"
            else
                DTK_VERSION="未知 (在 '$ROCM_PATH/.dtk_version' 中未找到以 'DTK-' 开头的行)"
            fi
        fi
        log_info "DTK版本: $DTK_VERSION"
    else
        # 如果 hy-smi 命令不存在
        log_info "DTK环境: hy-smi 命令未找到, DTK版本检查跳过。"
        # 如果需要在这种情况下也明确设置 DTK_VERSION，可以取消下面一行的注释
        # DTK_VERSION="未知 (hy-smi 未找到)"
    fi    
    
    # 检查Docker
    check_item "Docker 已安装" "command -v docker" false
    
    if command -v docker > /dev/null 2>&1; then
        DOCKER_VERSION=$(docker --version 2>/dev/null | cut -d' ' -f3 | cut -d',' -f1 || echo "未知")
        log_info "Docker版本: $DOCKER_VERSION"
    fi
    
    echo ""
}

# 4. Python环境检查
check_python_env() {
    log_info "开始Python环境检查..."
    
    # 检查PyTorch
    check_item "PyTorch 已安装" "python3 -c 'import torch'" true
    
    if python3 -c 'import torch' > /dev/null 2>&1; then
        TORCH_VERSION=$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "未知")
        log_info "PyTorch版本: $TORCH_VERSION"
        
        # 检查CUDA支持
        check_item "PyTorch CUDA支持" "python3 -c 'import torch; exit(0 if torch.cuda.is_available() else 1)'" true
        
        if python3 -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' > /dev/null 2>&1; then
            DCU_COUNT=$(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo "0")
            log_info "检测到DCU数量: $DCU_COUNT"
        fi
    fi
    
    # 检查其他重要库
    check_item "NumPy 已安装" "python3 -c 'import numpy'" true
    check_item "transformers 已安装" "python3 -c 'import transformers'" false
    check_item "accelerate 已安装" "python3 -c 'import accelerate'" false
    
    echo ""
}

# 5. 网络和权限检查
check_network_permissions() {
    log_info "开始网络和权限检查..."
    
    # 检查网络连接
    check_item "网络连接正常" "ping -c 1 baidu.com" false
    check_item "Hugging Face代理可访问" "curl -s --connect-timeout 5 https://hf-mirror.com/ > /dev/null" false
    
    # 检查DCU设备权限
    if [ -e /dev/dri/card0 ]; then
        check_item "DCU设备权限" "[ -r /dev/dri/card0 ] && [ -w /dev/dri/card0 ]" true
    fi
    
    # 检查用户组
    check_item "用户在render组" "groups | grep render" false
    check_item "用户在video组" "groups | grep video" false
    
    echo ""
}

# 6. 性能测试
check_performance() {
    log_info "开始基础性能测试..."
    
    # DCU基础测试
    if python3 -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' > /dev/null 2>&1; then
        log_info "执行DCU基础计算测试..."
        
        PERF_TEST_RESULT=$(python3 -c "
import torch
import time
try:
    device = torch.device('cuda')
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    
    start_time = time.time()
    c = torch.mm(a, b)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f'{end_time - start_time:.3f}')
except Exception as e:
    print('ERROR')
" 2>/dev/null)
        
        if [ "$PERF_TEST_RESULT" != "ERROR" ] && [ "$PERF_TEST_RESULT" != "" ]; then
            log_success "DCU矩阵乘法测试通过 (耗时: ${PERF_TEST_RESULT}s)"
            PASSED_CHECKS=$((PASSED_CHECKS + 1))
        else
            log_error "DCU性能测试失败"
        fi
        TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    else
        log_warning "跳过性能测试 (DCU不可用)"
    fi
    
    echo ""
}

# 生成详细报告
generate_report() {
    log_info "生成详细系统报告..."
    
    REPORT_FILE="dcu_environment_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "海光DCU环境检查报告"
        echo "====================="
        echo "生成时间: $(date)"
        echo "系统信息: $(uname -a)"
        echo ""
        
        echo "硬件信息:"
        echo "--------"
        lscpu | grep -E "(Model name|CPU\(s\)|Architecture)" || echo "CPU信息获取失败"
        free -h || echo "内存信息获取失败"
        df -h / || echo "磁盘信息获取失败"
        lspci | grep -i VGA || echo "显卡信息获取失败"
        echo ""
        
        echo "软件版本:"
        echo "--------"
        echo "操作系统: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2 || echo '未知')"
        echo "内核版本: $(uname -r)"
        echo "Python: $(python3 --version 2>&1 || echo '未安装')"
        echo "DTK: $(grep '^DTK-' "$ROCM_PATH/.dtk_version" | head -n 1 || echo '未安装')"
        echo "Docker: $(docker --version 2>/dev/null || echo '未安装')"
        echo ""
        
        echo "DCU信息:"
        echo "-------"
        hy-smi 2>/dev/null || echo "DCU设备信息获取失败"
        echo ""
        
        echo "Python包信息:"
        echo "------------"
        python3 -c "
import pkg_resources
packages = ['torch', 'numpy', 'transformers', 'accelerate', 'vllm']
for pkg in packages:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f'{pkg}: {version}')
    except:
        print(f'{pkg}: 未安装')
" 2>/dev/null || echo "Python包信息获取失败"
        
    } > "$REPORT_FILE"
    
    log_success "详细报告已保存到: $REPORT_FILE"
}

# 提供修复建议
provide_suggestions() {
    echo ""
    log_info "修复建议:"
    echo "========"
    
    if ! command -v hy-smi > /dev/null 2>&1; then
        echo "• DCU驱动未安装:"
        echo "  参考DCU开发社区安装: https://developer.sourcefind.cn/"
        echo ""
    fi
    
    if ! python3 -c 'import torch' > /dev/null 2>&1; then
        echo "• PyTorch未安装:"
        echo "  参考DCU开发社区安装: https://developer.sourcefind.cn/"
        echo ""
    fi
    
    if ! python3 -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' > /dev/null 2>&1; then
        echo "• DCU在PyTorch中不可用:"
        echo "  检查环境变量: export HIP_VISIBLE_DEVICES=0"
        echo "  重启系统或重新加载驱动模块"
        echo ""
    fi
    
    if ! groups | grep render > /dev/null 2>&1; then
        echo "• 用户权限问题:"
        echo "  sudo usermod -a -G render,video \$USER"
        echo "  注销并重新登录"
        echo ""
    fi
    
    echo "• 获取更多帮助:"
    echo "  官方文档: https://developer.sourcefind.cn/"
}

# 主函数
main() {
    print_header
    
    # 执行检查
    check_hardware
    check_drivers
    check_software
    check_python_env
    check_network_permissions
    check_performance
    
    # 生成报告
    generate_report
    
    # 显示总结
    print_summary
    
    # 提供建议
    if [ $PASSED_CHECKS -lt $TOTAL_CHECKS ]; then
        provide_suggestions
    fi
}

# 检查是否以root权限运行
if [ "$EUID" -eq 0 ]; then
    log_warning "检测到以root权限运行，某些检查可能不准确"
fi

# 运行主程序
main "$@" 