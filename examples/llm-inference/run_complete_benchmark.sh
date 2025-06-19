#!/bin/bash

# 海光DCU K100-AI大模型推理完整基准测试脚本
# 版本: v2.0
# 作者: DCU-in-Action Team

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
    echo -e "${CYAN}$1${NC}"
}

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认参数
MODELS="deepseek-7b,qwen-7b"
FRAMEWORKS="vllm,sglang"
TEST_SINGLE_GPU=true
TEST_MULTI_GPU=true
GPU_COUNTS="1,8"
TEST_SCENARIO="standard_test"
CONFIG_FILE="$SCRIPT_DIR/configs/benchmark_config.yaml"
RESULTS_DIR="$SCRIPT_DIR/results"
LOGS_DIR="$SCRIPT_DIR/logs"
SKIP_SETUP=false
GENERATE_REPORT=true
CLEANUP_AFTER=true

# 解析命令行参数
usage() {
    cat << EOF
海光DCU K100-AI大模型推理完整基准测试工具 v2.0

用法: $0 [选项]

选项:
  -m, --models MODELS           测试模型列表，逗号分隔 (默认: deepseek-7b,qwen-7b)
  -f, --frameworks FRAMEWORKS  推理框架列表，逗号分隔 (默认: vllm,sglang)
  -g, --gpu-counts GPU_COUNTS   GPU数量列表，逗号分隔 (默认: 1,8)
  -s, --scenario SCENARIO       测试场景 (默认: standard_test)
      --single-gpu-only         仅测试单卡
      --multi-gpu-only          仅测试多卡
      --skip-setup              跳过环境设置
      --no-report               不生成报告
      --no-cleanup              测试后不清理
  -c, --config CONFIG_FILE      配置文件路径
  -r, --results RESULTS_DIR     结果输出目录
  -l, --logs LOGS_DIR           日志输出目录
  -h, --help                    显示帮助信息

测试场景:
  - quick_test         快速测试 (开发调试用)
  - standard_test      标准测试 (常规评估用)
  - comprehensive_test 深度测试 (详细分析用)
  - stress_test        压力测试 (性能上限探测)

示例:
  # 运行标准测试
  $0
  
  # 仅测试DeepSeek模型的vLLM框架
  $0 -m deepseek-7b -f vllm
  
  # 运行快速测试
  $0 -s quick_test
  
  # 仅测试单卡
  $0 --single-gpu-only
  
  # 仅测试多卡
  $0 --multi-gpu-only -g 4,8

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--models)
            MODELS="$2"
            shift 2
            ;;
        -f|--frameworks)
            FRAMEWORKS="$2"
            shift 2
            ;;
        -g|--gpu-counts)
            GPU_COUNTS="$2"
            shift 2
            ;;
        -s|--scenario)
            TEST_SCENARIO="$2"
            shift 2
            ;;
        --single-gpu-only)
            TEST_SINGLE_GPU=true
            TEST_MULTI_GPU=false
            shift
            ;;
        --multi-gpu-only)
            TEST_SINGLE_GPU=false
            TEST_MULTI_GPU=true
            shift
            ;;
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --no-report)
            GENERATE_REPORT=false
            shift
            ;;
        --no-cleanup)
            CLEANUP_AFTER=false
            shift
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -r|--results)
            RESULTS_DIR="$2"
            shift 2
            ;;
        -l|--logs)
            LOGS_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            usage
            exit 1
            ;;
    esac
done

# 创建目录
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# 全局变量
TOTAL_TESTS=0
SUCCESSFUL_TESTS=0
FAILED_TESTS=0
START_TIME=$(date +%s)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 显示测试配置
show_test_config() {
    log_header "=========================================="
    log_header "  海光DCU K100-AI大模型推理基准测试系统"
    log_header "            版本: v2.0"
    log_header "=========================================="
    echo
    log_info "测试配置:"
    log_info "  模型: $MODELS"
    log_info "  框架: $FRAMEWORKS"
    log_info "  GPU数量: $GPU_COUNTS"
    log_info "  测试场景: $TEST_SCENARIO"
    log_info "  单卡测试: $([ "$TEST_SINGLE_GPU" = true ] && echo "是" || echo "否")"
    log_info "  多卡测试: $([ "$TEST_MULTI_GPU" = true ] && echo "是" || echo "否")"
    log_info "  配置文件: $CONFIG_FILE"
    log_info "  结果目录: $RESULTS_DIR"
    log_info "  日志目录: $LOGS_DIR"
    echo
}

# 环境检查
check_environment() {
    log_info "检查测试环境..."
    
    # 检查必要的命令
    local required_commands=("rocm-smi" "conda" "python3")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "$cmd 命令未找到"
            return 1
        fi
    done
    
    # 检查DCU设备
    local gpu_count=$(rocm-smi --showid 2>/dev/null | grep -c "GPU\[" || echo "0")
    if [[ $gpu_count -eq 0 ]]; then
        log_error "未检测到DCU设备"
        return 1
    fi
    
    log_success "检测到 $gpu_count 个DCU设备"
    
    # 检查配置文件
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "配置文件不存在: $CONFIG_FILE"
        return 1
    fi
    
    log_success "环境检查通过"
    return 0
}

# 环境设置
setup_environment() {
    if [[ "$SKIP_SETUP" = true ]]; then
        log_info "跳过环境设置"
        return 0
    fi
    
    log_info "设置测试环境..."
    
    if [[ -f "$SCRIPT_DIR/scripts/setup_environment.sh" ]]; then
        bash "$SCRIPT_DIR/scripts/setup_environment.sh"
        if [[ $? -ne 0 ]]; then
            log_error "环境设置失败"
            return 1
        fi
    else
        log_warning "环境设置脚本不存在，跳过自动设置"
    fi
    
    log_success "环境设置完成"
    return 0
}

# 运行单个测试
run_single_test() {
    local model="$1"
    local framework="$2"
    local gpu_count="$3"
    
    log_info "开始测试: $model + $framework + ${gpu_count}GPU"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    local test_log="$LOGS_DIR/test_${model}_${framework}_${gpu_count}gpu_${TIMESTAMP}.log"
    
    if [[ $gpu_count -eq 1 ]]; then
        # 单卡测试
        if [[ "$TEST_SINGLE_GPU" = true ]]; then
            bash "$SCRIPT_DIR/scripts/run_single_gpu_benchmark.sh" \
                -m "$model" \
                -f "$framework" \
                -s "$TEST_SCENARIO" \
                -c "$CONFIG_FILE" \
                -r "$RESULTS_DIR" \
                -l "$LOGS_DIR" > "$test_log" 2>&1
        else
            log_info "跳过单卡测试: $model + $framework"
            return 0
        fi
    else
        # 多卡测试
        if [[ "$TEST_MULTI_GPU" = true ]]; then
            # 检查框架是否支持多卡
            if [[ "$framework" = "xinference" ]]; then
                log_warning "Xinference暂不支持多卡，跳过"
                return 0
            fi
            
            bash "$SCRIPT_DIR/scripts/run_multi_gpu_benchmark.sh" \
                -m "$model" \
                -f "$framework" \
                -n "$gpu_count" \
                -s "$TEST_SCENARIO" \
                -c "$CONFIG_FILE" \
                -r "$RESULTS_DIR" \
                -l "$LOGS_DIR" > "$test_log" 2>&1
        else
            log_info "跳过多卡测试: $model + $framework + ${gpu_count}GPU"
            return 0
        fi
    fi
    
    if [[ $? -eq 0 ]]; then
        log_success "测试完成: $model + $framework + ${gpu_count}GPU"
        SUCCESSFUL_TESTS=$((SUCCESSFUL_TESTS + 1))
    else
        log_error "测试失败: $model + $framework + ${gpu_count}GPU"
        log_error "详细日志: $test_log"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    
    # 短暂休息，让系统恢复
    sleep 30
}

# 运行所有测试
run_all_tests() {
    log_info "开始运行所有测试..."
    
    # 解析参数列表
    IFS=',' read -ra MODEL_LIST <<< "$MODELS"
    IFS=',' read -ra FRAMEWORK_LIST <<< "$FRAMEWORKS"
    IFS=',' read -ra GPU_COUNT_LIST <<< "$GPU_COUNTS"
    
    # 计算总测试数量
    local total_combinations=0
    for model in "${MODEL_LIST[@]}"; do
        for framework in "${FRAMEWORK_LIST[@]}"; do
            for gpu_count in "${GPU_COUNT_LIST[@]}"; do
                if [[ $gpu_count -eq 1 && "$TEST_SINGLE_GPU" = true ]]; then
                    total_combinations=$((total_combinations + 1))
                elif [[ $gpu_count -gt 1 && "$TEST_MULTI_GPU" = true && "$framework" != "xinference" ]]; then
                    total_combinations=$((total_combinations + 1))
                fi
            done
        done
    done
    
    log_info "计划运行 $total_combinations 个测试组合"
    
    # 运行测试
    local current_test=0
    for model in "${MODEL_LIST[@]}"; do
        for framework in "${FRAMEWORK_LIST[@]}"; do
            for gpu_count in "${GPU_COUNT_LIST[@]}"; do
                current_test=$((current_test + 1))
                
                log_header "进度: $current_test/$total_combinations"
                run_single_test "$model" "$framework" "$gpu_count"
                
                # 显示进度
                local progress=$((current_test * 100 / total_combinations))
                log_info "总体进度: $progress% ($current_test/$total_combinations)"
            done
        done
    done
    
    log_success "所有测试完成"
}

# 合并测试结果
merge_results() {
    log_info "合并测试结果..."
    
    local merged_file="$RESULTS_DIR/merged_results_${TIMESTAMP}.json"
    local all_results=()
    
    # 查找所有结果文件
    local result_files=($(find "$RESULTS_DIR" -name "benchmark_results_*.json" -newer "$SCRIPT_DIR" 2>/dev/null))
    
    if [[ ${#result_files[@]} -eq 0 ]]; then
        log_warning "未找到测试结果文件"
        return 1
    fi
    
    # 合并结果
    python3 << EOF
import json
import sys
from pathlib import Path

result_files = [
    $(printf '"%s",' "${result_files[@]}")
]

merged_results = []
metadata = {
    'generation_time': '$(date -Iseconds)',
    'total_result_files': len(result_files),
    'models': '$MODELS'.split(','),
    'frameworks': '$FRAMEWORKS'.split(','),
    'gpu_counts': [int(x) for x in '$GPU_COUNTS'.split(',')],
    'test_scenario': '$TEST_SCENARIO'
}

try:
    for file_path in result_files:
        if file_path:  # 跳过空字符串
            try:
                with open(file_path.strip('"'), 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    if isinstance(results, list):
                        merged_results.extend(results)
                    else:
                        merged_results.append(results)
            except Exception as e:
                print(f"读取文件失败 {file_path}: {e}", file=sys.stderr)
    
    # 保存合并结果
    final_result = {
        'metadata': metadata,
        'results': merged_results,
        'summary': {
            'total_tests': len(merged_results),
            'unique_configurations': len(set((r.get('framework'), r.get('model_name'), r.get('gpu_count')) for r in merged_results))
        }
    }
    
    with open('$merged_file', 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    print(f"合并了 {len(merged_results)} 个测试结果")
    
except Exception as e:
    print(f"合并结果失败: {e}", file=sys.stderr)
    sys.exit(1)
EOF
    
    if [[ $? -eq 0 ]]; then
        log_success "结果合并完成: $merged_file"
        export MERGED_RESULTS_FILE="$merged_file"
    else
        log_error "结果合并失败"
        return 1
    fi
}

# 生成综合报告
generate_comprehensive_report() {
    if [[ "$GENERATE_REPORT" != true ]]; then
        log_info "跳过报告生成"
        return 0
    fi
    
    log_info "生成综合报告..."
    
    if [[ -z "$MERGED_RESULTS_FILE" ]] || [[ ! -f "$MERGED_RESULTS_FILE" ]]; then
        log_error "未找到合并的结果文件"
        return 1
    fi
    
    local report_dir="$RESULTS_DIR/comprehensive_report_${TIMESTAMP}"
    mkdir -p "$report_dir"
    
    # 生成报告
    python3 "$SCRIPT_DIR/tools/generate_comprehensive_report.py" \
        --input "$MERGED_RESULTS_FILE" \
        --output "$report_dir"
    
    if [[ $? -eq 0 ]]; then
        log_success "综合报告已生成: $report_dir"
        
        # 显示报告内容
        if [[ -f "$report_dir/comprehensive_report.html" ]]; then
            log_info "HTML报告: $report_dir/comprehensive_report.html"
        fi
        if [[ -f "$report_dir/comprehensive_report.json" ]]; then
            log_info "JSON报告: $report_dir/comprehensive_report.json"
        fi
    else
        log_error "综合报告生成失败"
        return 1
    fi
}

# 显示测试摘要
show_test_summary() {
    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    log_header "=========================================="
    log_header "           测试完成摘要"
    log_header "=========================================="
    echo
    log_info "测试统计:"
    log_info "  总测试数: $TOTAL_TESTS"
    log_info "  成功测试: $SUCCESSFUL_TESTS"
    log_info "  失败测试: $FAILED_TESTS"
    log_info "  成功率: $(( TOTAL_TESTS > 0 ? SUCCESSFUL_TESTS * 100 / TOTAL_TESTS : 0 ))%"
    echo
    log_info "执行时间:"
    log_info "  开始时间: $(date -d @$START_TIME)"
    log_info "  结束时间: $(date -d @$end_time)"
    log_info "  总耗时: ${hours}小时${minutes}分钟${seconds}秒"
    echo
    log_info "输出文件:"
    log_info "  结果目录: $RESULTS_DIR"
    log_info "  日志目录: $LOGS_DIR"
    
    if [[ -n "$MERGED_RESULTS_FILE" ]]; then
        log_info "  合并结果: $MERGED_RESULTS_FILE"
    fi
    
    echo
    
    if [[ $FAILED_TESTS -gt 0 ]]; then
        log_warning "有 $FAILED_TESTS 个测试失败，请检查日志"
    else
        log_success "所有测试成功完成！"
    fi
    
    log_header "=========================================="
}

# 清理函数
cleanup() {
    if [[ "$CLEANUP_AFTER" != true ]]; then
        return 0
    fi
    
    log_info "清理测试环境..."
    
    # 停止所有可能运行的推理服务
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "xinference-local" 2>/dev/null || true
    
    # 清理GPU显存
    python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
        except:
            pass
    print('GPU显存已清理')
" 2>/dev/null || true
    
    log_success "清理完成"
}

# 信号处理
trap cleanup EXIT INT TERM

# 主函数
main() {
    # 显示配置
    show_test_config
    
    # 环境检查
    if ! check_environment; then
        log_error "环境检查失败"
        exit 1
    fi
    
    # 环境设置
    if ! setup_environment; then
        log_error "环境设置失败"
        exit 1
    fi
    
    # 运行测试
    run_all_tests
    
    # 合并结果
    merge_results
    
    # 生成报告
    generate_comprehensive_report
    
    # 显示摘要
    show_test_summary
    
    # 退出码
    if [[ $FAILED_TESTS -gt 0 ]]; then
        exit 1
    else
        exit 0
    fi
}

# 执行主函数
main "$@" 