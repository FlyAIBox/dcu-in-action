#!/bin/bash

# 海光DCU K100-AI单卡基准测试脚本
# 版本: v2.0

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

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 默认参数
MODEL_NAME="deepseek-7b"
FRAMEWORK="vllm"
CONFIG_FILE="$BASE_DIR/configs/benchmark_config.yaml"
RESULTS_DIR="$BASE_DIR/results"
LOGS_DIR="$BASE_DIR/logs"
GPU_ID=0
TEST_SCENARIO="standard_test"

# 解析命令行参数
usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --model MODEL_NAME       模型名称 (默认: deepseek-7b)"
    echo "  -f, --framework FRAMEWORK    推理框架 (vllm|sglang|xinference, 默认: vllm)"
    echo "  -g, --gpu GPU_ID              GPU设备ID (默认: 0)"
    echo "  -c, --config CONFIG_FILE      配置文件路径"
    echo "  -s, --scenario SCENARIO       测试场景 (quick_test|standard_test|comprehensive_test|stress_test)"
    echo "  -r, --results RESULTS_DIR     结果输出目录"
    echo "  -l, --logs LOGS_DIR           日志输出目录"
    echo "  -h, --help                    显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -m deepseek-7b -f vllm -g 0"
    echo "  $0 -m qwen-7b -f sglang -s quick_test"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -f|--framework)
            FRAMEWORK="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -s|--scenario)
            TEST_SCENARIO="$2"
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

# 验证参数
if [[ ! "$FRAMEWORK" =~ ^(vllm|sglang|xinference)$ ]]; then
    log_error "不支持的框架: $FRAMEWORK"
    exit 1
fi

if [[ ! "$TEST_SCENARIO" =~ ^(quick_test|standard_test|comprehensive_test|stress_test)$ ]]; then
    log_error "不支持的测试场景: $TEST_SCENARIO"
    exit 1
fi

# 检查配置文件
if [[ ! -f "$CONFIG_FILE" ]]; then
    log_error "配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 创建目录
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# 设置环境变量
export HIP_VISIBLE_DEVICES="$GPU_ID"
export ROCM_PATH="/opt/rocm"
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"
export VLLM_ATTENTION_BACKEND="FLASHINFER"
export OMP_NUM_THREADS="8"
export CUDA_LAUNCH_BLOCKING="0"

# 显示测试信息
log_info "========== 海光DCU K100-AI单卡基准测试 =========="
log_info "模型: $MODEL_NAME"
log_info "框架: $FRAMEWORK"
log_info "GPU设备: $GPU_ID"
log_info "测试场景: $TEST_SCENARIO"
log_info "配置文件: $CONFIG_FILE"
log_info "结果目录: $RESULTS_DIR"
log_info "日志目录: $LOGS_DIR"
log_info "================================================"

# 检查DCU设备状态
check_dcu_status() {
    log_info "检查DCU设备状态..."
    
    if ! command -v rocm-smi &> /dev/null; then
        log_error "rocm-smi命令未找到"
        exit 1
    fi
    
    # 检查指定GPU是否存在
    if ! rocm-smi --showid | grep -q "GPU\[$GPU_ID\]"; then
        log_error "GPU[$GPU_ID]不存在"
        exit 1
    fi
    
    # 显示GPU状态
    log_info "GPU[$GPU_ID]状态:"
    rocm-smi --showid --showtemp --showpower --showuse -d $GPU_ID
    
    log_success "DCU设备状态检查完成"
}

# 激活Python环境
activate_python_env() {
    log_info "激活Python环境: $FRAMEWORK"
    
    if ! command -v conda &> /dev/null; then
        log_error "conda命令未找到，请先安装Miniconda"
        exit 1
    fi
    
    # 检查环境是否存在
    if ! conda env list | grep -q "^$FRAMEWORK "; then
        log_error "Python环境不存在: $FRAMEWORK"
        log_info "请运行环境配置脚本: ./scripts/setup_environment.sh"
        exit 1
    fi
    
    # 激活环境
    source ~/miniconda3/bin/activate $FRAMEWORK
    
    log_success "Python环境激活成功"
}

# 准备测试配置
prepare_test_config() {
    log_info "准备测试配置..."
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    TEST_CONFIG_FILE="$LOGS_DIR/test_config_${MODEL_NAME}_${FRAMEWORK}_${TIMESTAMP}.yaml"
    
    # 从主配置文件中提取相关配置
    python3 << EOF
import yaml
import os
import sys

config_file = "$CONFIG_FILE"
test_config_file = "$TEST_CONFIG_FILE"
model_name = "$MODEL_NAME"
framework = "$FRAMEWORK"
test_scenario = "$TEST_SCENARIO"

try:
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 查找匹配的测试用例
    test_case = None
    for case in config['test_cases']:
        if (case.get('model_name') == model_name and 
            case.get('framework') == framework and
            case.get('gpu_count') == 1):
            test_case = case.copy()
            break
    
    if not test_case:
        print(f"未找到匹配的测试用例: {model_name}-{framework}")
        sys.exit(1)
    
    # 应用测试场景参数
    if test_scenario in config.get('test_scenarios', {}):
        scenario_params = config['test_scenarios'][test_scenario]
        test_case.update(scenario_params)
    
    # 创建测试配置
    test_config = {
        'test_cases': [test_case],
        'global_config': config.get('global_config', {}),
        'optimization_config': config.get('optimization_config', {}),
        'monitoring_config': config.get('monitoring_config', {}),
        'environment': config.get('environment', {})
    }
    
    # 保存配置
    with open(test_config_file, 'w', encoding='utf-8') as f:
        yaml.dump(test_config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"测试配置已生成: {test_config_file}")
    
except Exception as e:
    print(f"准备测试配置失败: {e}")
    sys.exit(1)
EOF
    
    if [[ $? -ne 0 ]]; then
        log_error "准备测试配置失败"
        exit 1
    fi
    
    export TEST_CONFIG_FILE
    log_success "测试配置准备完成: $TEST_CONFIG_FILE"
}

# 启动性能监控
start_monitoring() {
    log_info "启动性能监控..."
    
    MONITOR_LOG="$LOGS_DIR/monitor_${MODEL_NAME}_${FRAMEWORK}_${TIMESTAMP}.log"
    
    # 启动性能监控器
    python3 "$BASE_DIR/benchmark/performance_monitor.py" \
        --interval 5 \
        --duration 3600 > "$MONITOR_LOG" 2>&1 &
    
    MONITOR_PID=$!
    export MONITOR_PID
    
    log_info "性能监控已启动 (PID: $MONITOR_PID)"
    sleep 5  # 等待监控器启动
}

# 停止性能监控
stop_monitoring() {
    if [[ -n "$MONITOR_PID" ]]; then
        log_info "停止性能监控..."
        kill $MONITOR_PID 2>/dev/null || true
        wait $MONITOR_PID 2>/dev/null || true
        log_success "性能监控已停止"
    fi
}

# 运行基准测试
run_benchmark() {
    log_info "开始运行基准测试..."
    
    BENCHMARK_LOG="$LOGS_DIR/benchmark_${MODEL_NAME}_${FRAMEWORK}_${TIMESTAMP}.log"
    
    # 运行基准测试控制器
    cd "$BASE_DIR"
    
    python3 "$BASE_DIR/benchmark/benchmark_controller.py" \
        --config "$TEST_CONFIG_FILE" > "$BENCHMARK_LOG" 2>&1
    
    if [[ $? -eq 0 ]]; then
        log_success "基准测试完成"
    else
        log_error "基准测试失败，请检查日志: $BENCHMARK_LOG"
        return 1
    fi
}

# 生成测试报告
generate_report() {
    log_info "生成测试报告..."
    
    REPORT_DIR="$RESULTS_DIR/reports"
    mkdir -p "$REPORT_DIR"
    
    # 查找最新的结果文件
    LATEST_RESULT=$(ls -t "$RESULTS_DIR"/benchmark_results_*.json 2>/dev/null | head -1)
    
    if [[ -z "$LATEST_RESULT" ]]; then
        log_warning "未找到测试结果文件"
        return 1
    fi
    
    # 生成报告
    python3 "$BASE_DIR/tools/generate_comprehensive_report.py" \
        --input "$LATEST_RESULT" \
        --output "$REPORT_DIR" \
        --model "$MODEL_NAME" \
        --framework "$FRAMEWORK" \
        --gpu-count 1
    
    if [[ $? -eq 0 ]]; then
        log_success "测试报告已生成: $REPORT_DIR"
    else
        log_warning "报告生成失败"
    fi
}

# 清理函数
cleanup() {
    log_info "清理资源..."
    
    # 停止监控
    stop_monitoring
    
    # 停止可能运行的推理服务
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "xinference-local" 2>/dev/null || true
    
    # 清理GPU显存
    python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU显存已清理')
" 2>/dev/null || true
    
    log_success "清理完成"
}

# 信号处理
trap cleanup EXIT INT TERM

# 主执行流程
main() {
    log_info "开始单卡基准测试..."
    
    # 检查环境
    check_dcu_status
    activate_python_env
    
    # 准备测试
    prepare_test_config
    start_monitoring
    
    # 运行测试
    if run_benchmark; then
        log_success "基准测试成功完成"
        
        # 生成报告
        generate_report
        
        # 显示结果摘要
        log_info "测试结果摘要:"
        if [[ -f "$LATEST_RESULT" ]]; then
            python3 -c "
import json
with open('$LATEST_RESULT', 'r') as f:
    results = json.load(f)
    
if results:
    result = results[-1]  # 最后一个结果
    print(f'模型: {result.get(\"model_name\", \"N/A\")}')
    print(f'框架: {result.get(\"framework\", \"N/A\")}')
    print(f'吞吐量: {result.get(\"throughput_tokens_per_sec\", 0):.2f} tokens/s')
    print(f'延迟P50: {result.get(\"latency_p50_ms\", 0):.2f} ms')
    print(f'GPU利用率: {result.get(\"gpu_utilization_avg\", 0):.1f}%')
    print(f'错误率: {result.get(\"error_rate\", 0):.2f}%')
"
        fi
        
    else
        log_error "基准测试失败"
        exit 1
    fi
}

# 执行主函数
main "$@" 