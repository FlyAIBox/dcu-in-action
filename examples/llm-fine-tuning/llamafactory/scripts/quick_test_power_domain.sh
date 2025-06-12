#!/bin/bash
# 电力领域微调快速测试脚本
# 用于验证环境配置和基本功能

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          电力领域大模型微调环境快速测试                     ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

# 测试结果统计
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# 测试函数
run_test() {
    local test_name=$1
    local test_command=$2
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -e "\n${BLUE}测试 $TOTAL_TESTS: $test_name${NC}"
    
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ 通过${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗ 失败${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

# 1. 检查DCU环境
echo -e "\n${YELLOW}=== DCU环境检查 ===${NC}"
run_test "DCU驱动" "command -v hy-smi"
run_test "DCU设备" "hy-smi -L | grep -q DCU"

# 显示DCU信息
if command -v hy-smi &> /dev/null; then
    echo -e "\n${YELLOW}DCU设备信息:${NC}"
    hy-smi --query-gpu=index,name,memory.total --format=csv
fi

# 2. 检查Python环境
echo -e "\n${YELLOW}=== Python环境检查 ===${NC}"
run_test "Conda环境" "conda info --envs | grep -q llamafactory-dcu"
run_test "Python版本" "python --version | grep -q 'Python 3'"

# 激活环境
if conda info --envs | grep -q llamafactory-dcu; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate llamafactory-dcu 2>/dev/null || true
fi

# 3. 检查依赖包
echo -e "\n${YELLOW}=== 依赖包检查 ===${NC}"
run_test "PyTorch" "python -c 'import torch'"
run_test "Transformers" "python -c 'import transformers'"
run_test "PEFT" "python -c 'import peft'"
run_test "DeepSpeed" "python -c 'import deepspeed'"
run_test "LLaMA Factory" "command -v llamafactory-cli"

# 4. 检查项目文件
echo -e "\n${YELLOW}=== 项目文件检查 ===${NC}"
run_test "DeepSpeed配置" "[ -f configs/deepspeed_qwen3_32b_8dcu.json ]"
run_test "训练配置" "[ -f configs/qwen3_32b_power_domain_8dcu.yaml ]"
run_test "数据集脚本" "[ -f scripts/prepare_power_domain_dataset.py ]"
run_test "训练脚本" "[ -f scripts/train_power_domain.sh ]"
run_test "推理脚本" "[ -f scripts/power_domain_inference.py ]"

# 5. 检查数据集
echo -e "\n${YELLOW}=== 数据集检查 ===${NC}"
run_test "示例数据" "[ -f data/power_domain_samples.json ]"

# 检查数据集内容
if [ -f data/power_domain_samples.json ]; then
    SAMPLE_COUNT=$(python -c "import json; print(len(json.load(open('data/power_domain_samples.json'))))" 2>/dev/null || echo "0")
    echo -e "示例数据包含 ${GREEN}$SAMPLE_COUNT${NC} 条样本"
fi

# 10. 显示测试总结
echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                      测试总结                               ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n总测试数: ${BLUE}$TOTAL_TESTS${NC}"
echo -e "通过测试: ${GREEN}$PASSED_TESTS${NC}"
echo -e "失败测试: ${RED}$FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}✓ 所有测试通过！环境配置正确，可以开始电力领域微调。${NC}"
    echo -e "\n${GREEN}下一步操作:${NC}"
    echo -e "1. 准备数据: ${BLUE}python scripts/prepare_power_domain_dataset.py${NC}"
    echo -e "2. 开始训练: ${BLUE}./scripts/train_power_domain.sh${NC}"
    echo -e "3. 监控训练: ${BLUE}./scripts/monitor_dcu_training.sh${NC}"
else
    echo -e "\n${RED}✗ 有 $FAILED_TESTS 个测试失败，请检查环境配置。${NC}"
    echo -e "\n${YELLOW}常见问题解决:${NC}"
    echo -e "1. DCU驱动问题: 确保已安装ROCK驱动"
    echo -e "2. Python环境: 运行 ${BLUE}./scripts/dcu_k100_ai_setup.sh${NC}"
    echo -e "3. 权限问题: 使用 ${BLUE}chmod +x scripts/*.sh${NC}"
fi

echo -e "\n${GREEN}测试完成！${NC}" 