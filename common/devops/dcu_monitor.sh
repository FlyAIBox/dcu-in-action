#!/bin/bash

# DCU监控脚本 - 实时显示DCU加速卡使用情况
# 功能:
#   - 显示DCU型号
#   - 显存使用情况(已用/总容量)
#   - 使用率
#   - 温度
#   - 显存带宽
#   - 卡间互联带宽(可选)
#   - 应用进程信息
#   - 模型微调进度监控
#   - 汇总信息

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
CLEAR='\033[0m'

# 设置默认值
SHOW_INTERCONNECT=false
INTERVAL=2
MODEL_DIR=""
LOG_DIR=""
VERBOSE=false
SHOW_SUMMARY=true
WATCH_MODE=false

# 设置ROCm工具路径
ROCM_PATH="/opt/dtk-25.04/bin"
ROCM_SMI="${ROCM_PATH}/rocm-smi"
ROCM_BANDWIDTH_TEST="${ROCM_PATH}/rocm-bandwidth-test"

# 帮助信息
show_help() {
    echo -e "${CYAN}DCU Monitor - 监控DCU加速卡使用情况${CLEAR}"
    echo "用法: $0 [选项]"
    echo
    echo "选项:"
    echo "  -w, --watch               启动持续监控模式 (默认: 单次运行)"
    echo "  -t, --time SECONDS        刷新时间间隔(仅在-w模式下有效，默认: 2秒)"
    echo "  -i, --interconnect        显示卡间互联带宽信息"
    echo "  -m, --model-dir DIR       模型训练目录，用于监控微调进度"
    echo "  -l, --log-dir DIR         日志目录，用于解析模型训练进度"
    echo "  -v, --verbose             显示详细信息"
    echo "  -s, --summary             显示汇总信息表格(默认: 开启)"
    echo "  -n, --no-summary          不显示汇总信息表格"
    echo "  -h, --help                显示此帮助信息"
    echo
    echo "示例:"
    echo "  $0                        运行一次并显示所有信息"
    echo "  $0 -w -t 5                每5秒刷新一次监控信息"
    echo "  $0 -i                     运行一次，并显示卡间互联带宽"
    echo "  $0 -m /path/to/model      监控指定目录下的模型训练进度"
    echo "  $0 -n                     不显示汇总信息表格"
    echo
    exit 0
}

# 处理命令行参数
while [ $# -gt 0 ]; do
    case $1 in
        -w|--watch)
            WATCH_MODE=true
            shift
            ;;
        -i|--interconnect)
            SHOW_INTERCONNECT=true
            shift
            ;;
        -t|--time)
            INTERVAL="$2"
            shift 2
            ;;
        -m|--model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        -l|--log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -s|--summary)
            SHOW_SUMMARY=true
            shift
            ;;
        -n|--no-summary)
            SHOW_SUMMARY=false
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "未知选项: $1"
            show_help
            ;;
    esac
done

# 检查必要命令是否存在
check_requirements() {
    if [ ! -x "$ROCM_SMI" ]; then
        echo -e "${RED}错误: rocm-smi 未找到。请确保ROCm已正确安装或调整ROCM_PATH路径。${CLEAR}"
        echo "当前设置的路径为: $ROCM_SMI"
        echo "尝试使用 'command -v rocm-smi' 查找正确路径"
        exit 1
    fi
    
    if [ "$SHOW_INTERCONNECT" = "true" ] && [ ! -x "$ROCM_BANDWIDTH_TEST" ]; then
        echo -e "${YELLOW}警告: rocm-bandwidth-test 未找到。卡间互联带宽功能将不可用。${CLEAR}"
        echo "当前设置的路径为: $ROCM_BANDWIDTH_TEST"
        SHOW_INTERCONNECT=false
    fi

    if ! command -v column &> /dev/null; then
        echo -e "${YELLOW}警告: column 命令未找到，表格显示可能不美观。${CLEAR}"
        echo "请安装 util-linux 包以获得更好的表格显示效果。"
    fi
    
    if ! command -v awk &> /dev/null; then
        echo -e "${RED}错误: awk 命令未找到。该脚本需要 awk 才能运行。${CLEAR}"
        exit 1
    fi
}

# 创建表格函数
create_table() {
    if command -v column &> /dev/null; then
        column -t -s "|" -o " | "
    else
        # Fallback to simple formatting if column is not available
        sed 's/|/\t/g'
    fi
}

# 获取DCU型号信息
get_dcu_model() {
    echo -e "${CYAN}=== DCU型号信息 ===${CLEAR}"
    
    # 使用 --showhw 获取信息
    local smi_output
    smi_output=$($ROCM_SMI --showhw 2>/dev/null)
    
    if [ -z "$smi_output" ]; then
        echo "无法获取DCU型号信息。"
        return
    fi
    
    (
        echo "ID|厂商|型号"
        echo "--|--|--"
        echo "$smi_output" | awk '
        /GPU\[[0-9]+\]/ || /HCU\[[0-9]+\]/ {
            if (current_id != "") {
                # In case a card entry ends
                if (vendor == "") vendor = "N/A"
                if (model == "") model = "N/A"
                printf "%s|%s|%s\n", current_id, vendor, model
            }
            current_id = $1
            sub(/:/, "", current_id)
            vendor = ""
            model = ""
        }
        /Card series/ {
            sub(/.*Card series:[[:space:]]*/, "")
            model = $0
        }
        /Card vendor/ {
            sub(/.*Card vendor:[[:space:]]*/, "")
            vendor = $0
        }
        END {
            if (current_id != "") {
                if (vendor == "") vendor = "N/A"
                if (model == "") model = "N/A"
                printf "%s|%s|%s\n", current_id, vendor, model
            }
        }'
    ) | create_table
    echo
}

# 获取显存使用情况
get_memory_usage() {
    echo -e "${CYAN}=== 显存使用情况 ===${CLEAR}"
    
    local smi_output
    smi_output=$($ROCM_SMI --showmeminfo vram 2>/dev/null)

    if [ -z "$smi_output" ]; then
        echo "无法获取显存信息。"
        return
    fi

    (
        echo "设备|总显存(MiB)|已用显存(MiB)|使用率(%)"
        echo "--|--|--|--"
        echo "$smi_output" | awk '
        /GPU\[[0-9]+\]/ || /HCU\[[0-9]+\]/ {
            dev = $1
            sub(/:/, "", dev)
        }
        /vram Total Memory/ {
            total[dev] = $NF
        }
        /vram Total Used Memory/ {
            used[dev] = $NF
        }
        END {
            total_all = 0
            used_all = 0
            for (d in total) {
                usage = 0
                if (total[d] > 0) {
                    usage = (used[d] / total[d]) * 100
                }
                printf "%s|%d|%d|%.2f\n", d, total[d], used[d], usage
                total_all += total[d]
                used_all += used[d]
            }
            if (total_all > 0) {
                usage_all = (used_all / total_all) * 100
                printf "总计|%d|%d|%.2f\n", total_all, used_all, usage_all
            }
        }'
    ) | create_table
    echo
}

# 获取DCU温度和使用率
get_temp_and_util() {
    echo -e "${CYAN}=== DCU温度和使用率 ===${CLEAR}"

    local smi_output
    smi_output=$($ROCM_SMI 2>/dev/null)

    if [ -z "$smi_output" ] || ! echo "$smi_output" | grep -q "Temp"; then
        echo "无法获取温度和使用率信息。"
        return
    fi
    
    (
        echo "设备|温度(°C)|GPU使用率(%)"
        echo "--|--|--"
        # Skip header and footer lines of rocm-smi output
        echo "$smi_output" | awk '
        /Perf/ && /PwrCap/ {next} # Skip header
        /=/ {next} # Skip separators
        NF > 5 && ($1 ~ /^[0-9]+$/) {
            dev_id = "HCU[" $1 "]"
            temp = $2
            sub(/C/, "", temp)
            gpu_use = $7
            sub(/%/, "", gpu_use)
            printf "%s|%s|%s\n", dev_id, temp, gpu_use
        }'
    ) | create_table
    echo
}

# 计算显存带宽
get_memory_bandwidth() {
    echo -e "${CYAN}=== 显存带宽 ===${CLEAR}"
    
    local smi_output
    smi_output=$($ROCM_SMI --showclocks 2>/dev/null)

    if [ -z "$smi_output" ]; then
        echo "无法获取时钟频率信息。"
        return
    fi

    (
        echo "设备|时钟频率(MHz)|理论峰值带宽(GB/s)"
        echo "--|--|--"
        echo "$smi_output" | awk '
        /mclk/ {
            gpu_id = "";
            if (match($0, /GPU\[[0-9]*\]/)) {
                gpu_id = substr($0, RSTART, RLENGTH)
            } else if (match($0, /HCU\[[0-9]*\]/)) {
                gpu_id = substr($0, RSTART, RLENGTH)
            }
            
            if (match($0, /[0-9]+Mhz/)) {
                clock_mhz = substr($0, RSTART, RLENGTH-3)
                
                # Bandwidth calculation (assuming HBM2, 4096-bit width)
                bit_width = 4096
                multiplier = 2  # DDR
                bandwidth = (clock_mhz * multiplier * bit_width / 8) / 1000
                
                printf "%s|%s|%.2f\n", gpu_id, clock_mhz, bandwidth
            }
        }'
    ) | create_table
    
    echo -e "\n注意: 理论带宽计算基于默认内存规格(位宽4096bit)，请根据实际DCU型号调整"
    echo
}

# 获取卡间互联带宽(可选)
get_interconnect_bandwidth() {
    if [ "$SHOW_INTERCONNECT" = "true" ]; then
        echo -e "${CYAN}=== 卡间互联带宽 ===${CLEAR}"
        echo "正在测试卡间带宽，这可能需要几分钟..."
        
        local bandwidth_output
        bandwidth_output=$($ROCM_BANDWIDTH_TEST 2>/dev/null)

        if [ -z "$bandwidth_output" ]; then
            echo "无法获取带宽测试结果。"
            return
        fi

        (
            echo "$bandwidth_output" | awk '/D2D Bandwidth Matrix/ {p=1; next} /^[^0-9]/ {if(p) exit} p' | head -n 20
        ) | sed 's/^[ \t]*//' | awk 'NF > 0' | create_table
        
        echo
    fi
}

# 获取应用进程信息
get_process_info() {
    echo -e "${CYAN}=== DCU应用进程 ===${CLEAR}"
    
    local pids_output
    pids_output=$($ROCM_SMI --showpids 2>/dev/null)

    if [ -z "$pids_output" ] || ! echo "$pids_output" | grep -q "Process ID"; then
        echo "无正在运行的DCU进程"
        echo
        return
    fi

    (
        echo "设备|进程ID|GPU内存(MiB)|命令行"
        echo "--|--|--|--"
        echo "$pids_output" | awk '
        /GPU\[[0-9]+\]/ || /HCU\[[0-9]+\]/ {
            dev = $1
            sub(/:/, "", dev)
            next
        }
        /No KFD PIDs/ { next }
        /Process ID/ { next }
        /==/ { next }
        NF >= 3 {
            pid = $1
            mem = $2
            cmd = ""
            for (i=3; i<=NF; i++) cmd = cmd " " $i
            sub(/^ /, "", cmd)
            printf "%s|%s|%s|%s\n", dev, pid, mem, cmd
        }'
    ) | create_table
    echo
}

# 监控模型微调进度
monitor_model_progress() {
    if [ -n "$MODEL_DIR" ] || [ -n "$LOG_DIR" ]; then
        echo -e "${CYAN}=== 模型微调进度 ===${CLEAR}"
        
        if [ -n "$MODEL_DIR" ] && [ -d "$MODEL_DIR" ]; then
            (
                echo "检查点|保存时间"
                echo "--|--"
                latest_checkpoint=$(find "$MODEL_DIR" -type f -name "checkpoint-*" | sort -V | tail -n 1)
                if [ -n "$latest_checkpoint" ]; then
                    checkpoint_name=$(basename "$latest_checkpoint")
                    checkpoint_time=$(stat -c %y "$latest_checkpoint" 2>/dev/null || stat -f "%Sm" "$latest_checkpoint")
                    echo "$checkpoint_name|$checkpoint_time"
                else
                    echo "未找到检查点|N/A"
                fi
            ) | create_table
            echo
        fi
        
        if [ -n "$LOG_DIR" ] && [ -d "$LOG_DIR" ]; then
            echo -e "最新训练日志进度:"
            latest_log=$(find "$LOG_DIR" -type f \( -name "*.log" -o -name "train*.txt" \) -print0 | xargs -0 ls -t | head -n 1)
            if [ -n "$latest_log" ]; then
                echo "日志文件: $latest_log"
                echo "最近进度:"
                tail -n 10 "$latest_log" | grep -E -i "loss|accuracy|epoch|step|progress" | tail -n 3
            else
                echo "未找到训练日志"
            fi
        fi
        
        if [ -z "$MODEL_DIR" ] && [ -z "$LOG_DIR" ]; then
            echo "未指定模型目录或日志目录"
        fi
        echo
    fi
}

# 生成DCU汇总信息表格
get_dcu_summary() {
    if [ "$SHOW_SUMMARY" = "true" ]; then
        echo -e "${CYAN}=== DCU汇总信息 ===${CLEAR}"
        
        # Gather all data
        local model_info=$($ROCM_SMI --showhw 2>/dev/null)
        local mem_info=$($ROCM_SMI --showmeminfo vram 2>/dev/null)
        local util_info=$($ROCM_SMI 2>/dev/null)
        
        (
            echo "设备|型号|显存总量(MiB)|已用显存(MiB)|使用率(%)|温度(°C)|GPU使用率(%)"
            echo "--|--|--|--|--|--|--"

            # Combine data using awk
            awk -v model_info="$model_info" -v mem_info="$mem_info" -v util_info="$util_info" '
            BEGIN {
                # Parse model info
                split(model_info, lines, "\n")
                for (i in lines) {
                    if (lines[i] ~ /GPU\[[0-9]+\]/ || lines[i] ~ /HCU\[[0-9]+\]/) {
                        id = lines[i]; sub(/:.*/, "", id)
                    }
                    if (lines[i] ~ /Card series/) {
                        model = lines[i]; sub(/.*Card series:[ ]*/, "", model)
                        models[id] = model
                    }
                }

                # Parse memory info
                split(mem_info, lines, "\n")
                for (i in lines) {
                    if (lines[i] ~ /GPU\[[0-9]+\]/ || lines[i] ~ /HCU\[[0-9]+\]/) {
                        id = lines[i]; sub(/:.*/, "", id)
                    }
                    if (lines[i] ~ /vram Total Memory/) {
                        total_mem[id] = $NF
                    }
                    if (lines[i] ~ /vram Total Used Memory/) {
                        used_mem[id] = $NF
                    }
                }

                # Parse utilization info
                split(util_info, lines, "\n")
                for (i in lines) {
                    if (lines[i] ~ /^[0-9]/ && NF > 5) {
                        id = "HCU[" $1 "]"
                        temps[id] = $2; sub(/C/, "", temps[id])
                        gpu_usages[id] = $7; sub(/%/, "", gpu_usages[id])
                    }
                }

                # Print combined table
                for (id in models) {
                    mem_usage = 0
                    if (total_mem[id] > 0) {
                        mem_usage = (used_mem[id] / total_mem[id]) * 100
                    }
                    printf "%s|%s|%d|%d|%.2f|%s|%s\n",
                        id,
                        models[id] ? models[id] : "N/A",
                        total_mem[id] ? total_mem[id] : 0,
                        used_mem[id] ? used_mem[id] : 0,
                        mem_usage,
                        temps[id] ? temps[id] : "N/A",
                        gpu_usages[id] ? gpu_usages[id] : "N/A"
                }
            }'
        ) | create_table
        echo
    fi
}

# 显示所有加速卡总显存
get_total_vram_summary() {
    echo -e "${CYAN}=== 所有加速卡显存汇总 ===${CLEAR}"
    $ROCM_SMI --showmeminfo vram | awk '
    /vram Total Memory \(MiB\):/ { total_vram += $NF }
    /vram Total Used Memory \(MiB\):/ { used_vram += $NF }
    END {
      printf "所有加速卡总显存 (Total VRAM): %s MiB\n", total_vram;
      printf "所有加速卡已用显存 (Total Used VRAM): %s MiB\n", used_vram;
      if (total_vram > 0) {
          usage = (used_vram / total_vram) * 100;
          printf "总体使用率: %.2f%%\n", usage;
      }
    }'
    echo
}

run_monitor() {
    if [ "$WATCH_MODE" = "true" ]; then
        clear
        echo -e "${GREEN}============= DCU 监控 ($(date '+%Y-%m-%d %H:%M:%S')) [刷新间隔: ${INTERVAL}秒] =============${CLEAR}"
    else
        echo -e "${GREEN}============= DCU 监控 ($(date '+%Y-%m-%d %H:%M:%S')) =============${CLEAR}"
    fi

    get_dcu_model
    get_memory_usage
    get_temp_and_util
    get_memory_bandwidth
    get_process_info
    if [ "$SHOW_SUMMARY" = "true" ]; then
        get_dcu_summary
    fi
    monitor_model_progress

    if [ "$SHOW_INTERCONNECT" = "true" ]; then
        get_interconnect_bandwidth
    fi
    
    get_total_vram_summary

    if [ "$VERBOSE" = "true" ]; then
        echo -e "${PURPLE}==== 帮助信息 ====${CLEAR}"
        echo "按 Ctrl+C 退出监控"
        echo "运行 '$0 --help' 查看更多选项"
    fi
    
    echo -e "${GREEN}========================================================================${CLEAR}"
}

# 主函数
main() {
    check_requirements
    
    if [ "$WATCH_MODE" = "true" ]; then
        while true; do
            run_monitor
            sleep "$INTERVAL"
        done
    else
        run_monitor
    fi
}

# 运行主函数
main 