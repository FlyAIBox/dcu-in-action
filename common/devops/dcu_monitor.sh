#!/bin/bash

# DCU监控脚本 - 实时显示DCU加速卡使用情况
#
# 功能:
# - 以美观的表格形式显示DCU状态
# - 包含型号、显存、温度、显存带宽、进程信息等
# - 支持单次运行和持续监控模式
# - 可选的耗时操作（如卡间带宽测试）

# --- 颜色定义 ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;34m'
NC='\033[0m' # No Color

# --- 默认值 ---
WATCH_MODE=false
INTERVAL=2
RUN_INTERCONNECT_TEST=false

# --- Spinner ---
spinner_pid=
trap 'kill "$spinner_pid" 2>/dev/null; exit' SIGINT SIGTERM

start_spinner() {
    (
        while :; do
            for s in / - \\ \|; do
                printf "\r\e[0;36m[%s]\e[0m %s..." "$s" "$1"
                sleep 0.2
            done
        done
    ) &
    spinner_pid=$!
    disown
}

stop_spinner() {
    if [[ -n $spinner_pid ]]; then
        kill "$spinner_pid"
        wait "$spinner_pid" 2>/dev/null
    fi
    printf "\r%s\n" "                                                                               "
}

# --- 工具路径 ---
# 优先使用环境变量 $ROCM_PATH, 否则使用默认值
ROCM_PATH=${ROCM_PATH:-/opt/dtk-25.04}
ROCM_SMI="${ROCM_PATH}/bin/rocm-smi"
ROCM_BANDWIDTH_TEST="${ROCM_PATH}/bin/rocm-bandwidth-test"


# --- 帮助信息 ---
show_help() {
    echo -e "${CYAN}DCU 监控面板 - v2.2${NC}"
    echo "一个用于监控和显示DCU状态的增强脚本。"
    echo
    echo "用法: $0 [选项]"
    echo
    echo "选项:"
    echo "  -w, --watch          启动持续监控模式 (默认: 单次运行)"
    echo "  -t, --time SEC       刷新时间间隔 (仅在 -w 模式下有效, 默认: 2秒)"
    echo "  -i, --interconnect   运行并显示卡间互联带宽测试 (耗时较长，会增加一列)"
    echo "  -h, --help           显示此帮助信息"
    echo
    echo "示例:"
    echo "  $0                   # 运行一次，显示基本信息"
    echo "  $0 -w -t 5           # 每5秒刷新一次基本信息"
    echo "  $0 -i                # 运行一次，包含卡间带宽测试"
    echo
    echo "【推荐】将本脚本目录加入PATH，并设置别名，便于全局调用："
    echo "  echo \"export PATH=\$PATH:/your/script/dir\" >> ~/.bashrc"
    echo "  echo \"alias dcu-mon='dcu_monitor.sh'\" >> ~/.bashrc"
    echo "  source ~/.bashrc"
    echo "  # 之后可直接用 dcu-mon 调用"
    exit 0
}

# --- 参数处理 ---
while [ $# -gt 0 ]; do
    case "$1" in
        -w|--watch)
            WATCH_MODE=true
            shift
            ;;
        -t|--time)
            if [[ "$2" =~ ^[0-9]+$ ]]; then
            INTERVAL="$2"
            shift 2
            else
                echo -e "${RED}错误: -t/--time 需要一个有效的秒数。${NC}" >&2
                exit 1
            fi
            ;;
        -i|--interconnect)
            RUN_INTERCONNECT_TEST=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}" >&2
            show_help
            ;;
    esac
done

# --- 检查依赖 ---
check_requirements() {
    if ! command -v "$ROCM_SMI" &> /dev/null; then
        echo -e "${RED}错误: rocm-smi 未在 '$ROCM_SMI' 找到。${NC}" >&2
        echo "请确保ROCm已正确安装，或设置正确的 \$ROCM_PATH 环境变量。" >&2
        exit 1
    fi
    if "$RUN_INTERCONNECT_TEST" && ! command -v "$ROCM_BANDWIDTH_TEST" &> /dev/null; then
        echo -e "${YELLOW}警告: rocm-bandwidth-test 未在 '$ROCM_BANDWIDTH_TEST' 找到。${NC}" >&2
        echo "将跳过卡间互联带宽测试。" >&2
        RUN_INTERCONNECT_TEST=false
    fi
}

# --- 数据采集模块 ---
declare -A gpus # 使用关联数组存储所有GPU数据

get_gpu_list() {
    mapfile -t gpu_ids < <("$ROCM_SMI" -i | grep -o 'HCU\[[0-9]\+\]' | sort -u)
    for id in "${gpu_ids[@]}"; do
        gpus["$id,name"]="$id"
    done
}

get_base_info() {
    local smi_output
    smi_output=$("$ROCM_SMI")
    while read -r line; do
        if [[ $line =~ ^[0-9]+[[:space:]] ]]; then
            local id="HCU[$(echo "$line" | awk '{print $1}')]"
            gpus["$id,temp"]=$(echo "$line" | awk '{print $2}')
            gpus["$id,util"]=$(echo "$line" | awk '{print $7}')
        fi
    done <<< "$smi_output"

    local hw_info
    hw_info=$("$ROCM_SMI" --showhw --showallinfo)
    while read -r line; do
        if [[ $line =~ "Card Series" ]]; then
            local id
            id=$(echo "$line" | grep -o 'HCU\[[0-9]\+\]')
            if [[ -n "$id" ]]; then
                local model
                model=$(echo "$line" | sed 's/.*Card Series:[[:space:]]*//' | xargs)
                gpus["$id,model"]=$model
            fi
        fi
    done <<< "$hw_info"
}

get_mem_info() {
    local mem_output
    mem_output=$("$ROCM_SMI" --showmeminfo vram)
    local current_id=""
    while read -r line; do
        if [[ $line =~ HCU\[[0-9]+\] ]]; then
            current_id=$(echo "$line" | grep -o 'HCU\[[0-9]\+\]')
        fi
        if [[ $line =~ "Total Memory" && -n "$current_id" ]]; then
            gpus["$current_id,mem_total"]=$(echo "$line" | awk '{print $NF}')
        fi
        if [[ $line =~ "Total Used Memory" && -n "$current_id" ]]; then
            gpus["$current_id,mem_used"]=$(echo "$line" | awk '{print $NF}')
        fi
    done <<< "$mem_output"
}

get_mem_bandwidth() {
    local clocks_output
    clocks_output=$("$ROCM_SMI" --showclocks)
    local current_id=""
    while read -r line; do
        if [[ $line =~ HCU\[[0-9]+\] ]]; then
            current_id=$(grep -o 'HCU\[[0-9]\+\]' <<< "$line")
        fi
        if [[ $line =~ mclk && -n $current_id ]]; then
            local clock_mhz
            clock_mhz=$(echo "$line" | grep -o '[0-9]\+Mhz' | sed 's/Mhz//')
            if [[ -n "$clock_mhz" ]]; then
                # HBM2, 4096bit width, DDR
                local bandwidth
                bandwidth=$(echo "scale=2; ($clock_mhz * 2 * 4096 / 8) / 1000" | bc)
                gpus["$current_id,mem_bw"]="$bandwidth"
            fi
        fi
    done <<< "$clocks_output"
}

get_process_info() {
    # Initialize all GPUs with '无' for processes.
    for id in $(for key in "${!gpus[@]}"; do if [[ $key == *,name ]]; then echo "${gpus[$key]}"; fi; done); do
        gpus["$id,procs"]="无"
    done

    local pids_output
    pids_output=$("$ROCM_SMI" --showpids -P)
    
    local current_pid=""
    while read -r line; do
        # Capture the PID when a new process block starts
        if [[ $line =~ ^PID:[[:space:]]+([0-9]+) ]]; then
            current_pid=${BASH_REMATCH[1]}
        
        # When HCU Index is found for the current PID, associate them
        elif [[ -n "$current_pid" && $line =~ HCU\ Index:[[:space:]]*\[\'([0-9]+)\'\] ]]; then
            local hcu_index=${BASH_REMATCH[1]}
            local id="HCU[$hcu_index]"
            
            # Ensure we are tracking this GPU
            if [[ -v "gpus[$id,name]" ]]; then
                local proc_name
                proc_name=$(ps -o comm= -p "$current_pid" 2>/dev/null || echo "pid-$current_pid")
                local proc_info="$current_pid:$proc_name"

                if [[ ${gpus["$id,procs"]} == "无" ]]; then
                    gpus["$id,procs"]="$proc_info"
                else
                    gpus["$id,procs"]+=", $proc_info"
                fi
            fi
            # Reset current_pid after processing to avoid mis-association with processes that have no HCU index
            current_pid=""
        
        # Reset current_pid if we reach the end of a block (an empty line often separates them)
        elif [[ -n "$current_pid" && -z "$line" ]]; then
             current_pid=""
        fi
    done <<< "$pids_output"
}

get_interconnect_bandwidth() {
    if ! "$RUN_INTERCONNECT_TEST"; then
        return
    fi
    echo -e "${YELLOW}正在运行卡间带宽测试，这可能需要几分钟...${NC}"
    
    local bw_output
    bw_output=$($ROCM_BANDWIDTH_TEST 2>/dev/null)
    
    local -A gpu_map
    local -a gpu_indices
    local smi_devices=$("$ROCM_SMI" --showhw | grep -c 'HCU')
    
    while read -r line; do
        if [[ $line =~ ^[[:space:]]*Device:[[:space:]]*([0-9]+),[[:space:]]*K100_AI ]]; then
            gpu_indices+=("${BASH_REMATCH[1]}")
        fi
    done <<< "$bw_output"

    for (( i=0; i<smi_devices; i++ )); do
        if [ -n "${gpu_indices[$i]}" ]; then
            gpu_map["${gpu_indices[$i]}"]="HCU[$i]"
        fi
    done
    
    local matrix_started=false
    local -A raw_bw
    
    while read -r line; do
        if [[ $line =~ "Unidirectional copy peak bandwidth GB/s" ]]; then
            matrix_started=true
            continue
        fi
        if ! $matrix_started || [[ -z $line ]] || [[ $line =~ D/D ]]; then
            continue
        fi

        local -a fields=($line)
        local row_idx=${fields[0]}
        
        if [[ -n "${gpu_map[$row_idx]}" ]]; then
            for (( i=0; i<smi_devices; i++ )); do
                local col_idx=${gpu_indices[$i]}
                local bw_val=${fields[i+1]}
                if [[ "$row_idx" != "$col_idx" && "$bw_val" != "N/A" ]]; then
                     raw_bw["$row_idx"]+="$bw_val "
                fi
            done
        fi
    done <<< "$bw_output"
    
    for dev_idx in "${!raw_bw[@]}"; do
        local hcu_id=${gpu_map[$dev_idx]}
        local -a values=(${raw_bw[$dev_idx]})
        local sum=0
        local count=0
        for val in "${values[@]}"; do
            sum=$(echo "$sum + $val" | bc)
            ((count++))
        done
        if (( count > 0 )); then
            local avg
            avg=$(echo "scale=2; $sum / $count" | bc)
            gpus["$hcu_id,interconnect_bw"]="$avg"
        fi
    done
}

# --- 渲染模块 ---
render_table() {
    # 标题
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                             DCU 监控面板                                                  ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝${NC}"
    
    # 时间戳
    echo -e "${CYAN}监控时间: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo ""
    
    # 构建表头，第一列加宽以适应"总计"
    local header_top="┌──────┬──────────┬───────────────────┬─────────┬────────┬──────────────────┬──────────────────────────"
    local header_mid1="│  GPU │   型号   │      显存(MiB)    │ 使用率  │ 温度   │ 显存带宽(GB/s)   │ 进程 (PID:名称)          "
    local header_mid2="│      │          │    (已用/总量)    │ (GPU %) │ (°C)   │     (理论)       │                          "
    local header_bottom="├──────┼──────────┼───────────────────┼─────────┼────────┼──────────────────┼──────────────────────────"
    local footer="└──────┴──────────┴───────────────────┴─────────┴────────┴──────────────────┴──────────────────────────"
    
    if "$RUN_INTERCONNECT_TEST"; then
        header_top+="┬────────────────┐"
        header_mid1+="│ 互联带宽(G/s)  │"
        header_mid2+="│ (Avg Unidir)   │"
        header_bottom+="┼────────────────┤"
        footer+="┴────────────────┘"
    else
        header_top+="┐"
        header_mid1+="│"
        header_mid2+="│"
        header_bottom+="┤"
        footer+="┘"
    fi

    echo -e "${YELLOW}DCU设备状态:${NC}"
    echo -e "$header_top"
    echo -e "$header_mid1"
    echo -e "$header_mid2"
    echo -e "$header_bottom"

    # 渲染数据行
    local gpu_ids
    mapfile -t gpu_ids < <(for key in "${!gpus[@]}"; do if [[ $key == *,name ]]; then echo "${gpus[$key]}"; fi; done | sort -V)
    
    for id in "${gpu_ids[@]}"; do
        local model=${gpus["$id,model"]:-N/A}
        local mem_used=${gpus["$id,mem_used"]:-0}
        local mem_total=${gpus["$id,mem_total"]:-0}
        local util=${gpus["$id,util"]:-0%}
        local temp=${gpus["$id,temp"]:-0.0C}
        local mem_bw=${gpus["$id,mem_bw"]:-N/A}
        local procs=${gpus["$id,procs"]:-无}
        
        local mem_str="$mem_used / $mem_total"
        
        local line
        line=$(printf "│ %-4s │ %-8s │ %-17s │ %-7s │ %-6s │ %-16s │ %-24s " \
            "${id//HCU/}" \
            "$model" \
            "$mem_str" \
            "$util" \
            "${temp%C}" \
            "$mem_bw" \
            "$procs")
        
        if "$RUN_INTERCONNECT_TEST"; then
            local interconnect_bw=${gpus["$id,interconnect_bw"]:-N/A}
            line+=$(printf "│ %-14s │" "$interconnect_bw")
        else
            line+="│"
        fi
        echo "$line"
    done
    
    # --- 计算并渲染总计行 ---
    local total_mem_used=0
    local total_mem_total=0
    local temp_sum=0.0
    local util_sum=0.0
    local mem_bw_sum=0.0
    local gpu_count=${#gpu_ids[@]}
    local total_procs=0

    for id in "${gpu_ids[@]}"; do
        ((total_mem_used += ${gpus["$id,mem_used"]:-0}))
        ((total_mem_total += ${gpus["$id,mem_total"]:-0}))
        
        local temp_val=${gpus["$id,temp"]%C}
        temp_sum=$(echo "$temp_sum + $temp_val" | bc)
        
        local util_val=${gpus["$id,util"]%?}
        util_sum=$(echo "$util_sum + $util_val" | bc)

        local mem_bw_val=${gpus["$id,mem_bw"]:-0}
        mem_bw_sum=$(echo "$mem_bw_sum + $mem_bw_val" | bc)

        local procs_str=${gpus["$id,procs"]:-无}
        if [[ "$procs_str" != "无" ]]; then
            local num_procs=$(echo "$procs_str" | grep -o ',' | wc -l)
            ((total_procs += num_procs + 1))
        fi
    done

    local avg_temp="N/A"
    local avg_util="N/A"
    local avg_mem_bw="N/A"
    if (( gpu_count > 0 )); then
        avg_temp=$(printf "%.1f" "$(echo "scale=2; $temp_sum / $gpu_count" | bc)")
        avg_util=$(printf "%.1f" "$(echo "scale=2; $util_sum / $gpu_count" | bc)")%
        avg_mem_bw=$(printf "%.2f" "$(echo "scale=2; $mem_bw_sum / $gpu_count" | bc)")
    fi

    local total_mem_str="$total_mem_used / $total_mem_total"
    local total_procs_str="$total_procs 个进程"
    
    echo -e "$header_bottom"
    
    local total_line
    total_line=$(printf "│ %-4s │ %-8s │ %-17s │ %-7s │ %-6s │ %-16s │ %-24s " \
        "总计" "($gpu_count GPUs)" "$total_mem_str" "$avg_util" "$avg_temp" "$avg_mem_bw" "$total_procs_str")
    
    if "$RUN_INTERCONNECT_TEST"; then
        total_line+=$(printf "│ %-14s │" "N/A")
    else
        total_line+="│"
    fi
    echo "$total_line"
    
    echo -e "$footer"
}

render_usage_hints() {
    echo ""
    echo -e "${YELLOW}--- 使用提示 ---${NC}"
    echo -e "  > 如需持续刷新监控，请使用 ${CYAN}-w${NC} 或 ${CYAN}--watch${NC} 参数。"
    echo -e "    示例: ${GREEN}$0 -w -t 5${NC}"
    
    if ! $RUN_INTERCONNECT_TEST; then
        echo -e "  > 如需进行卡间带宽测试 (此操作耗时较长)，请添加 ${CYAN}-i${NC} 参数。"
        echo -e "    示例: ${GREEN}$0 -i${NC}"
    fi
    
    echo -e "  > 查看所有可用选项，请使用 ${CYAN}-h${NC} 或 ${CYAN}--help${NC}。"
    echo ""
    echo -e "${YELLOW}* 进程信息查看提示:${NC}"
    echo -e "  如果你想查看启动进程的完整命令行（包括参数），可以运行："
    echo -e "  ${GREEN}cat /proc/<PID>/cmdline${NC}"
}


# --- 主逻辑 ---
main() {
    check_requirements
    
    run_monitor() {
        start_spinner "正在采集DCU数据"
        # 清空旧数据
        unset gpus
        declare -A gpus

        # 采集数据
        get_gpu_list
        get_base_info
        get_mem_info
        get_mem_bandwidth
        get_process_info
        stop_spinner

        # 互联带宽测试独立于spinner，因为它有自己的提示信息且耗时很长
                get_interconnect_bandwidth

        # 渲染输出
        if "$WATCH_MODE"; then clear; fi
        render_table
    }

    trap 'stop_spinner; echo -e "\n监控已终止。"; exit 0' SIGINT

    if "$WATCH_MODE"; then
        while true; do
            run_monitor
            sleep "$INTERVAL"
        done
    else
        run_monitor
        render_usage_hints
    fi
}

main 