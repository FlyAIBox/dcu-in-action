#!/bin/bash
# 如果脚本不是在 bash 下运行，则自动切换到 bash
if [ -z "$BASH_VERSION" ]; then
    exec /usr/bin/env bash "$0" "$@"
fi
# DCU监控脚本
# 实时显示DCU的使用情况(型号、已用显存/总显存、使用率、温度、PCIe带宽、卡间互联带宽 )；使用DCU加速卡的应用信息；模型微调进度

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 确定 smi 命令路径
SMI_CMD=""
if command -v hy-smi &> /dev/null; then
    SMI_CMD="$(command -v hy-smi)"
elif command -v rocm-smi &> /dev/null; then
    SMI_CMD="$(command -v rocm-smi)"
elif [ -x "/opt/hyhal/bin/hy-smi" ]; then
    SMI_CMD="/opt/hyhal/bin/hy-smi"
elif [ -x "/opt/rocm/bin/rocm-smi" ]; then # 通用rocm路径
    SMI_CMD="/opt/rocm/bin/rocm-smi"
else
    echo -e "${RED}错误: 未找到 hy-smi 或 rocm-smi 命令，请确认已正确安装并配置${NC}"
    exit 1
fi

# 检查jq是否可用
if ! command -v jq &> /dev/null; then
    echo -e "${RED}错误: 未找到jq命令，请使用 'apt install jq' 或 'yum install jq' 安装后重试${NC}"
    exit 1
fi

# 检查SMI命令是否支持--json选项
if ! $SMI_CMD --json --help &>/dev/null && ! $SMI_CMD --help 2>&1 | grep -q -- "--json"; then
    echo -e "${YELLOW}警告: $SMI_CMD 可能不支持 --json 选项，将使用备用方法获取数据${NC}"
    # 设置标志以便后续使用备用方法
    USE_FALLBACK=1
else
    USE_FALLBACK=0
fi

# 清屏函数
clear_screen() {
    printf "\033c"
}

# 获取DCU信息
get_dcu_info() {
    # 如果需要使用备用方法，则解析文本输出而非JSON
    if [ "$USE_FALLBACK" -eq 1 ]; then
        # 备用方法：解析文本输出
        local card_info=()
        local card_count=0
        
        # 获取卡的基本信息 (海光DCU专用格式)
        local basic_info=$($SMI_CMD 2>/dev/null)
        if [ -n "$basic_info" ]; then
            # 解析hy-smi的标准输出格式
            while read -r line; do
                if [[ "$line" =~ ^([0-9]+)[[:space:]]+([0-9.]+)C[[:space:]]+([0-9.]+)W[[:space:]]+([a-z]+)[[:space:]]+([0-9.]+)W[[:space:]]+([0-9]+)%[[:space:]]+([0-9.]+)%[[:space:]]+([A-Za-z]+)[[:space:]]*$ ]]; then
                    local idx="${BASH_REMATCH[1]}"
                    local temp="${BASH_REMATCH[2]%%.*}"  # 去掉小数部分
                    local util="${BASH_REMATCH[7]%%.*}"  # 去掉小数部分
                    local model="DCU"  # 默认型号
                    
                    # 检查是否已有这个卡的信息
                    if [[ -z "${card_info[$idx]}" ]]; then
                        card_info[$idx]="$idx,$model,0,0,$util,$temp,0/0,N/A"
                        card_count=$((card_count+1))
                    else
                        # 更新已有信息
                        IFS=',' read -r i m u t ut te pb lb <<< "${card_info[$idx]}"
                        card_info[$idx]="$i,$m,$u,$t,$util,$temp,$pb,$lb"
                    fi
                fi
            done < <(echo "$basic_info" | grep -v "=" | grep -v "HCU" | grep -v "End of SMI" | grep -v "System Management")
        fi
        
        # 如果没有找到卡信息，尝试rocm-smi的其他输出格式
        if [ $card_count -eq 0 ]; then
            # 尝试直接获取GPU数量
            local gpu_count=$($SMI_CMD -i 2>/dev/null | grep -c "GPU\|HCU")
            if [ $gpu_count -gt 0 ]; then
                for ((i=0; i<$gpu_count; i++)); do
                    card_info[$i]="$i,DCU,0,0,0,0,0/0,N/A"
                    card_count=$((card_count+1))
                done
            fi
        fi
        
        # 如果仍然没有卡信息，创建一个占位符
        if [ $card_count -eq 0 ]; then
            card_info[0]="0,未检测到DCU,0,0,0,0,0/0,N/A"
            card_count=1
        fi
        
        # 获取显存使用情况
        local mem_info=$($SMI_CMD --showmeminfo vram 2>/dev/null)
        if [ -n "$mem_info" ]; then
            while read -r line; do
                if [[ "$line" =~ (HCU|GPU)\[([0-9]+)\].*vram\ Total\ Memory\ \(MiB\):\ +([0-9]+) ]]; then
                    local idx="${BASH_REMATCH[2]}"
                    local total="${BASH_REMATCH[3]}"
                    # 更新卡信息，保留原有信息，更新总显存
                    if [[ -n "${card_info[$idx]}" ]]; then
                        IFS=',' read -r i m u t ut te pb lb <<< "${card_info[$idx]}"
                        card_info[$idx]="$i,$m,$u,$total,$ut,$te,$pb,$lb"
                    fi
                elif [[ "$line" =~ (HCU|GPU)\[([0-9]+)\].*vram\ Total\ Used\ Memory\ \(MiB\):\ +([0-9]+) ]]; then
                    local idx="${BASH_REMATCH[2]}"
                    local used="${BASH_REMATCH[3]}"
                    # 更新卡信息，保留原有信息，更新已用显存
                    if [[ -n "${card_info[$idx]}" ]]; then
                        IFS=',' read -r i m u t ut te pb lb <<< "${card_info[$idx]}"
                        card_info[$idx]="$i,$m,$used,$t,$ut,$te,$pb,$lb"
                    fi
                fi
            done < <(echo "$mem_info")
        fi
        
        # 获取PCIe带宽
        local pcie_data=$($SMI_CMD --showbw 2>/dev/null)
        if [ -n "$pcie_data" ]; then
            for idx in "${!card_info[@]}"; do
                local sent=0
                local recv=0
                while read -r line; do
                    if [[ "$line" =~ (HCU|GPU)\[$idx\].*PCIe\ Bus\ Bandwidth\ Sent:\ +([0-9.]+)\ (MiB|MB)/s ]]; then
                        sent=$(echo "${BASH_REMATCH[2]} / 1024" | bc -l 2>/dev/null || echo "0")
                        sent=$(printf "%.0f" $sent)
                    elif [[ "$line" =~ (HCU|GPU)\[$idx\].*PCIe\ Bus\ Bandwidth\ Received:\ +([0-9.]+)\ (MiB|MB)/s ]]; then
                        recv=$(echo "${BASH_REMATCH[2]} / 1024" | bc -l 2>/dev/null || echo "0")
                        recv=$(printf "%.0f" $recv)
                    fi
                done < <(echo "$pcie_data")
                
                # 更新卡信息
                if [[ -n "${card_info[$idx]}" ]]; then
                    IFS=',' read -r i m u t ut te pb lb <<< "${card_info[$idx]}"
                    card_info[$idx]="$i,$m,$u,$t,$ut,$te,$sent/$recv,$lb"
                fi
            done
        fi
        
        # 输出所有卡信息
        for i in "${!card_info[@]}"; do
            echo "${card_info[$i]}"
        done
        return
    fi

    # 标准JSON方法
    # hy-smi可能不支持一次性查询多个信息并以JSON格式输出，
    # 因此改为多次调用，用jq合并结果。
    local json_base json_use json_temp json_mem json_bw
    json_base=$("$SMI_CMD" --showproductname --json 2>/dev/null)
    json_use=$("$SMI_CMD" --showuse --json 2>/dev/null)
    json_temp=$("$SMI_CMD" --showtemp --json 2>/dev/null)
    json_mem=$("$SMI_CMD" --showmeminfo vram --json 2>/dev/null)
    json_bw=$("$SMI_CMD" --showbw --json 2>/dev/null)
    
    local json_link="{}" # Default to empty JSON
    if [[ "$SMI_CMD" == *hy-smi ]]; then
        # hy-smi specific command for interconnect bandwidth
        json_link=$("$SMI_CMD" --showlink --json 2>/dev/null)
    fi

    # 检查JSON格式是否有效
    for json_var in "$json_base" "$json_use" "$json_temp" "$json_mem" "$json_bw" "$json_link"; do
        if [ -n "$json_var" ] && ! echo "$json_var" | jq -e . >/dev/null 2>&1; then
            # 如果JSON无效，切换到备用方法
            USE_FALLBACK=1
            return $(get_dcu_info)
        fi
    done

    if [ -z "$json_base" ]; then
        echo "0,未检测到DCU,0,0,0,0,0/0,N/A"
        return
    fi

    # 尝试使用jq处理JSON，如果失败则返回默认值
    (jq -s -r '
      reduce .[] as $item ({}; . * $item) |
      keys_unsorted | .[] |
      . as $card_key |
      ( $card_key | ltrimstr("card") ) as $index |
      .[$card_key] |
      (
        # PCIe bandwidth (convert MB/s to GB/s)
        # rocm-smi uses "MB/s", hy-smi uses "MiB/s"
        ( .["PCIe Bandwidth (sent)"] // "0" | gsub(" MiB/s| MB/s"; "") | tonumber / 1024 ) as $pcie_sent_gb |
        ( .["PCIe Bandwidth (received)"] // "0" | gsub(" MiB/s| MB/s"; "") | tonumber / 1024 ) as $pcie_recv_gb |
        "\( ($pcie_sent_gb | round) )/\( ($pcie_recv_gb | round) )"
      ) as $pcie_bw |
      (
        # Interconnect bandwidth for hy-smi (sum of all links)
        if .["HCCL link"] then
           ( .["HCCL link"] | values | map(.["BW"] | gsub(" GB/s"; "") | tonumber) | add )
        else
           "N/A"
        end
      ) as $link_bw_gb |
      [
        $index,
        .["Card series"] // "N/A",
        ( .["VRAM Used Memory (B)"] // 0 | tonumber / 1024 / 1024 | floor ),
        ( .["VRAM Total Memory (B)"] // 0 | tonumber / 1024 / 1024 | floor ),
        .["GPU use (%)"] // "0",
        ( .["Temperature (C)"] // .["Temperature (Sensor 1) (C)"] // "0.0" | split(".") | .[0] ),
        $pcie_bw,
        $link_bw_gb
      ] | join(",")
    ' <(echo "$json_base") <(echo "$json_use") <(echo "$json_temp") <(echo "$json_mem") <(echo "$json_bw") <(echo "$json_link") 2>/dev/null) || echo "0,解析错误,0,0,0,0,0/0,N/A"
}

# 获取在GPU上运行的进程信息
get_gpu_processes() {
    # 如果需要使用备用方法
    if [ "$USE_FALLBACK" -eq 1 ]; then
        # 使用ps命令获取可能的GPU进程
        ps aux | grep -E "python|deepspeed|llamafactory|train|inference" | grep -v grep | head -5 | \
        awk '{printf "可能的GPU进程 | PID: %-8s | CPU: %5s%% | MEM: %5s%% | CMD: %s\n", $2, $3, $4, substr($0, index($0,$11))}'
        return
    fi

    local pids_json
    pids_json=$($SMI_CMD --showpids --json 2>/dev/null)
    
    # 检查JSON格式是否有效
    if [ -z "$pids_json" ] || [ "$pids_json" == "{}" ] || ! echo "$pids_json" | jq -e . >/dev/null 2>&1; then
        # 如果JSON无效，切换到备用方法
        USE_FALLBACK=1
        get_gpu_processes
        return
    fi
    
    # 安全地处理JSON
    (echo "$pids_json" | jq -r '
      if type == "object" and (keys | length) > 0 then
        keys_unsorted | .[] | 
        . as $card_key | 
        ( $card_key | ltrimstr("card") ) as $index |
        (.[$card_key] | if type == "array" then .[] else . end) | # Handle both single object and array of objects for processes
        "GPU \($index) | PID: \(.["Process ID"] // "N/A") | 显存: \(.["VRAM usage"] // "N/A") | CMD: \(.["Process Name"] // "N/A")"
      else
        "无GPU计算进程"
      end
    ' 2>/dev/null) || echo "无法解析GPU进程信息"
}

# 获取训练日志最新信息
get_training_info() {
    local log_file="logs/llm-fine-tuning/train.log"
    if [ -f "$log_file" ]; then
        # 获取最新的训练步数和损失
        local latest_info=$(tail -n 50 "$log_file" | grep -E "step|loss" | tail -n 1)
        echo "$latest_info"
    else
        echo "等待训练开始..."
    fi
}

# 格式化显存大小
format_memory() {
    local mem=$1
    if [ "$mem" -gt 1024 ]; then
        echo "$(($mem / 1024))GB"
    else
        echo "${mem}MB"
    fi
}

# 主监控循环
monitor_loop() {
    while true; do
        clear_screen
        
        # 标题
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║                           DCU 监控面板                          ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        
        # 时间戳
        echo -e "${CYAN}监控时间: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
        echo ""
        
        # DCU状态表格
        echo -e "${YELLOW}DCU设备状态:${NC}"
        echo -e "┌─────┬──────────────┬─────────────────┬──────────┬────────┬──────────────────┬─────────────┐"
        echo -e "│ GPU │     型号     │      显存       │  使用率  │  温度  │ PCIe 带宽(G/s) │ 互联带宽(G/s)│"
        echo -e "├─────┼──────────────┼─────────────────┼──────────┼────────┼──────────────────┼─────────────┤"
        
        # 获取并显示每个DCU的信息
        total_mem_used=0
        total_mem_total=0
        avg_util=0
        max_temp=0
        card_count=0
        
        while IFS=',' read -r index name mem_used mem_total util temp pcie_bw link_bw; do
            # 累计统计
            card_count=$((card_count + 1))
            total_mem_used=$((total_mem_used + mem_used))
            total_mem_total=$((total_mem_total + mem_total))
            avg_util=$((avg_util + util))
            # Temperature might be N/A
            if [[ "$temp" =~ ^[0-9]+$ ]] && [ "$temp" -gt "$max_temp" ]; then
                max_temp=$temp
            fi
            
            # 格式化显示
            mem_display="$(format_memory $mem_used)/$(format_memory $mem_total)"
            
            # 根据使用率显示不同颜色
            if [ "$util" -gt 80 ]; then
                util_color=$GREEN
            elif [ "$util" -gt 50 ]; then
                util_color=$YELLOW
            else
                util_color=$RED
            fi
            
            # 根据温度显示不同颜色
            if [[ "$temp" =~ ^[0-9]+$ ]] && [ "$temp" -gt 85 ]; then
                temp_color=$RED
            elif [[ "$temp" =~ ^[0-9]+$ ]] && [ "$temp" -gt 75 ]; then
                temp_color=$YELLOW
            else
                temp_color=$GREEN
            fi
            
            printf "│ %-3s │ %-12s │ %-15s │ ${util_color}%6s%%${NC}  │ ${temp_color}%4s°C${NC} │ %-16s │ %-12s│\n" \
                   "$index" "${name:0:12}" "$mem_display" "$util" "$temp" "$pcie_bw" "$link_bw"
        done < <(get_dcu_info)
        
        echo -e "└─────┴──────────────┴─────────────────┴──────────┴────────┴──────────────────┴─────────────┘"
        
        # 计算平均值
        if [ "$card_count" -gt 0 ]; then
             avg_util=$((avg_util / card_count))
        else
             avg_util=0
        fi
        
        # 汇总信息
        echo ""
        echo -e "${CYAN}汇总信息:${NC}"
        echo -e "总显存使用: $(format_memory $total_mem_used) / $(format_memory $total_mem_total)"
        echo -e "平均使用率: ${avg_util}%"
        echo -e "最高温度: ${max_temp}°C"
        
        # 训练进度信息
        echo ""
        echo -e "${YELLOW}训练进度:${NC}"
        training_info=$(get_training_info)
        if [ -n "$training_info" ]; then
            echo -e "$training_info"
        fi
        
        # 系统资源信息
        echo ""
        echo -e "${CYAN}系统资源:${NC}"
        # CPU使用率
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
        echo -e "CPU使用率: ${cpu_usage}%"
        
        # 内存使用
        mem_info=$(free -h | grep "Mem:" | awk '{print $3 " / " $2}')
        echo -e "内存使用: $mem_info"
        
        # 进程信息
        echo ""
        echo -e "${YELLOW}相关进程:${NC}"
        get_gpu_processes
        
        # 提示信息
        echo ""
        echo -e "${GREEN}提示: 按 Ctrl+C 退出监控${NC}"
        
        # 刷新间隔
        sleep 2
    done
}

# 捕获Ctrl+C信号
trap 'echo -e "\n${YELLOW}监控已停止${NC}"; exit 0' INT

# 启动监控
echo -e "${GREEN}启动DCU训练监控...${NC}"
monitor_loop 