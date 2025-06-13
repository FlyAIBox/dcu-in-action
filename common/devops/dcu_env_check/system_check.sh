#!/bin/bash
#set -x
# 默认参数配置
DEFAULT_OUTPUT_DIR="system_info_$(date +%Y%m%d_%H%M%S)"
DEFAULT_KEYWORD="hydcu"
DEFAULT_LOG_AGE=24  # 小时
DEFAULT_LOG_SIZE_LIMIT=10  # 单位：MB
QUIET_MODE=0
DEBUG_MODE=0

DEVICE_NAME=""
DEVICE_ID=""


# 显示帮助信息
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "系统诊断脚本 - 收集系统信息并检测驱动问题"
    echo
    echo "选项："
    echo "  -o DIR      指定输出目录 (默认: 自动生成)"
    echo "  -k KEYWORD  设置检测关键字 (默认: $DEFAULT_KEYWORD)"
    echo "  -t HOURS    收集日志的时间范围(小时) (默认: 24)"
    echo "  -s SIZE     日志文件大小限制(MB) (默认: 10)"
    echo "  -q          静默模式(仅输出错误)"
    echo "  -d          调试模式"
    echo "  -h          显示此帮助信息"
    echo
    echo "示例："
    echo "  $0 -o /tmp/logs -k mydriver -t 48"
}

# 解析参数
while getopts "o:k:t:s:qdh" opt; do
    case $opt in
        o) CUSTOM_OUTPUT_DIR="$OPTARG" ;;
        k) KEYWORD="$OPTARG" ;;
        t) LOG_AGE="$OPTARG" ;;
        s) LOG_SIZE_LIMIT="$OPTARG" ;;
        q) QUIET_MODE=1 ;;
        d) DEBUG_MODE=1; set -x ;;
        h) show_help; exit 0 ;;
        \?) echo "无效选项: -$OPTARG" >&2; exit 1 ;;
        :) echo "选项 -$OPTARG 需要参数" >&2; exit 1 ;;
    esac
done

# 设置默认值
: ${OUTPUT_DIR:=${CUSTOM_OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}}
: ${KEYWORD:=$DEFAULT_KEYWORD}
: ${LOG_AGE:=$DEFAULT_LOG_AGE}
: ${LOG_SIZE_LIMIT:=$DEFAULT_LOG_SIZE_LIMIT}


init_check() {
    local pkgs_debian=(dmidecode lshw pciutils numactl-devel)
    local pkgs_centos=(dmidecode lshw pciutils numactl-dev)
    local cmd=(dmidecode lshw lspci numactl )

    for ((i=0; i<${#cmd[@]}; i++)); do
        if ! command -v ${cmd[i]} &>/dev/null; then
            if command -v apt-get &>/dev/null; then
                echo "没有${cmd[i]} 命令，请先安装${pkgs_debian[i]}"
                apt-get install -y ${pkgs_debian[i]}
            elif command -v yum &>/dev/null; then
                echo "没有${cmd[i]} 命令，请先安装${pkgs_centos[i]}"
                yum install -y ${pkgs_centos[i]}
            fi
        fi
    done
}
    

declare -A devices_id=(
    ["Z100"]="54b7"
    ["Z100L"]="55b7"
    ["K100"]="62b7"
    ["K100-AI"]="6210"
    ["K100-AI-ECO"]="6211"
    ["BW1000"]="6320"
)

# 构建反向映射表（设备ID → 设备名称）
declare -A devices
for name in "${!devices_id[@]}"; do
    id="${devices_id[$name]}"
    devices["${id}"]+=" $name" 
done

get_dcu() {
    # 获取设备ID列表
    mapfile -t dcu_list < <(lspci -nn | grep -i -E "display|co-processor" | awk -F'[][]' '{print $4}' | awk -F ":" '{print $2}')

    local index=0
    local dcu_num=0
    local total=${#dcu_list[@]}

    while [ $index -lt $total ]; do
        current_id="${dcu_list[$index]}"

        if [ -n "${devices[$current_id]}" ]; then
            echo "dcu #$dcu_num 型号为：${devices[$current_id]}"
            ((dcu_num++))
        else
            echo "未知设备ID: $current_id" >&2
        fi

        ((index++))
    done
	echo "总计: $dcu_num张${devices[$current_id]} DCU 设备"
	DEVICE_NAME=${devices[$current_id]}
	DEVICE_ID=$current_id
	# echo $DEVICE_NAME $DEVICE_ID
}



# 日志函数
log() {
    [ $QUIET_MODE -eq 0 ] && echo "$@"
}

hline() {
    printf "%0.s=" {1..80}
}

# 初始化目录
mkdir -p "$OUTPUT_DIR" || exit 1

# 带大小限制的日志复制函数
copy_log_with_limit() {
    local src=$1
    local dest=$2
    local size_limit_mb=$3
    
    if [ -f "$src" ]; then
        file_size=$(du -m "$src" | cut -f1)
        if [ $file_size -gt $size_limit_mb ]; then
            log "跳过大文件: $src (${file_size}MB > ${size_limit_mb}MB)"
            echo "[日志文件超过大小限制未采集]" > "$dest"
            return
        fi
        cp "$src" "$dest" 2>/dev/null || echo "无权限读取日志" > "$dest"
    else
        echo "日志文件不存在" > "$dest"
    fi
}

echoAndRun(){
    hline
    echo
    echo "[root@dcu ~]# "$1;
    eval $1 ;
    echo;
}
# 收集系统信息
collect_system_info() {
    log "收集CPU信息..."
    echoAndRun "lscpu" > "$OUTPUT_DIR/cpuinfo.txt" 2>&1

    log "收集内存信息..."
    echoAndRun "free -h" > "$OUTPUT_DIR/meminfo.txt" 2>&1
    echoAndRun "dmidecode -t memory" >> "$OUTPUT_DIR/meminfo.txt" 2>&1

    log "收集网络信息..."
    echoAndRun "ip a" > "$OUTPUT_DIR/network.txt" 2>&1
    echoAndRun "lspci -nn | grep -i eth" >> "$OUTPUT_DIR/network.txt" 2>&1

    log "收集系统版本..."
    echoAndRun "cat /etc/os-release" > "$OUTPUT_DIR/os_info.txt" 2>&1
    echoAndRun "uname -a" >> "$OUTPUT_DIR/os_info.txt" 2>&1
	echoAndRun "cat /proc/cmdline" >> "$OUTPUT_DIR/os_info.txt" 2>&1
    echoAndRun "numactl -H" >> "$OUTPUT_DIR/os_info.txt" 2>&1
    echoAndRun "rpm -qf $(which ldd)" >> "$OUTPUT_DIR/os_info.txt" 2>&1
    echoAndRun "ldd --version" >> "$OUTPUT_DIR/os_info.txt" 2>&1
    echoAndRun "strings $(find /usr/ -name libc.so.6)  | grep ^GLIBC_" >> "$OUTPUT_DIR/os_info.txt" 2>&1
    echoAndRun "strings $(find /usr -name libstdc++.so.6)  | grep GLIBCXX" >> "$OUTPUT_DIR/os_info.txt" 2>&1
    # echoAndRun "rpm -qi $(rpm -qf $(which ldd))"  >> "$OUTPUT_DIR/os_info.txt" 2>&1

    log "收集服务器信息..."
    echoAndRun "ipmitool fru" > "$OUTPUT_DIR/hardware.txt" 2>&1
    echoAndRun "ipmitool mc info" >> "$OUTPUT_DIR/hardware.txt" 2>&1
    echoAndRun "dmidecode -s system-product-name" >> "$OUTPUT_DIR/hardware.txt" 2>&1
    echoAndRun "dmidecode -t bios" >> "$OUTPUT_DIR/hardware.txt" 2>&1
    

}

# 收集系统日志
collect_logs() {
    log "收集系统日志(最近${LOG_AGE}小时)..."
    
    # 识别系统日志位置
    local syslog_path
    [ -f /var/log/syslog ] && syslog_path=/var/log/syslog
    [ -f /var/log/messages ] && syslog_path=/var/log/messages

    if [ -n "$syslog_path" ]; then
        copy_log_with_limit "$syslog_path" "$OUTPUT_DIR/system.log" $LOG_SIZE_LIMIT
    else
        log "收集journalctl日志..."
        journalctl --since "${LOG_AGE} hours ago" > "$OUTPUT_DIR/system.log" 2>/dev/null || \
        echo "无法获取系统日志" > "$OUTPUT_DIR/system.log"
    fi

    log "收集dmesg日志..."
    dmesg -T > "$OUTPUT_DIR/dmesg.log" 2>&1
}

# 收集pcie信息
parse_regions() {
    lspci -vv -s "$1" | awk '/Region [0-9]+:/'
}

get_pcie_topo() {
    lspci -vt
}

show_acs() {
    lspci -vvs "$1" | grep ACS
}

show_link_status() {
    local info=$(lspci -vv -s "$1")

    declare -A lnk=(
        [cur_speed]=$(grep -ioP 'LnkSta:\s+Speed\s\K[\d.]+' <<< "$info")
        [max_speed]=$(grep -ioP 'LnkCap:\s+Speed\s\K[\d.]+' <<< "$info")
        [cur_width]=$(grep -ioP 'LnkSta.*Width\sx\K\d+' <<< "$info")
        [max_width]=$(grep -ioP 'LnkCap.*Width\sx\K\d+' <<< "$info")
    )
    echo "当前状态  : x${lnk[cur_width]} @ ${lnk[cur_speed]}GT/s "
}
show_busmaster() {
    lspci -vv -s "$1" | grep BusMaster | awk '{print $4}'
}
collect_pcie_logs() {
	log "收集PCIe系统信息"
	lspci -D -d :$DEVICE_ID | while read -r dev; do
		id=${dev:0:12}
        name=${dev:12}
        echo
        echo "$(hline)"
        echo "设备 ${id}"
        echo "型号      : ${name}"
        echo "$(hline)"
        echo
        echo "BAR 内存映射："
        parse_regions "$id" | sed 's/^/  /'
        echo
        echo "PCIe 链路状态："
        show_link_status "$id" | sed 's/^/  /'
        echo
        echo "PCIe ACS设置"
        show_acs "$id"  | sed 's/^/  /'
        echo
        echo "BusMaster设置"
        show_busmaster "$id"  | sed 's/^/  /'
    done
	
}

get_pcie_info() {
	log "收集PCIe系统信息"
	collect_pcie_logs >  $OUTPUT_DIR/pcie_info.log
	get_pcie_topo > $OUTPUT_DIR/pcie_vt.log 2>&1
	lspci -vvv > $OUTPUT_DIR/pcie_more.log 2>&1
	log "PCIe 信息收集完毕"
}

analyze_regions() {
    local address
    echo "$1"

    # 提取Region关键参数
	address=$(echo "$1" | awk '/Memory at/ {print $5}')
    
    # 判定逻辑实现
    if [[ "$address" == "unassigned" ]]; then
        echo "[ERROR] Bar地址未分配，需要检查卡的状态（物理连接或供电异常）" 
        echo "建议操作：执行'lspci -vvv'确认设备识别状态}"
        return 1
    elif [[ `echo $address | wc -c` -gt 12 ]]; then
        echo "[WARNING] Bar地址超出44bit（当前地址：0x${address}）"
        echo "解决方案：调整BIOS的MMIO High Base < 16T}"
        return 2
    fi
	if [[ "$address"  == "<ignored>" ]]; then
        echo "[ERROR] 获取不到bar地址"
        echo "修复建议：检查/proc/cmdline是否包含'pcie=realloc'配置"
        grep -q "pcie=realloc" /proc/cmdline || echo "  当前配置：$(cat /proc/cmdline)"
        return 3
    fi
    echo "PCIe 状态正常"
    return 0
}

pcie_check() {
    if [ ! -f "$1" ]; then
        echo "file not exists" >&2
        exit 1
    fi
	echo "Region 0地址测试"
	grep "Region 0" $1 | while read -r line; do
		analyze_regions "$line"
	done
	echo "Region 5 地址测试"
	grep "Region 5" $1 | while read -r line; do
		analyze_regions "$line"
	done
}

sme_check() {
    if [ ! -f "$1" ]; then
        echo "file not exists" >&2
        exit 1
    fi
    grep -i sme  $1 > $OUTPUT_DIR/sme.log
    if [ -s "$$OUTPUT_DIR/sme.log" ]; then
        echo "如果不是CSV场景，需要BIOS关闭SME设置"
        return 1
    else
        echo "OS SME目前没有打开, 非CSV场景下，该状态正常"
    fi
}
# 驱动安装位置定位
kernel_check() {
    # 当前kernel版本
    kernel_version=`uname -r`
    # 驱动安装到的kernel
    drive_in_kernel=`find /lib/ | grep -E "hydcu|hycu" | head -n 1 | awk -F "/" '{print $4}'`

    if [ "$kernel_version" = "$drive_in_kernel" ]; then
        echo "驱动安装在当前kernel版本下, 符合正常情况。"
    else
        echo "你的内核可能有所变更，检查下环境是否是多个内核"
    fi
}

# 驱动
# 分析错误信息
analyze_errors() {
	
    log "分析关键字 $1 相关错误..."
    
    local error_pattern="$1"
    local error_flags="fail|error|uncorrect|warn|exception"
    
    # 在dmesg和系统日志中搜索
    grep -iE "$error_pattern.*($error_flags)|($error_flags).*$error_pattern" \
        "$OUTPUT_DIR/dmesg.log" "$OUTPUT_DIR/system.log" > "$OUTPUT_DIR/driver_issues.log"
    
    if [ -s "$OUTPUT_DIR/driver_issues.log" ]; then
        log "发现潜在问题："
        [ $QUIET_MODE -eq 0 ] && cat "$OUTPUT_DIR/driver_issues.log"
        return 1
    else
        log "未发现相关错误信息"
        rm -f "$OUTPUT_DIR/driver_issues.log"
        return 0
    fi
}

## 标准化提示信息格式
head_normal() {
    echo "\n########$1########"
}

# 主流程
main() {
	hline && echo
    get_dcu
    collect_system_info
    collect_logs
    #analyze_errors
	get_pcie_info
    
    echo "\n### 日志分析 ###"
    hline
    head_normal "分析pcie信息"
    pcie_check $OUTPUT_DIR/pcie_info.log
    head_normal "分析sme信息"
    sme_check $OUTPUT_DIR/dmesg.log
    head_normal "分析驱动安装位置"
    kernel_check
    
	./tools/driver_load_check.sh > $OUTPUT_DIR/driver_status.log
    ./tools/board_check.sh > $OUTPUT_DIR/board_check.log
	product_name=`dmidecode -s system-product-name`
	if [ "$product_name" != "X785-H30" ]; then
    	./tools/pcie_speed_check.sh > $OUTPUT_DIR/pcie_speek_check.log
    fi
	local status=$?
    
    # 打包结果
    log "打包诊断数据..."
    tar -czf "${OUTPUT_DIR}.tar.gz" "$OUTPUT_DIR" 2>/dev/null
    rm -rf "$OUTPUT_DIR"
    
    log "诊断文件已保存为：${OUTPUT_DIR}.tar.gz"
    return $status
}

# 执行主程序
main
exit $?

