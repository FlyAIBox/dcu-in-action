#!/bin/bash

# 检查是否具有root权限
if [ "$EUID" -ne 0 ]; then
    echo "错误：该脚本需要root权限运行（请使用sudo执行）" >&2
    exit 1
fi

# 检查dmidecode命令是否存在
if ! command -v dmidecode &> /dev/null; then
    echo "错误：未找到dmidecode命令，请先安装dmidecode工具" >&2
    exit 1
fi

# 获取主板序列号
baseboard_SN=$(dmidecode -t 2 | grep -i  "Serial Number" | awk '{print $3}' )

# 检查是否成功获取序列号
if [ -z "$baseboard_SN" ]; then
    echo "错误：无法获取主板序列号" >&2
    exit 1
fi

# 型号判断逻辑
case $baseboard_SN in
    *AS*)
        echo "检测到主板型号：[${baseboard_SN}] 太老，满负载情况会出现掉卡" >&2
        exit 1
        ;;
    *BH*)
        echo "检测到主板型号：[${board_model}] 符合要求"
        exit 0
        ;;
    *)
        echo "未知主板型号，需要进一步查看"
        exit 2
        ;;
esac

