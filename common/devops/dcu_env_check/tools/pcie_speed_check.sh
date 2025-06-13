#!/bin/bash

# 检查是否具有root权限
if [ "$EUID" -ne 0 ]; then
    echo "错误：该脚本需要root权限运行（请使用sudo执行）" >&2
    exit 1
fi

# 获取主板序列号
speed=$(./tools/hydcutune -pciestatus | grep -i speed | awk '{print $5'})

# 检查是否成功获取序列号
if [ -z "$speed" ]; then
    echo "没有获取到当前pcie 速率" >&2
    exit 1
fi

# 型号判断逻辑
case $speed in
    Gen1|Gen2|Gen3)
        echo "当前PCIe 速率偏低，需要检查vbios或者使用hydcutune修复" >&2
        exit 1
        ;;
    Gen4|Gen5)
        echo "PCIe速率正常"
        exit 0
        ;;
    *)
        echo "未检测到PCIe速率"
        exit 2
        ;;
esac
