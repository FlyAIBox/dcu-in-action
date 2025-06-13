#!/bin/bash
#huangjun@hygon.cn
#v0.2

DEVICE_ID="1d94:(5|6)[0-9a-z]{3,3}"
pn["5"]="zifang"
pn["6"]="kongming"
dev=($(lspci -nn | grep -oE "1d94:(5|6)[0-9a-z]{3,3}" | awk -F: '{print $2}' | grep -o [56]))
devname=${pn[${dev[0]}]}
echo "===THIS SCRIPT JUST FOR 5.16.21  5.2 V1.10 and later==="

function ko_is_loaded()
{
	local st=$(lsmod | grep "\<$1\>")
	local ret=yes
	if [ "$st" = "" ];then
		ret="no"
	fi
	echo $ret
}

function have_mod()
{
	echo "$(modinfo $1)"
}

function check_iommu()
{
	if [ "$(ko_is_loaded iommu_v2)" = "yes" ];then
		echo "use iommu_v2, ready"
		return 0
	fi

	if [ "$(ko_is_loaded amd_iommu_v2)" = "yes" ];then
		echo "use amd_iommu_v2, ready"
		return 0
	fi

	if [ "$(have_mod iommu_v2)" != "" ];then
		echo "have iommu_v2 in disk, but not loaded"
		return -1
	fi

	if [ "$(have_mod amd_iommu_v2)" != "" ];then
		echo "have amd_iommu_v2 in disk, but not loaded"
		return -2
	fi

	echo "no iommu driver on this system"
	return -3
}

function check_vfio_pci()
{
	if [ "$(ko_is_loaded vfio-pci)" = "yes" ];then
		echo "Some device have attach to VM"
		echo "pls check it"
	fi
}

function _have_read_perm()
{
	[[ -r $1 ]] && echo "yes"
}

function _find_ucode_in_path()
{
	local u=$2
	local p=$1
	local au=""
	local tc=0
	local cnt=0

	au="$(find $p -name ${devname}_$u.bin 2> /dev/null)"
	tc=$(find $p -name ${devname}_$u.bin 2> /dev/null | wc -l)
	cnt=$(($cnt + $tc))
	echo $au
	return $cnt
}

function check_ucode()
{
	local ucodes="sdma sdma1 mec mec2 rlc smu"
	local v=$(uname -r)
	local paths="/lib/firmware/updates/$v /lib/firmware/updates/ /lib/firmware/$v"
	local cnt=0
	local au=""
	local e=
	local u
	local p
	local rp

	for u in $ucodes;do
		for p in $paths;do
			au="$au $(_find_ucode_in_path $p $u)"
			cnt=$(($cnt + $?))
		done
		if [[ $cnt -gt 1 ]];then
			echo "our firmware is local:[/lib/firmware/$v]"
			echo "pls rmove the other firmware."
			echo "all:[$au]"
			e="yes"
		fi
		if [ "$cnt" = "0" ];then
			echo "no ${devname}_$u.bin found! pls reinstall driver."
			e="yes"
		fi
		for p in $au;do 
			local r=$(_have_read_perm $p)
			if [ "$r" != "yes" ];then
				echo "no read perm on firmware: $p"
				e="yes"
			fi
		done
		au=""
		cnt=0
	done
	if [ "$e" = "yes" ];then
		exit -1
	fi
	echo "firmware, ready"
}

function check_ko()
{
	local kos="hydcu.ko  hydcu-sched.ko  hydrm_ttm_helper.ko  hy-extra.ko  hykcl.ko  hyttm.ko"
	local dir="/opt/hyhal/dkms/"
	local ret=

	for k in $kos;do
		local r=$(_have_read_perm $dir/$k)
		if [ "$r" != "yes" ];then
			ret="$ret $k"
		fi
	done
	if [ "$ret" != "" ];then
		echo "no driver installed or loss read perm"
		echo "pls check[$kos] in $dir"
		exit -1
	fi
	echo "dcu ko, ready"
}

function check_cuser_if_video()
{
	local r=$(cat /etc/group | grep video | grep $USER)
	if [ "$r" = "" ];then
		echo "you should add user:$USER to video group. sudo usermod -aG video $USER"
		exit -1
	fi
	echo "user group, ready"
}

function check_system_cap()
{
	if [ -r /sys/fs/selinux/enforce ] && [ "$(semodule -l | grep hydcu)" = "" ];then
		echo "system service no cap to load module"
		echo "pls install driver again"
		exit -1
	fi
	echo "system service policy, ready"
}

#0
if [ "$(ko_is_loaded hydcu)" = "yes" ];then
	echo "driver loaded"
	exit 0
fi
#1
check_iommu
check_vfio_pci
#2
check_ko
#3
check_ucode
#4
check_system_cap
#5
check_cuser_if_video

echo "驱动检查结束，没有发现明显问题"
