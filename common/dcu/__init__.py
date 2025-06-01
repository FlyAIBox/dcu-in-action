"""
DCU设备管理模块
提供海光DCU加速卡的设备管理、监控、优化等功能
"""

from .device_manager import DCUDeviceManager, DCUDeviceInfo, DCUMemoryInfo, DCUPerformanceInfo

# 为了向后兼容性，提供别名
DCUManager = DCUDeviceManager

__all__ = [
    "DCUDeviceManager", 
    "DCUManager",  # 别名
    "DCUDeviceInfo",
    "DCUMemoryInfo", 
    "DCUPerformanceInfo"
] 