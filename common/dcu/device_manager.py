"""
DCU设备管理器
提供DCU设备的检测、管理、监控和优化功能
"""

import torch
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import psutil
import json
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DCUDeviceInfo:
    """DCU设备信息"""
    device_id: int
    name: str
    total_memory: int  # bytes
    major: int
    minor: int
    multi_processor_count: int
    
    @property
    def total_memory_gb(self) -> float:
        """总显存(GB)"""
        return self.total_memory / (1024 ** 3)
    
    @property
    def compute_capability(self) -> str:
        """计算能力"""
        return f"{self.major}.{self.minor}"


@dataclass
class DCUMemoryInfo:
    """DCU显存信息"""
    device_id: int
    allocated: int  # bytes
    reserved: int   # bytes
    total: int      # bytes
    
    @property
    def allocated_gb(self) -> float:
        return self.allocated / (1024 ** 3)
    
    @property
    def reserved_gb(self) -> float:
        return self.reserved / (1024 ** 3)
    
    @property
    def total_gb(self) -> float:
        return self.total / (1024 ** 3)
    
    @property
    def utilization(self) -> float:
        """显存利用率(%)"""
        return (self.allocated / self.total) * 100 if self.total > 0 else 0.0


@dataclass
class DCUPerformanceInfo:
    """DCU性能信息"""
    device_id: int
    utilization: float  # %
    temperature: float  # °C
    power_usage: float  # W
    memory_info: DCUMemoryInfo


class DCUDeviceManager:
    """DCU设备管理器"""
    
    def __init__(self):
        self.devices: List[DCUDeviceInfo] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._performance_history: Dict[int, List[DCUPerformanceInfo]] = {}
        self._initialize_devices()
    
    def _initialize_devices(self):
        """初始化设备信息"""
        if not torch.cuda.is_available():
            logger.warning("DCU设备不可用")
            return
        
        device_count = torch.cuda.device_count()
        logger.info(f"检测到 {device_count} 个DCU设备")
        
        for i in range(device_count):
            try:
                props = torch.cuda.get_device_properties(i)
                device_info = DCUDeviceInfo(
                    device_id=i,
                    name=props.name,
                    total_memory=props.total_memory,
                    major=props.major,
                    minor=props.minor,
                    multi_processor_count=props.multi_processor_count
                )
                self.devices.append(device_info)
                self._performance_history[i] = []
                logger.info(f"DCU {i}: {device_info.name} ({device_info.total_memory_gb:.1f} GB)")
            except Exception as e:
                logger.error(f"获取DCU {i} 信息失败: {e}")
    
    def get_device_count(self) -> int:
        """获取设备数量"""
        return len(self.devices)
    
    def get_device_info(self, device_id: Optional[int] = None) -> Dict[str, Any]:
        """获取设备信息"""
        if device_id is None:
            return {
                'device_count': len(self.devices),
                'devices': [
                    {
                        'id': dev.device_id,
                        'name': dev.name,
                        'memory_gb': dev.total_memory_gb,
                        'compute_capability': dev.compute_capability
                    }
                    for dev in self.devices
                ]
            }
        
        if device_id >= len(self.devices):
            raise ValueError(f"设备ID {device_id} 超出范围")
        
        device = self.devices[device_id]
        return {
            'id': device.device_id,
            'name': device.name,
            'memory_gb': device.total_memory_gb,
            'compute_capability': device.compute_capability,
            'multi_processor_count': device.multi_processor_count
        }
    
    def get_memory_info(self, device_id: int = 0) -> DCUMemoryInfo:
        """获取显存信息"""
        if device_id >= len(self.devices):
            raise ValueError(f"设备ID {device_id} 超出范围")
        
        allocated = torch.cuda.memory_allocated(device_id)
        reserved = torch.cuda.memory_reserved(device_id)
        total = self.devices[device_id].total_memory
        
        return DCUMemoryInfo(
            device_id=device_id,
            allocated=allocated,
            reserved=reserved,
            total=total
        )
    
    def get_all_memory_info(self) -> List[DCUMemoryInfo]:
        """获取所有设备的显存信息"""
        return [self.get_memory_info(i) for i in range(len(self.devices))]
    
    def optimize_memory(self, device_id: Optional[int] = None):
        """优化显存使用"""
        if device_id is None:
            # 优化所有设备
            for i in range(len(self.devices)):
                self._optimize_device_memory(i)
        else:
            self._optimize_device_memory(device_id)
    
    def _optimize_device_memory(self, device_id: int):
        """优化单个设备的显存"""
        try:
            # 清理显存缓存
            torch.cuda.empty_cache()
            
            # 设置显存分配策略
            torch.cuda.set_per_process_memory_fraction(0.95, device_id)
            
            logger.info(f"DCU {device_id} 显存优化完成")
        except Exception as e:
            logger.error(f"DCU {device_id} 显存优化失败: {e}")
    
    def set_performance_mode(self, mode: str = 'high'):
        """设置性能模式"""
        if mode == 'high':
            # 高性能模式
            for i in range(len(self.devices)):
                try:
                    torch.cuda.set_device(i)
                    # 启用TensorFloat-32
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    # 启用cuDNN基准模式
                    torch.backends.cudnn.benchmark = True
                    logger.info(f"DCU {i} 设置为高性能模式")
                except Exception as e:
                    logger.error(f"DCU {i} 性能模式设置失败: {e}")
        
        elif mode == 'balanced':
            # 平衡模式
            for i in range(len(self.devices)):
                try:
                    torch.cuda.set_device(i)
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = False
                    torch.backends.cudnn.benchmark = False
                    logger.info(f"DCU {i} 设置为平衡模式")
                except Exception as e:
                    logger.error(f"DCU {i} 性能模式设置失败: {e}")
        
        elif mode == 'power_save':
            # 节能模式
            for i in range(len(self.devices)):
                try:
                    torch.cuda.set_device(i)
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                    torch.backends.cudnn.benchmark = False
                    logger.info(f"DCU {i} 设置为节能模式")
                except Exception as e:
                    logger.error(f"DCU {i} 性能模式设置失败: {e}")
    
    def start_monitoring(self, interval: float = 1.0):
        """开始性能监控"""
        if self._monitoring:
            logger.warning("监控已在运行")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("开始DCU性能监控")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("停止DCU性能监控")
    
    def _monitor_loop(self, interval: float):
        """监控循环"""
        while self._monitoring:
            try:
                for i in range(len(self.devices)):
                    perf_info = self._get_performance_info(i)
                    self._performance_history[i].append(perf_info)
                    
                    # 保持历史记录在合理范围内（最多1000条）
                    if len(self._performance_history[i]) > 1000:
                        self._performance_history[i] = self._performance_history[i][-1000:]
                
                time.sleep(interval)
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
    
    def _get_performance_info(self, device_id: int) -> DCUPerformanceInfo:
        """获取设备性能信息"""
        memory_info = self.get_memory_info(device_id)
        
        # 尝试获取设备利用率（可能需要特定的DCU监控工具）
        try:
            utilization = self._get_device_utilization(device_id)
        except:
            utilization = 0.0
        
        # 尝试获取温度信息
        try:
            temperature = self._get_device_temperature(device_id)
        except:
            temperature = 0.0
        
        # 尝试获取功耗信息
        try:
            power_usage = self._get_device_power_usage(device_id)
        except:
            power_usage = 0.0
        
        return DCUPerformanceInfo(
            device_id=device_id,
            utilization=utilization,
            temperature=temperature,
            power_usage=power_usage,
            memory_info=memory_info
        )
    
    def _get_device_utilization(self, device_id: int) -> float:
        """获取设备利用率（需要DCU专用工具）"""
        # 这里需要调用DCU特定的监控API
        # 暂时返回模拟值
        return 0.0
    
    def _get_device_temperature(self, device_id: int) -> float:
        """获取设备温度"""
        # 这里需要调用DCU特定的温度监控API
        return 0.0
    
    def _get_device_power_usage(self, device_id: int) -> float:
        """获取设备功耗"""
        # 这里需要调用DCU特定的功耗监控API
        return 0.0
    
    def get_performance_summary(self, device_id: Optional[int] = None) -> Dict[str, Any]:
        """获取性能摘要"""
        if device_id is None:
            # 所有设备的摘要
            summary = {}
            for i in range(len(self.devices)):
                summary[f"device_{i}"] = self._get_device_summary(i)
            return summary
        else:
            return self._get_device_summary(device_id)
    
    def _get_device_summary(self, device_id: int) -> Dict[str, Any]:
        """获取单个设备的性能摘要"""
        if device_id not in self._performance_history:
            return {"error": "无性能历史数据"}
        
        history = self._performance_history[device_id]
        if not history:
            return {"error": "无性能历史数据"}
        
        latest = history[-1]
        
        # 计算平均值（最近10个数据点）
        recent_data = history[-10:] if len(history) >= 10 else history
        avg_utilization = sum(p.utilization for p in recent_data) / len(recent_data)
        avg_memory_usage = sum(p.memory_info.utilization for p in recent_data) / len(recent_data)
        
        return {
            "device_id": device_id,
            "current": {
                "utilization": latest.utilization,
                "memory_usage": latest.memory_info.utilization,
                "temperature": latest.temperature,
                "power_usage": latest.power_usage
            },
            "average": {
                "utilization": avg_utilization,
                "memory_usage": avg_memory_usage
            },
            "memory": {
                "allocated_gb": latest.memory_info.allocated_gb,
                "total_gb": latest.memory_info.total_gb,
                "utilization": latest.memory_info.utilization
            }
        }
    
    def save_performance_report(self, filepath: str):
        """保存性能报告"""
        report = {
            "timestamp": time.time(),
            "devices": self.get_device_info(),
            "performance": self.get_performance_summary()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"性能报告已保存到: {filepath}")
    
    def reset_device(self, device_id: int):
        """重置设备状态"""
        try:
            torch.cuda.set_device(device_id)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device_id)
            logger.info(f"DCU {device_id} 重置完成")
        except Exception as e:
            logger.error(f"DCU {device_id} 重置失败: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        result = {
            "status": "healthy",
            "issues": [],
            "devices": []
        }
        
        for i, device in enumerate(self.devices):
            device_status = {
                "device_id": i,
                "name": device.name,
                "status": "healthy",
                "issues": []
            }
            
            try:
                # 检查显存
                memory_info = self.get_memory_info(i)
                if memory_info.utilization > 95:
                    device_status["issues"].append("显存使用率过高")
                    device_status["status"] = "warning"
                
                # 检查设备可访问性
                torch.cuda.set_device(i)
                test_tensor = torch.randn(100, 100, device=i)
                torch.cuda.synchronize()
                del test_tensor
                
            except Exception as e:
                device_status["issues"].append(f"设备访问错误: {e}")
                device_status["status"] = "error"
                result["status"] = "unhealthy"
            
            result["devices"].append(device_status)
            
            if device_status["issues"]:
                result["issues"].extend([f"DCU {i}: {issue}" for issue in device_status["issues"]])
        
        return result


# 全局设备管理器实例
_device_manager = None

def get_device_manager() -> DCUDeviceManager:
    """获取全局设备管理器实例"""
    global _device_manager
    if _device_manager is None:
        _device_manager = DCUDeviceManager()
    return _device_manager 