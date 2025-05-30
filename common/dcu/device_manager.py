"""
DCU Device Manager for HygonDCU
"""

import logging
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DCUInfo:
    """DCU device information"""
    device_id: int
    name: str
    memory_total: int  # MB
    memory_used: int   # MB
    memory_free: int   # MB
    compute_capability: str


class DCUManager:
    """DCU device manager"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._is_available = torch.cuda.is_available()
        
    def is_available(self) -> bool:
        return self._is_available
    
    def get_device_count(self) -> int:
        if not self._is_available:
            return 0
        try:
            return torch.cuda.device_count()
        except Exception as e:
            self.logger.error(f"Failed to get device count: {e}")
            return 0
    
    def get_device_info(self, device_id: int = 0) -> Optional[DCUInfo]:
        if not self._is_available or device_id >= self.get_device_count():
            return None
            
        try:
            properties = torch.cuda.get_device_properties(device_id)
            memory_total = properties.total_memory // (1024*1024)
            memory_free, _ = torch.cuda.mem_get_info(device_id)
            memory_free = memory_free // (1024*1024)
            memory_used = memory_total - memory_free
            
            return DCUInfo(
                device_id=device_id,
                name=properties.name,
                memory_total=memory_total,
                memory_used=memory_used,
                memory_free=memory_free,
                compute_capability=f"{properties.major}.{properties.minor}"
            )
        except Exception as e:
            self.logger.error(f"Failed to get device info: {e}")
            return None
    
    def get_all_devices_info(self) -> List[DCUInfo]:
        devices = []
        for i in range(self.get_device_count()):
            device_info = self.get_device_info(i)
            if device_info:
                devices.append(device_info)
        return devices
    
    def set_device(self, device_id: int) -> bool:
        try:
            if device_id >= self.get_device_count():
                return False
            torch.cuda.set_device(device_id)
            return True
        except Exception as e:
            self.logger.error(f"Failed to set device: {e}")
            return False
    
    def clear_cache(self) -> bool:
        try:
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False
    
    def get_memory_info(self, device_id: int = 0) -> Dict[str, int]:
        try:
            if not self._is_available:
                return {"total": 0, "used": 0, "free": 0}
                
            free_memory, total_memory = torch.cuda.mem_get_info(device_id)
            used_memory = total_memory - free_memory
            
            return {
                "total": total_memory // (1024*1024),
                "used": used_memory // (1024*1024),
                "free": free_memory // (1024*1024)
            }
        except Exception as e:
            self.logger.error(f"Failed to get memory info: {e}")
            return {"total": 0, "used": 0, "free": 0} 