"""
DCU-in-Action 通用模块
提供DCU设备管理、大模型工具、HPC计算、通用工具等功能
"""

from .dcu.device_manager import DCUDeviceManager
from .utils.logger import get_logger, setup_global_logging
from .utils.monitor import SystemMonitor, DCUMonitor

# 版本信息
__version__ = "1.0.0"
__author__ = "DCU-in-Action Team"

# 公开接口
__all__ = [
    "DCUDeviceManager",
    "get_logger",
    "setup_global_logging", 
    "SystemMonitor",
    "DCUMonitor",
]

# 懒加载模块避免依赖问题
def get_dcu_manager():
    """获取DCU管理器"""
    from .dcu import DCUManager
    return DCUManager

def get_config_manager():
    """获取配置管理器"""
    from .utils import ConfigManager
    return ConfigManager

# 保持向后兼容性
def __getattr__(name):
    if name == "DCUManager":
        from .dcu import DCUManager
        return DCUManager
    elif name == "ConfigManager":
        from .utils import ConfigManager
        return ConfigManager
    elif name == "ModelLoader":
        from .llm import ModelLoader
        return ModelLoader
    elif name == "TokenizerUtils":
        from .llm import TokenizerUtils
        return TokenizerUtils
    elif name == "TrainingUtils":
        from .llm import TrainingUtils
        return TrainingUtils
    elif name == "InferenceServer":
        from .llm import InferenceServer
        return InferenceServer
    elif name == "LoRAFinetuner":
        from .llm import LoRAFinetuner
        return LoRAFinetuner
    elif name == "ParallelUtils":
        from .hpc import ParallelUtils
        return ParallelUtils
    elif name == "MatrixComputer":
        from .hpc import MatrixComputer
        return MatrixComputer
    elif name == "NumericalSolver":
        from .hpc import NumericalSolver
        return NumericalSolver
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # DCU相关
    "DCUManager",
    
    # 工具类
    "ConfigManager",
    
    # 大模型相关
    "ModelLoader",
    "TokenizerUtils", 
    "TrainingUtils",
    "InferenceServer",
    "LoRAFinetuner",
    
    # HPC相关
    "ParallelUtils",
    "MatrixComputer", 
    "NumericalSolver",
] 