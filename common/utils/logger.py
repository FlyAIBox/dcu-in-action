"""
统一日志系统
提供项目级别的日志管理，支持多种输出格式和性能监控
"""

import logging
import logging.handlers
import sys
import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime
import traceback
import functools
from contextlib import contextmanager

try:
    from loguru import logger as loguru_logger
    HAS_LOGURU = True
except ImportError:
    HAS_LOGURU = False

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.traceback import install as install_rich_traceback
    HAS_RICH = True
    # 安装rich的异常处理
    install_rich_traceback()
except ImportError:
    HAS_RICH = False


class DCULogger:
    """DCU项目专用日志器"""
    
    def __init__(self, 
                 name: str = "dcu-in-action",
                 level: str = "INFO",
                 log_dir: Optional[str] = None,
                 use_rich: bool = True,
                 use_loguru: bool = False):
        
        self.name = name
        self.level = level.upper()
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.use_rich = use_rich and HAS_RICH
        self.use_loguru = use_loguru and HAS_LOGURU
        
        # 创建日志目录
        self.log_dir.mkdir(exist_ok=True)
        
        # 初始化日志器
        if self.use_loguru:
            self._setup_loguru()
        else:
            self._setup_standard_logger()
        
        # 性能监控相关
        self._performance_logs = []
        self._start_times = {}
    
    def _setup_loguru(self):
        """设置Loguru日志器"""
        # 清除默认处理器
        loguru_logger.remove()
        
        # 控制台输出
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        
        loguru_logger.add(
            sys.stdout,
            format=console_format,
            level=self.level,
            colorize=True
        )
        
        # 文件输出
        log_file = self.log_dir / f"{self.name}.log"
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
        
        loguru_logger.add(
            log_file,
            format=file_format,
            level=self.level,
            rotation="1 day",
            retention="7 days",
            compression="zip",
            encoding="utf-8"
        )
        
        # 错误日志单独文件
        error_file = self.log_dir / f"{self.name}_error.log"
        loguru_logger.add(
            error_file,
            format=file_format,
            level="ERROR",
            rotation="1 day",
            retention="30 days",
            compression="zip",
            encoding="utf-8"
        )
        
        self.logger = loguru_logger
    
    def _setup_standard_logger(self):
        """设置标准库日志器"""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(getattr(logging, self.level))
        
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 创建格式器
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        if self.use_rich:
            console_handler = RichHandler(
                console=Console(stderr=True),
                show_time=False,
                show_path=False,
                markup=True
            )
            console_handler.setFormatter(
                logging.Formatter(fmt='%(message)s')
            )
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
        
        console_handler.setLevel(getattr(logging, self.level))
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        log_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, self.level))
        self.logger.addHandler(file_handler)
        
        # 错误日志处理器
        error_file = self.log_dir / f"{self.name}_error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=10*1024*1024,
            backupCount=10,
            encoding='utf-8'
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        self.logger.addHandler(error_handler)
    
    def debug(self, message: str, **kwargs):
        """调试日志"""
        if self.use_loguru:
            self.logger.debug(message, **kwargs)
        else:
            self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """信息日志"""
        if self.use_loguru:
            self.logger.info(message, **kwargs)
        else:
            self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """警告日志"""
        if self.use_loguru:
            self.logger.warning(message, **kwargs)
        else:
            self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """错误日志"""
        if self.use_loguru:
            if exc_info:
                self.logger.exception(message, **kwargs)
            else:
                self.logger.error(message, **kwargs)
        else:
            self.logger.error(message, exc_info=exc_info, extra=kwargs)
    
    def critical(self, message: str, exc_info: bool = False, **kwargs):
        """严重错误日志"""
        if self.use_loguru:
            if exc_info:
                self.logger.exception(message, **kwargs)
            else:
                self.logger.critical(message, **kwargs)
        else:
            self.logger.critical(message, exc_info=exc_info, extra=kwargs)
    
    def exception(self, message: str, **kwargs):
        """异常日志（自动包含堆栈信息）"""
        if self.use_loguru:
            self.logger.exception(message, **kwargs)
        else:
            self.logger.exception(message, extra=kwargs)
    
    @contextmanager
    def timer(self, operation: str, log_level: str = "INFO"):
        """性能计时上下文管理器"""
        start_time = time.time()
        self.info(f"开始 {operation}")
        
        try:
            yield
        except Exception as e:
            elapsed = time.time() - start_time
            self.error(f"{operation} 失败 (耗时: {elapsed:.2f}s): {e}", exc_info=True)
            raise
        else:
            elapsed = time.time() - start_time
            log_func = getattr(self, log_level.lower())
            log_func(f"{operation} 完成 (耗时: {elapsed:.2f}s)")
            
            # 记录性能日志
            self._performance_logs.append({
                "operation": operation,
                "duration": elapsed,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            })
    
    def start_timer(self, operation: str):
        """开始计时"""
        self._start_times[operation] = time.time()
        self.info(f"开始 {operation}")
    
    def end_timer(self, operation: str, log_level: str = "INFO"):
        """结束计时"""
        if operation not in self._start_times:
            self.warning(f"未找到操作 {operation} 的开始时间")
            return
        
        elapsed = time.time() - self._start_times[operation]
        del self._start_times[operation]
        
        log_func = getattr(self, log_level.lower())
        log_func(f"{operation} 完成 (耗时: {elapsed:.2f}s)")
        
        # 记录性能日志
        self._performance_logs.append({
            "operation": operation,
            "duration": elapsed,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        })
    
    def log_performance(self, operation: str, duration: float, metadata: Optional[Dict] = None):
        """记录性能指标"""
        perf_log = {
            "operation": operation,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self._performance_logs.append(perf_log)
        self.info(f"性能: {operation} 耗时 {duration:.3f}s", **metadata)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能统计摘要"""
        if not self._performance_logs:
            return {"message": "无性能数据"}
        
        operations = {}
        for log in self._performance_logs:
            op = log["operation"]
            if op not in operations:
                operations[op] = []
            operations[op].append(log["duration"])
        
        summary = {}
        for op, durations in operations.items():
            summary[op] = {
                "count": len(durations),
                "total": sum(durations),
                "average": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations)
            }
        
        return summary
    
    def save_performance_report(self, filepath: Optional[str] = None):
        """保存性能报告"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.log_dir / f"performance_report_{timestamp}.json"
        
        report = {
            "summary": self.get_performance_summary(),
            "detailed_logs": self._performance_logs,
            "generated_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.info(f"性能报告已保存到: {filepath}")
    
    def log_system_info(self):
        """记录系统信息"""
        import platform
        import psutil
        
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
        }
        
        try:
            import torch
            system_info["pytorch_version"] = torch.__version__
            system_info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                system_info["cuda_device_count"] = torch.cuda.device_count()
        except ImportError:
            pass
        
        self.info("系统信息", **system_info)
    
    def configure_level(self, level: str):
        """动态调整日志级别"""
        self.level = level.upper()
        if self.use_loguru:
            # Loguru需要重新配置
            self._setup_loguru()
        else:
            self.logger.setLevel(getattr(logging, self.level))
            for handler in self.logger.handlers:
                handler.setLevel(getattr(logging, self.level))


def performance_monitor(operation: str = None, logger_name: str = "dcu-in-action"):
    """性能监控装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation or f"{func.__module__}.{func.__name__}"
            logger = get_logger(logger_name)
            
            with logger.timer(op_name):
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def log_exceptions(logger_name: str = "dcu-in-action"):
    """异常记录装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"函数 {func.__name__} 执行失败")
                raise
        
        return wrapper
    
    return decorator


# 全局日志器实例
_loggers: Dict[str, DCULogger] = {}

def get_logger(name: str = "dcu-in-action", 
               level: str = "INFO",
               log_dir: Optional[str] = None,
               use_rich: bool = True,
               use_loguru: bool = False) -> DCULogger:
    """获取日志器实例"""
    
    # 检查环境变量
    if level == "INFO":
        level = os.getenv("DCU_LOG_LEVEL", "INFO")
    
    if log_dir is None:
        log_dir = os.getenv("DCU_LOG_DIR", "logs")
    
    # 创建或获取日志器
    logger_key = f"{name}_{level}_{log_dir}"
    if logger_key not in _loggers:
        _loggers[logger_key] = DCULogger(
            name=name,
            level=level,
            log_dir=log_dir,
            use_rich=use_rich,
            use_loguru=use_loguru
        )
    
    return _loggers[logger_key]


def setup_global_logging(level: str = "INFO", 
                        log_dir: str = "logs",
                        use_rich: bool = True,
                        use_loguru: bool = False):
    """设置全局日志配置"""
    # 设置环境变量
    os.environ["DCU_LOG_LEVEL"] = level
    os.environ["DCU_LOG_DIR"] = log_dir
    
    # 获取主日志器
    main_logger = get_logger(
        "dcu-in-action",
        level=level,
        log_dir=log_dir,
        use_rich=use_rich,
        use_loguru=use_loguru
    )
    
    # 记录系统信息
    main_logger.log_system_info()
    main_logger.info("全局日志系统初始化完成")
    
    return main_logger


# 兼容性接口
def create_logger(name: str, level: str = "INFO") -> logging.Logger:
    """创建标准库日志器（兼容性接口）"""
    dcu_logger = get_logger(name, level)
    return dcu_logger.logger if hasattr(dcu_logger.logger, 'handlers') else logging.getLogger(name) 