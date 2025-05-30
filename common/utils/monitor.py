"""
系统监控工具
提供DCU设备、系统资源、性能指标的实时监控和报告功能
"""

import time
import threading
import psutil
import json
import os
import subprocess
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import queue
import signal
import sys

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from prometheus_client import start_http_server, Gauge, Counter, Histogram
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_sent_mb: float
    network_recv_mb: float
    load_average: List[float]
    process_count: int


@dataclass
class DCUMetrics:
    """DCU设备指标"""
    timestamp: str
    device_id: int
    name: str
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    utilization_percent: float
    temperature: float
    power_usage: float
    processes: List[Dict[str, Any]]


@dataclass
class ProcessMetrics:
    """进程指标"""
    timestamp: str
    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    memory_rss_mb: float
    status: str
    create_time: str


class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, 
                 interval: float = 5.0,
                 history_size: int = 1000,
                 enable_prometheus: bool = False,
                 prometheus_port: int = 8000):
        
        self.interval = interval
        self.history_size = history_size
        self.enable_prometheus = enable_prometheus
        self.prometheus_port = prometheus_port
        
        # 监控状态
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # 数据存储
        self.system_history: List[SystemMetrics] = []
        self.dcu_history: Dict[int, List[DCUMetrics]] = {}
        self.process_history: List[ProcessMetrics] = []
        
        # 回调函数
        self.callbacks: List[Callable] = []
        
        # 警报配置
        self.alerts = {
            'cpu_threshold': 90.0,
            'memory_threshold': 90.0,
            'disk_threshold': 90.0,
            'dcu_memory_threshold': 95.0,
            'dcu_temperature_threshold': 85.0
        }
        
        # Prometheus指标
        if self.enable_prometheus and HAS_PROMETHEUS:
            self._setup_prometheus_metrics()
        
        # 初始化DCU监控
        if HAS_TORCH and torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                self.dcu_history[i] = []
        
        logger.info(f"系统监控器初始化完成 (间隔: {interval}s)")
    
    def _setup_prometheus_metrics(self):
        """设置Prometheus指标"""
        # 系统指标
        self.prom_cpu = Gauge('system_cpu_percent', 'CPU使用率')
        self.prom_memory = Gauge('system_memory_percent', '内存使用率')
        self.prom_disk = Gauge('system_disk_percent', '磁盘使用率')
        
        # DCU指标
        self.prom_dcu_memory = Gauge('dcu_memory_percent', 'DCU显存使用率', ['device_id'])
        self.prom_dcu_utilization = Gauge('dcu_utilization_percent', 'DCU利用率', ['device_id'])
        self.prom_dcu_temperature = Gauge('dcu_temperature', 'DCU温度', ['device_id'])
        
        # 启动Prometheus HTTP服务器
        start_http_server(self.prometheus_port)
        logger.info(f"Prometheus指标服务启动在端口 {self.prometheus_port}")
    
    def add_callback(self, callback: Callable[[Dict], None]):
        """添加监控数据回调函数"""
        self.callbacks.append(callback)
    
    def set_alert_threshold(self, metric: str, threshold: float):
        """设置警报阈值"""
        if metric in self.alerts:
            self.alerts[metric] = threshold
            logger.info(f"设置 {metric} 警报阈值为 {threshold}")
        else:
            logger.warning(f"未知的警报指标: {metric}")
    
    def start(self):
        """开始监控"""
        if self._running:
            logger.warning("监控已在运行")
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("系统监控已启动")
    
    def stop(self):
        """停止监控"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控主循环"""
        while self._running:
            try:
                # 收集系统指标
                system_metrics = self._collect_system_metrics()
                self._store_system_metrics(system_metrics)
                
                # 收集DCU指标
                if HAS_TORCH and torch.cuda.is_available():
                    for device_id in self.dcu_history.keys():
                        dcu_metrics = self._collect_dcu_metrics(device_id)
                        if dcu_metrics:
                            self._store_dcu_metrics(dcu_metrics)
                
                # 收集进程指标（可选）
                # process_metrics = self._collect_process_metrics()
                # self._store_process_metrics(process_metrics)
                
                # 检查警报
                self._check_alerts(system_metrics)
                
                # 调用回调函数
                self._call_callbacks({
                    'system': system_metrics,
                    'dcu': {k: v[-1] if v else None for k, v in self.dcu_history.items()}
                })
                
                # 更新Prometheus指标
                if self.enable_prometheus and HAS_PROMETHEUS:
                    self._update_prometheus_metrics(system_metrics)
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}", exc_info=True)
                time.sleep(self.interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # 内存信息
        memory = psutil.virtual_memory()
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # 磁盘信息
        disk = psutil.disk_usage('/')
        disk_used_gb = disk.used / (1024**3)
        disk_total_gb = disk.total / (1024**3)
        
        # 网络信息
        network = psutil.net_io_counters()
        network_sent_mb = network.bytes_sent / (1024**2)
        network_recv_mb = network.bytes_recv / (1024**2)
        
        # 系统负载
        try:
            load_average = list(os.getloadavg())
        except (OSError, AttributeError):
            load_average = [0.0, 0.0, 0.0]
        
        # 进程数量
        process_count = len(psutil.pids())
        
        return SystemMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            disk_percent=disk.percent,
            disk_used_gb=disk_used_gb,
            disk_total_gb=disk_total_gb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            load_average=load_average,
            process_count=process_count
        )
    
    def _collect_dcu_metrics(self, device_id: int) -> Optional[DCUMetrics]:
        """收集DCU设备指标"""
        if not HAS_TORCH or not torch.cuda.is_available():
            return None
        
        try:
            # 设备基本信息
            props = torch.cuda.get_device_properties(device_id)
            
            # 显存信息
            memory_allocated = torch.cuda.memory_allocated(device_id)
            memory_reserved = torch.cuda.memory_reserved(device_id)
            memory_total = props.total_memory
            
            memory_used_gb = memory_allocated / (1024**3)
            memory_total_gb = memory_total / (1024**3)
            memory_percent = (memory_allocated / memory_total) * 100
            
            # 尝试获取利用率（需要DCU特定工具）
            utilization_percent = self._get_dcu_utilization(device_id)
            
            # 尝试获取温度
            temperature = self._get_dcu_temperature(device_id)
            
            # 尝试获取功耗
            power_usage = self._get_dcu_power(device_id)
            
            # DCU进程信息
            processes = self._get_dcu_processes(device_id)
            
            return DCUMetrics(
                timestamp=datetime.now().isoformat(),
                device_id=device_id,
                name=props.name,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                memory_percent=memory_percent,
                utilization_percent=utilization_percent,
                temperature=temperature,
                power_usage=power_usage,
                processes=processes
            )
            
        except Exception as e:
            logger.error(f"收集DCU {device_id} 指标失败: {e}")
            return None
    
    def _get_dcu_utilization(self, device_id: int) -> float:
        """获取DCU利用率"""
        # 这里需要调用DCU特定的监控工具
        # 暂时返回模拟值
        try:
            # 可以尝试调用系统命令获取
            # result = subprocess.run(['rocm-smi', '--showuse'], capture_output=True, text=True)
            return 0.0
        except:
            return 0.0
    
    def _get_dcu_temperature(self, device_id: int) -> float:
        """获取DCU温度"""
        try:
            # 可以尝试调用系统命令获取
            # result = subprocess.run(['rocm-smi', '--showtemp'], capture_output=True, text=True)
            return 0.0
        except:
            return 0.0
    
    def _get_dcu_power(self, device_id: int) -> float:
        """获取DCU功耗"""
        try:
            # 可以尝试调用系统命令获取
            # result = subprocess.run(['rocm-smi', '--showpower'], capture_output=True, text=True)
            return 0.0
        except:
            return 0.0
    
    def _get_dcu_processes(self, device_id: int) -> List[Dict[str, Any]]:
        """获取DCU上运行的进程"""
        processes = []
        try:
            # 这里可以解析nvidia-ml-py类似的信息
            # 或调用DCU特定的进程监控命令
            pass
        except Exception as e:
            logger.debug(f"获取DCU {device_id} 进程信息失败: {e}")
        
        return processes
    
    def _collect_process_metrics(self) -> List[ProcessMetrics]:
        """收集进程指标"""
        metrics = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'memory_info', 'status', 'create_time']):
                try:
                    info = proc.info
                    memory_rss_mb = info['memory_info'].rss / (1024**2) if info['memory_info'] else 0
                    
                    metric = ProcessMetrics(
                        timestamp=datetime.now().isoformat(),
                        pid=info['pid'],
                        name=info['name'] or 'Unknown',
                        cpu_percent=info['cpu_percent'] or 0,
                        memory_percent=info['memory_percent'] or 0,
                        memory_rss_mb=memory_rss_mb,
                        status=info['status'],
                        create_time=datetime.fromtimestamp(info['create_time']).isoformat() if info['create_time'] else ""
                    )
                    metrics.append(metric)
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            logger.error(f"收集进程指标失败: {e}")
        
        return metrics
    
    def _store_system_metrics(self, metrics: SystemMetrics):
        """存储系统指标"""
        self.system_history.append(metrics)
        if len(self.system_history) > self.history_size:
            self.system_history = self.system_history[-self.history_size:]
    
    def _store_dcu_metrics(self, metrics: DCUMetrics):
        """存储DCU指标"""
        device_id = metrics.device_id
        self.dcu_history[device_id].append(metrics)
        if len(self.dcu_history[device_id]) > self.history_size:
            self.dcu_history[device_id] = self.dcu_history[device_id][-self.history_size:]
    
    def _store_process_metrics(self, metrics: List[ProcessMetrics]):
        """存储进程指标"""
        self.process_history.extend(metrics)
        if len(self.process_history) > self.history_size * 10:
            self.process_history = self.process_history[-self.history_size * 10:]
    
    def _check_alerts(self, system_metrics: SystemMetrics):
        """检查警报条件"""
        alerts = []
        
        # CPU警报
        if system_metrics.cpu_percent > self.alerts['cpu_threshold']:
            alerts.append(f"CPU使用率过高: {system_metrics.cpu_percent:.1f}%")
        
        # 内存警报
        if system_metrics.memory_percent > self.alerts['memory_threshold']:
            alerts.append(f"内存使用率过高: {system_metrics.memory_percent:.1f}%")
        
        # 磁盘警报
        if system_metrics.disk_percent > self.alerts['disk_threshold']:
            alerts.append(f"磁盘使用率过高: {system_metrics.disk_percent:.1f}%")
        
        # DCU警报
        for device_id, history in self.dcu_history.items():
            if history:
                latest = history[-1]
                if latest.memory_percent > self.alerts['dcu_memory_threshold']:
                    alerts.append(f"DCU {device_id} 显存使用率过高: {latest.memory_percent:.1f}%")
                
                if latest.temperature > self.alerts['dcu_temperature_threshold']:
                    alerts.append(f"DCU {device_id} 温度过高: {latest.temperature:.1f}°C")
        
        # 发送警报
        for alert in alerts:
            logger.warning(f"⚠️ 警报: {alert}")
    
    def _call_callbacks(self, data: Dict[str, Any]):
        """调用回调函数"""
        for callback in self.callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"回调函数执行失败: {e}")
    
    def _update_prometheus_metrics(self, system_metrics: SystemMetrics):
        """更新Prometheus指标"""
        if not (self.enable_prometheus and HAS_PROMETHEUS):
            return
        
        # 更新系统指标
        self.prom_cpu.set(system_metrics.cpu_percent)
        self.prom_memory.set(system_metrics.memory_percent)
        self.prom_disk.set(system_metrics.disk_percent)
        
        # 更新DCU指标
        for device_id, history in self.dcu_history.items():
            if history:
                latest = history[-1]
                self.prom_dcu_memory.labels(device_id=device_id).set(latest.memory_percent)
                self.prom_dcu_utilization.labels(device_id=device_id).set(latest.utilization_percent)
                self.prom_dcu_temperature.labels(device_id=device_id).set(latest.temperature)
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "monitoring": self._running,
            "system": asdict(self.system_history[-1]) if self.system_history else None,
            "dcu": {}
        }
        
        for device_id, history in self.dcu_history.items():
            if history:
                status["dcu"][device_id] = asdict(history[-1])
        
        return status
    
    def get_summary_report(self, hours: int = 1) -> Dict[str, Any]:
        """获取汇总报告"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # 过滤最近的数据
        recent_system = [
            m for m in self.system_history
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]
        
        recent_dcu = {}
        for device_id, history in self.dcu_history.items():
            recent_dcu[device_id] = [
                m for m in history
                if datetime.fromisoformat(m.timestamp) > cutoff_time
            ]
        
        # 计算统计信息
        system_stats = {}
        if recent_system:
            system_stats = {
                "cpu_avg": sum(m.cpu_percent for m in recent_system) / len(recent_system),
                "cpu_max": max(m.cpu_percent for m in recent_system),
                "memory_avg": sum(m.memory_percent for m in recent_system) / len(recent_system),
                "memory_max": max(m.memory_percent for m in recent_system),
                "disk_usage": recent_system[-1].disk_percent
            }
        
        dcu_stats = {}
        for device_id, data in recent_dcu.items():
            if data:
                dcu_stats[device_id] = {
                    "memory_avg": sum(m.memory_percent for m in data) / len(data),
                    "memory_max": max(m.memory_percent for m in data),
                    "utilization_avg": sum(m.utilization_percent for m in data) / len(data),
                    "utilization_max": max(m.utilization_percent for m in data),
                    "temperature_avg": sum(m.temperature for m in data) / len(data),
                    "temperature_max": max(m.temperature for m in data)
                }
        
        return {
            "period_hours": hours,
            "system": system_stats,
            "dcu": dcu_stats,
            "generated_at": datetime.now().isoformat()
        }
    
    def save_report(self, filepath: str, hours: int = 24):
        """保存监控报告"""
        report = {
            "current_status": self.get_current_status(),
            "summary": self.get_summary_report(hours),
            "config": {
                "interval": self.interval,
                "alerts": self.alerts,
                "prometheus_enabled": self.enable_prometheus
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"监控报告已保存到: {filepath}")
    
    def export_metrics(self, format: str = "json") -> str:
        """导出监控数据"""
        data = {
            "system_history": [asdict(m) for m in self.system_history],
            "dcu_history": {
                str(k): [asdict(m) for m in v] 
                for k, v in self.dcu_history.items()
            },
            "exported_at": datetime.now().isoformat()
        }
        
        if format.lower() == "json":
            return json.dumps(data, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的导出格式: {format}")


class DCUMonitor:
    """DCU设备专用监控器"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.system_monitor = SystemMonitor(interval=interval)
    
    def start_monitoring(self):
        """开始监控"""
        self.system_monitor.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.system_monitor.stop()
    
    def get_device_status(self) -> Dict[str, Any]:
        """获取DCU设备状态"""
        status = self.system_monitor.get_current_status()
        return status.get("dcu", {})
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return self.system_monitor.get_summary_report()


def start_daemon_monitor(interval: float = 5.0, 
                        log_file: str = "monitor.log",
                        pid_file: str = "monitor.pid"):
    """启动守护进程监控"""
    
    def signal_handler(signum, frame):
        logger.info("收到终止信号，停止监控...")
        monitor.stop()
        sys.exit(0)
    
    # 注册信号处理器
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # 写入PID文件
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))
    
    try:
        # 启动监控
        monitor = SystemMonitor(interval=interval, enable_prometheus=True)
        monitor.start()
        
        logger.info(f"守护进程监控已启动 (PID: {os.getpid()})")
        
        # 保持运行
        while True:
            time.sleep(60)  # 每分钟保存一次报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            report_file = f"monitor_report_{timestamp}.json"
            monitor.save_report(report_file, hours=1)
            
    except KeyboardInterrupt:
        logger.info("监控被用户中断")
    finally:
        monitor.stop()
        # 清理PID文件
        if os.path.exists(pid_file):
            os.remove(pid_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DCU系统监控工具")
    parser.add_argument("--interval", type=float, default=5.0, help="监控间隔(秒)")
    parser.add_argument("--daemon", action="store_true", help="守护进程模式")
    parser.add_argument("--prometheus", action="store_true", help="启用Prometheus监控")
    parser.add_argument("--port", type=int, default=8000, help="Prometheus端口")
    
    args = parser.parse_args()
    
    if args.daemon:
        start_daemon_monitor(interval=args.interval)
    else:
        monitor = SystemMonitor(
            interval=args.interval,
            enable_prometheus=args.prometheus,
            prometheus_port=args.port
        )
        
        try:
            monitor.start()
            logger.info("监控已启动，按Ctrl+C停止...")
            
            while True:
                time.sleep(10)
                status = monitor.get_current_status()
                print(f"\r当前状态: CPU {status['system']['cpu_percent']:.1f}% | 内存 {status['system']['memory_percent']:.1f}%", end="")
                
        except KeyboardInterrupt:
            logger.info("监控被用户中断")
        finally:
            monitor.stop() 