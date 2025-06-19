#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
海光DCU K100-AI性能监控器
版本: v2.0
作者: DCU-in-Action Team

主要功能:
1. 实时监控DCU设备状态（温度、功耗、利用率、显存）
2. 监控系统资源（CPU、内存、网络、磁盘）
3. 记录和分析性能数据
4. 生成性能报告和可视化图表
5. 提供性能预警和优化建议
"""

import os
import sys
import time
import json
import threading
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import csv

import psutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DCUMetrics:
    """DCU设备指标"""
    device_id: int
    timestamp: str
    temperature: float  # 温度 (°C)
    power_usage: float  # 功耗 (W)
    gpu_utilization: float  # GPU利用率 (%)
    memory_used: float  # 显存使用 (GB)
    memory_total: float  # 显存总量 (GB)
    memory_utilization: float  # 显存利用率 (%)
    clock_graphics: float  # 图形时钟 (MHz)
    clock_memory: float  # 显存时钟 (MHz)
    fan_speed: float  # 风扇转速 (%)
    voltage: float  # 电压 (V)

@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: str
    cpu_utilization: float  # CPU利用率 (%)
    cpu_frequency: float  # CPU频率 (MHz)
    memory_used: float  # 内存使用 (GB)
    memory_total: float  # 内存总量 (GB)
    memory_utilization: float  # 内存利用率 (%)
    disk_usage: float  # 磁盘使用率 (%)
    network_sent: float  # 网络发送 (MB/s)
    network_recv: float  # 网络接收 (MB/s)
    load_average: Tuple[float, float, float]  # 负载平均值
    process_count: int  # 进程数量

class DCUMonitor:
    """DCU设备监控器"""
    
    def __init__(self):
        self.device_count = self._get_device_count()
        self.metrics_history = {i: deque(maxlen=1000) for i in range(self.device_count)}
        
    def _get_device_count(self) -> int:
        """获取DCU设备数量"""
        try:
            result = subprocess.run(['rocm-smi', '--showid'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                count = len([line for line in lines if 'GPU[' in line])
                logger.info(f"检测到 {count} 个DCU设备")
                return count
            else:
                logger.error("无法获取DCU设备数量")
                return 0
        except Exception as e:
            logger.error(f"获取DCU设备数量失败: {e}")
            return 0
    
    def collect_metrics(self) -> List[DCUMetrics]:
        """收集所有DCU设备指标"""
        if self.device_count == 0:
            return []
        
        metrics = []
        timestamp = datetime.now().isoformat()
        
        try:
            # 收集各种指标
            temp_data = self._get_temperature_data()
            power_data = self._get_power_data()
            util_data = self._get_utilization_data()
            memory_data = self._get_memory_data()
            clock_data = self._get_clock_data()
            fan_data = self._get_fan_data()
            
            for device_id in range(self.device_count):
                metric = DCUMetrics(
                    device_id=device_id,
                    timestamp=timestamp,
                    temperature=temp_data.get(device_id, 0.0),
                    power_usage=power_data.get(device_id, 0.0),
                    gpu_utilization=util_data.get(device_id, 0.0),
                    memory_used=memory_data.get(device_id, {}).get('used', 0.0),
                    memory_total=memory_data.get(device_id, {}).get('total', 0.0),
                    memory_utilization=memory_data.get(device_id, {}).get('utilization', 0.0),
                    clock_graphics=clock_data.get(device_id, {}).get('graphics', 0.0),
                    clock_memory=clock_data.get(device_id, {}).get('memory', 0.0),
                    fan_speed=fan_data.get(device_id, 0.0),
                    voltage=0.0  # ROCm-SMI不直接提供电压信息
                )
                
                metrics.append(metric)
                self.metrics_history[device_id].append(metric)
        
        except Exception as e:
            logger.error(f"收集DCU指标失败: {e}")
        
        return metrics
    
    def _get_temperature_data(self) -> Dict[int, float]:
        """获取温度数据"""
        try:
            result = subprocess.run(['rocm-smi', '--showtemp', '--csv'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return {}
            
            data = {}
            lines = result.stdout.strip().split('\n')
            
            for line in lines[1:]:  # 跳过标题行
                if line.strip() and 'GPU[' in line:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        try:
                            # 解析GPU ID
                            gpu_id_str = parts[0].strip()
                            device_id = int(gpu_id_str.split('[')[1].split(']')[0])
                            
                            # 解析温度值
                            temp_str = parts[2].strip()
                            if temp_str.endswith('°C'):
                                temp = float(temp_str[:-2])
                            else:
                                temp = float(temp_str)
                            
                            data[device_id] = temp
                        except (ValueError, IndexError) as e:
                            logger.debug(f"解析温度数据失败: {line}, 错误: {e}")
            
            return data
            
        except Exception as e:
            logger.error(f"获取温度数据失败: {e}")
            return {}
    
    def _get_power_data(self) -> Dict[int, float]:
        """获取功耗数据"""
        try:
            result = subprocess.run(['rocm-smi', '--showpower', '--csv'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return {}
            
            data = {}
            lines = result.stdout.strip().split('\n')
            
            for line in lines[1:]:  # 跳过标题行
                if line.strip() and 'GPU[' in line:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        try:
                            device_id = int(parts[0].strip().split('[')[1].split(']')[0])
                            power_str = parts[2].strip()
                            
                            if power_str.endswith('W'):
                                power = float(power_str[:-1])
                            else:
                                power = float(power_str)
                            
                            data[device_id] = power
                        except (ValueError, IndexError):
                            pass
            
            return data
            
        except Exception as e:
            logger.error(f"获取功耗数据失败: {e}")
            return {}
    
    def _get_utilization_data(self) -> Dict[int, float]:
        """获取利用率数据"""
        try:
            result = subprocess.run(['rocm-smi', '--showuse', '--csv'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return {}
            
            data = {}
            lines = result.stdout.strip().split('\n')
            
            for line in lines[1:]:
                if line.strip() and 'GPU[' in line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            device_id = int(parts[0].strip().split('[')[1].split(']')[0])
                            util_str = parts[1].strip()
                            
                            if util_str.endswith('%'):
                                util = float(util_str[:-1])
                            else:
                                util = float(util_str)
                            
                            data[device_id] = util
                        except (ValueError, IndexError):
                            pass
            
            return data
            
        except Exception as e:
            logger.error(f"获取利用率数据失败: {e}")
            return {}
    
    def _get_memory_data(self) -> Dict[int, Dict[str, float]]:
        """获取显存数据"""
        try:
            result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram', '--csv'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return {}
            
            data = {}
            lines = result.stdout.strip().split('\n')
            
            for line in lines[1:]:
                if line.strip() and 'GPU[' in line:
                    parts = line.split(',')
                    if len(parts) >= 4:
                        try:
                            device_id = int(parts[0].strip().split('[')[1].split(']')[0])
                            
                            # 解析显存使用量和总量 (假设格式为 "used / total")
                            memory_str = parts[2].strip()
                            if '/' in memory_str:
                                used_str, total_str = memory_str.split('/')
                                used_mb = float(used_str.strip().replace('MB', ''))
                                total_mb = float(total_str.strip().replace('MB', ''))
                                
                                used_gb = used_mb / 1024
                                total_gb = total_mb / 1024
                                utilization = (used_gb / total_gb * 100) if total_gb > 0 else 0
                                
                                data[device_id] = {
                                    'used': used_gb,
                                    'total': total_gb,
                                    'utilization': utilization
                                }
                        except (ValueError, IndexError):
                            pass
            
            return data
            
        except Exception as e:
            logger.error(f"获取显存数据失败: {e}")
            return {}
    
    def _get_clock_data(self) -> Dict[int, Dict[str, float]]:
        """获取时钟频率数据"""
        try:
            result = subprocess.run(['rocm-smi', '--showclocks', '--csv'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return {}
            
            data = {}
            lines = result.stdout.strip().split('\n')
            
            for line in lines[1:]:
                if line.strip() and 'GPU[' in line:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        try:
                            device_id = int(parts[0].strip().split('[')[1].split(']')[0])
                            
                            # 解析图形和显存时钟
                            graphics_clock = 0.0
                            memory_clock = 0.0
                            
                            # 时钟数据格式可能因版本而异，需要适配
                            if len(parts) >= 2:
                                graphics_str = parts[1].strip().replace('MHz', '')
                                graphics_clock = float(graphics_str) if graphics_str.isdigit() else 0.0
                            
                            if len(parts) >= 3:
                                memory_str = parts[2].strip().replace('MHz', '')
                                memory_clock = float(memory_str) if memory_str.isdigit() else 0.0
                            
                            data[device_id] = {
                                'graphics': graphics_clock,
                                'memory': memory_clock
                            }
                        except (ValueError, IndexError):
                            pass
            
            return data
            
        except Exception as e:
            logger.error(f"获取时钟数据失败: {e}")
            return {}
    
    def _get_fan_data(self) -> Dict[int, float]:
        """获取风扇转速数据"""
        try:
            result = subprocess.run(['rocm-smi', '--showfan', '--csv'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return {}
            
            data = {}
            lines = result.stdout.strip().split('\n')
            
            for line in lines[1:]:
                if line.strip() and 'GPU[' in line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            device_id = int(parts[0].strip().split('[')[1].split(']')[0])
                            fan_str = parts[1].strip()
                            
                            if fan_str.endswith('%'):
                                fan_speed = float(fan_str[:-1])
                            else:
                                fan_speed = float(fan_str)
                            
                            data[device_id] = fan_speed
                        except (ValueError, IndexError):
                            pass
            
            return data
            
        except Exception as e:
            logger.error(f"获取风扇数据失败: {e}")
            return {}

class SystemMonitor:
    """系统监控器"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.network_stats_prev = None
        
    def collect_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        timestamp = datetime.now().isoformat()
        
        try:
            # CPU指标
            cpu_util = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0.0
            
            # 内存指标
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            memory_util = memory.percent
            
            # 磁盘指标
            disk = psutil.disk_usage('/')
            disk_util = disk.percent
            
            # 网络指标
            network_sent, network_recv = self._get_network_speed()
            
            # 负载平均值 (仅Linux)
            try:
                load_avg = os.getloadavg()
            except:
                load_avg = (0.0, 0.0, 0.0)
            
            # 进程数量
            process_count = len(psutil.pids())
            
            metrics = SystemMetrics(
                timestamp=timestamp,
                cpu_utilization=cpu_util,
                cpu_frequency=cpu_freq,
                memory_used=memory_used_gb,
                memory_total=memory_total_gb,
                memory_utilization=memory_util,
                disk_usage=disk_util,
                network_sent=network_sent,
                network_recv=network_recv,
                load_average=load_avg,
                process_count=process_count
            )
            
            self.metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
            return SystemMetrics(
                timestamp=timestamp, cpu_utilization=0, cpu_frequency=0,
                memory_used=0, memory_total=0, memory_utilization=0,
                disk_usage=0, network_sent=0, network_recv=0,
                load_average=(0, 0, 0), process_count=0
            )
    
    def _get_network_speed(self) -> Tuple[float, float]:
        """获取网络速度 (MB/s)"""
        try:
            network_stats = psutil.net_io_counters()
            current_time = time.time()
            
            if self.network_stats_prev is None:
                self.network_stats_prev = (network_stats, current_time)
                return 0.0, 0.0
            
            prev_stats, prev_time = self.network_stats_prev
            time_delta = current_time - prev_time
            
            if time_delta > 0:
                sent_speed = (network_stats.bytes_sent - prev_stats.bytes_sent) / time_delta / (1024**2)
                recv_speed = (network_stats.bytes_recv - prev_stats.bytes_recv) / time_delta / (1024**2)
            else:
                sent_speed, recv_speed = 0.0, 0.0
            
            self.network_stats_prev = (network_stats, current_time)
            return sent_speed, recv_speed
            
        except Exception as e:
            logger.error(f"获取网络速度失败: {e}")
            return 0.0, 0.0

class PerformanceMonitor:
    """综合性能监控器"""
    
    def __init__(self, monitor_interval: float = 5.0, save_interval: float = 60.0):
        self.monitor_interval = monitor_interval
        self.save_interval = save_interval
        self.monitoring = False
        
        self.dcu_monitor = DCUMonitor()
        self.system_monitor = SystemMonitor()
        
        self.all_metrics = []
        self.monitor_thread = None
        self.save_thread = None
        
        # 创建输出目录
        self.output_dir = Path('logs/monitoring')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def start(self):
        """开始监控"""
        if self.monitoring:
            logger.warning("监控已在运行")
            return
        
        self.monitoring = True
        logger.info(f"开始性能监控，监控间隔: {self.monitor_interval}s")
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # 启动保存线程
        self.save_thread = threading.Thread(target=self._save_loop, daemon=True)
        self.save_thread.start()
    
    def stop(self):
        """停止监控"""
        if not self.monitoring:
            return
        
        logger.info("停止性能监控")
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        if self.save_thread:
            self.save_thread.join(timeout=10)
        
        # 保存最后的数据
        self._save_metrics()
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 收集DCU指标
                dcu_metrics = self.dcu_monitor.collect_metrics()
                
                # 收集系统指标
                system_metrics = self.system_monitor.collect_metrics()
                
                # 组合指标
                combined_metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'dcu_metrics': [asdict(m) for m in dcu_metrics],
                    'system_metrics': asdict(system_metrics)
                }
                
                self.all_metrics.append(combined_metrics)
                
                # 检查异常情况
                self._check_alerts(dcu_metrics, system_metrics)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(self.monitor_interval)
    
    def _save_loop(self):
        """保存循环"""
        while self.monitoring:
            time.sleep(self.save_interval)
            try:
                self._save_metrics()
            except Exception as e:
                logger.error(f"保存指标失败: {e}")
    
    def _save_metrics(self):
        """保存指标到文件"""
        if not self.all_metrics:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存JSON格式
        json_file = self.output_dir / f'metrics_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_metrics[-100:], f, ensure_ascii=False, indent=2)  # 只保存最近100条
        
        # 保存CSV格式
        self._save_metrics_csv(timestamp)
        
        logger.debug(f"指标已保存: {json_file}")
    
    def _save_metrics_csv(self, timestamp: str):
        """保存指标为CSV格式"""
        try:
            # DCU指标CSV
            dcu_csv_file = self.output_dir / f'dcu_metrics_{timestamp}.csv'
            dcu_data = []
            
            for metric in self.all_metrics[-100:]:  # 最近100条
                for dcu_metric in metric['dcu_metrics']:
                    dcu_data.append(dcu_metric)
            
            if dcu_data:
                df_dcu = pd.DataFrame(dcu_data)
                df_dcu.to_csv(dcu_csv_file, index=False)
            
            # 系统指标CSV
            system_csv_file = self.output_dir / f'system_metrics_{timestamp}.csv'
            system_data = [metric['system_metrics'] for metric in self.all_metrics[-100:]]
            
            if system_data:
                df_system = pd.DataFrame(system_data)
                df_system.to_csv(system_csv_file, index=False)
                
        except Exception as e:
            logger.error(f"保存CSV格式失败: {e}")
    
    def _check_alerts(self, dcu_metrics: List[DCUMetrics], system_metrics: SystemMetrics):
        """检查异常情况并告警"""
        try:
            # DCU异常检查
            for metric in dcu_metrics:
                # 温度过高
                if metric.temperature > 85:
                    logger.warning(f"DCU[{metric.device_id}] 温度过高: {metric.temperature}°C")
                
                # 功耗过高
                if metric.power_usage > 300:
                    logger.warning(f"DCU[{metric.device_id}] 功耗过高: {metric.power_usage}W")
                
                # 利用率异常
                if metric.gpu_utilization < 10:
                    logger.debug(f"DCU[{metric.device_id}] 利用率低: {metric.gpu_utilization}%")
                
                # 显存使用率过高
                if metric.memory_utilization > 95:
                    logger.warning(f"DCU[{metric.device_id}] 显存使用率过高: {metric.memory_utilization}%")
            
            # 系统异常检查
            if system_metrics.cpu_utilization > 90:
                logger.warning(f"CPU利用率过高: {system_metrics.cpu_utilization}%")
            
            if system_metrics.memory_utilization > 90:
                logger.warning(f"内存使用率过高: {system_metrics.memory_utilization}%")
            
            if system_metrics.disk_usage > 90:
                logger.warning(f"磁盘使用率过高: {system_metrics.disk_usage}%")
                
        except Exception as e:
            logger.error(f"异常检查失败: {e}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        if not self.all_metrics:
            return {}
        
        latest = self.all_metrics[-1]
        return {
            'timestamp': latest['timestamp'],
            'dcu_count': len(latest['dcu_metrics']),
            'dcu_status': latest['dcu_metrics'],
            'system_status': latest['system_metrics']
        }
    
    def generate_summary_report(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """生成汇总报告"""
        if not self.all_metrics:
            return {}
        
        # 计算时间范围
        now = datetime.now()
        start_time = now - timedelta(minutes=duration_minutes)
        
        # 过滤指定时间范围的数据
        filtered_metrics = []
        for metric in self.all_metrics:
            metric_time = datetime.fromisoformat(metric['timestamp'])
            if metric_time >= start_time:
                filtered_metrics.append(metric)
        
        if not filtered_metrics:
            return {}
        
        # 计算统计信息
        report = {
            'time_range': {
                'start': start_time.isoformat(),
                'end': now.isoformat(),
                'duration_minutes': duration_minutes
            },
            'dcu_summary': self._calculate_dcu_summary(filtered_metrics),
            'system_summary': self._calculate_system_summary(filtered_metrics)
        }
        
        return report
    
    def _calculate_dcu_summary(self, metrics: List[Dict]) -> Dict[str, Any]:
        """计算DCU汇总统计"""
        dcu_data = {}
        
        for metric in metrics:
            for dcu_metric in metric['dcu_metrics']:
                device_id = dcu_metric['device_id']
                if device_id not in dcu_data:
                    dcu_data[device_id] = {
                        'temperatures': [],
                        'power_usages': [],
                        'utilizations': [],
                        'memory_utilizations': []
                    }
                
                dcu_data[device_id]['temperatures'].append(dcu_metric['temperature'])
                dcu_data[device_id]['power_usages'].append(dcu_metric['power_usage'])
                dcu_data[device_id]['utilizations'].append(dcu_metric['gpu_utilization'])
                dcu_data[device_id]['memory_utilizations'].append(dcu_metric['memory_utilization'])
        
        summary = {}
        for device_id, data in dcu_data.items():
            summary[f'dcu_{device_id}'] = {
                'temperature': {
                    'avg': sum(data['temperatures']) / len(data['temperatures']),
                    'max': max(data['temperatures']),
                    'min': min(data['temperatures'])
                },
                'power_usage': {
                    'avg': sum(data['power_usages']) / len(data['power_usages']),
                    'max': max(data['power_usages']),
                    'min': min(data['power_usages'])
                },
                'utilization': {
                    'avg': sum(data['utilizations']) / len(data['utilizations']),
                    'max': max(data['utilizations']),
                    'min': min(data['utilizations'])
                },
                'memory_utilization': {
                    'avg': sum(data['memory_utilizations']) / len(data['memory_utilizations']),
                    'max': max(data['memory_utilizations']),
                    'min': min(data['memory_utilizations'])
                }
            }
        
        return summary
    
    def _calculate_system_summary(self, metrics: List[Dict]) -> Dict[str, Any]:
        """计算系统汇总统计"""
        cpu_utils = [m['system_metrics']['cpu_utilization'] for m in metrics]
        memory_utils = [m['system_metrics']['memory_utilization'] for m in metrics]
        
        return {
            'cpu_utilization': {
                'avg': sum(cpu_utils) / len(cpu_utils),
                'max': max(cpu_utils),
                'min': min(cpu_utils)
            },
            'memory_utilization': {
                'avg': sum(memory_utils) / len(memory_utils),
                'max': max(memory_utils),
                'min': min(memory_utils)
            }
        }

def main():
    """主函数 - 演示使用"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DCU性能监控器')
    parser.add_argument('--interval', type=float, default=5.0, help='监控间隔(秒)')
    parser.add_argument('--duration', type=int, default=60, help='监控持续时间(秒)')
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor(monitor_interval=args.interval)
    
    try:
        monitor.start()
        logger.info(f"开始监控，持续时间: {args.duration}秒")
        
        time.sleep(args.duration)
        
        # 生成报告
        report = monitor.generate_summary_report(duration_minutes=args.duration//60 + 1)
        print("\n=== 性能监控报告 ===")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        
    except KeyboardInterrupt:
        logger.info("监控被用户中断")
    finally:
        monitor.stop()
        logger.info("监控已停止")

if __name__ == '__main__':
    main() 