#!/usr/bin/env python3
"""
海光DCU性能监控工具
实时监控DCU设备状态、内存使用、温度等信息
"""

import time
import subprocess
import json
import argparse
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional
import threading

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class DCUMonitor:
    def __init__(self, interval=2, log_file=None, json_output=False):
        """
        DCU性能监控器
        
        Args:
            interval: 监控间隔(秒)
            log_file: 日志文件路径
            json_output: 是否输出JSON格式
        """
        self.interval = interval
        self.log_file = log_file
        self.json_output = json_output
        self.running = True
        self.start_time = time.time()
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # 初始化日志文件
        if self.log_file:
            with open(self.log_file, 'w') as f:
                f.write("timestamp,device_id,gpu_util,memory_used,memory_total,temperature,power\n")
    
    def _signal_handler(self, signum, frame):
        """信号处理函数"""
        print("\n正在停止监控...")
        self.running = False
    
    def get_dcu_info_via_smi(self) -> List[Dict]:
        """通过hy-smi命令获取DCU信息"""
        try:
            # 尝试运行hy-smi
            result = subprocess.run(['hy-smi', '--json'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return []
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, 
                FileNotFoundError, json.JSONDecodeError):
            return []
    
    def get_dcu_info_via_torch(self) -> List[Dict]:
        """通过PyTorch获取DCU信息"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return []
        
        devices = []
        for i in range(torch.cuda.device_count()):
            try:
                # 获取设备属性
                props = torch.cuda.get_device_properties(i)
                
                # 获取内存信息
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_total = props.total_memory
                
                device_info = {
                    'device_id': i,
                    'name': props.name,
                    'memory_allocated': memory_allocated,
                    'memory_reserved': memory_reserved,
                    'memory_total': memory_total,
                    'memory_util': (memory_allocated / memory_total) * 100,
                    'compute_capability': f"{props.major}.{props.minor}",
                    'multiprocessor_count': props.multi_processor_count
                }
                
                devices.append(device_info)
                
            except Exception as e:
                print(f"获取设备 {i} 信息失败: {e}")
                
        return devices
    
    def get_system_info(self) -> Dict:
        """获取系统信息"""
        system_info = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': 0,
            'memory_percent': 0,
            'memory_available': 0,
            'load_average': [0, 0, 0]
        }
        
        if PSUTIL_AVAILABLE:
            try:
                # CPU使用率
                system_info['cpu_percent'] = psutil.cpu_percent(interval=None)
                
                # 内存信息
                memory = psutil.virtual_memory()
                system_info['memory_percent'] = memory.percent
                system_info['memory_available'] = memory.available
                
                # 负载平均值
                system_info['load_average'] = list(psutil.getloadavg())
                
            except Exception as e:
                print(f"获取系统信息失败: {e}")
        
        return system_info
    
    def format_bytes(self, bytes_val):
        """格式化字节数"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f} PB"
    
    def display_header(self):
        """显示表头"""
        if self.json_output:
            return
            
        print("\n" + "="*80)
        print(f"{'DCU监控器':<20} {'运行时间':<15} {'刷新间隔':<10} {'时间':<20}")
        print("="*80)
        
        print(f"{'设备':<6} {'名称':<15} {'GPU使用':<10} {'显存':<20} {'温度':<8} {'功耗':<8}")
        print("-"*80)
    
    def display_devices(self, devices):
        """显示设备信息"""
        current_time = datetime.now().strftime("%H:%M:%S")
        runtime = time.time() - self.start_time
        
        if self.json_output:
            output = {
                'timestamp': current_time,
                'runtime': runtime,
                'devices': devices
            }
            print(json.dumps(output, indent=2))
            return
        
        # 清屏并显示表头
        print("\033[2J\033[H", end="")  # 清屏并移动光标到左上角
        
        print(f"🖥️  DCU监控器 v1.0    ⏱️  运行时间: {runtime:.0f}s    🔄 间隔: {self.interval}s    📅 {current_time}")
        print("="*80)
        
        if not devices:
            print("❌ 未检测到DCU设备或获取信息失败")
            print("   请检查:")
            print("   1. DCU驱动是否正确安装")
            print("   2. hy-smi命令是否可用")
            print("   3. PyTorch是否支持DCU")
            return
        
        print(f"{'设备':<6} {'名称':<15} {'GPU使用':<10} {'显存使用':<25} {'温度':<8} {'功耗':<8}")
        print("-"*80)
        
        for device in devices:
            device_id = device.get('device_id', 'N/A')
            name = device.get('name', 'Unknown')[:14]
            
            # GPU使用率
            gpu_util = device.get('gpu_utilization', device.get('memory_util', 0))
            gpu_util_str = f"{gpu_util:.1f}%" if gpu_util else "N/A"
            
            # 内存信息
            memory_used = device.get('memory_allocated', device.get('memory_used', 0))
            memory_total = device.get('memory_total', 1)
            memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0
            
            memory_str = f"{self.format_bytes(memory_used)}/{self.format_bytes(memory_total)} ({memory_percent:.1f}%)"
            
            # 温度
            temperature = device.get('temperature', 0)
            temp_str = f"{temperature}°C" if temperature else "N/A"
            
            # 功耗
            power = device.get('power', 0)
            power_str = f"{power}W" if power else "N/A"
            
            print(f"DCU{device_id:<3} {name:<15} {gpu_util_str:<10} {memory_str:<25} {temp_str:<8} {power_str:<8}")
        
        # 显示系统信息
        system_info = self.get_system_info()
        print("\n" + "-"*80)
        print(f"💻 系统状态:")
        print(f"   CPU使用率: {system_info['cpu_percent']:.1f}%")
        print(f"   内存使用率: {system_info['memory_percent']:.1f}%")
        print(f"   可用内存: {self.format_bytes(system_info['memory_available'])}")
        print(f"   负载平均: {system_info['load_average'][0]:.2f}, {system_info['load_average'][1]:.2f}, {system_info['load_average'][2]:.2f}")
    
    def log_to_file(self, devices):
        """记录数据到文件"""
        if not self.log_file:
            return
            
        timestamp = datetime.now().isoformat()
        
        try:
            with open(self.log_file, 'a') as f:
                for device in devices:
                    line = f"{timestamp},{device.get('device_id', 'N/A')}," \
                           f"{device.get('gpu_utilization', 0)}," \
                           f"{device.get('memory_used', 0)}," \
                           f"{device.get('memory_total', 0)}," \
                           f"{device.get('temperature', 0)}," \
                           f"{device.get('power', 0)}\n"
                    f.write(line)
        except Exception as e:
            print(f"写入日志文件失败: {e}")
    
    def run(self):
        """运行监控循环"""
        print("🚀 启动DCU监控器...")
        print("按 Ctrl+C 停止监控")
        
        if not self.json_output:
            self.display_header()
        
        while self.running:
            try:
                # 尝试通过hy-smi获取信息
                devices = self.get_dcu_info_via_smi()
                
                # 如果hy-smi失败，尝试通过PyTorch获取
                if not devices:
                    devices = self.get_dcu_info_via_torch()
                
                # 显示信息
                self.display_devices(devices)
                
                # 记录到文件
                self.log_to_file(devices)
                
                # 等待下一次刷新
                time.sleep(self.interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"监控过程中出现错误: {e}")
                time.sleep(self.interval)
        
        print("\n监控已停止")

class DCUBenchmark:
    """DCU性能基准测试"""
    
    def __init__(self):
        self.device_count = 0
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device_count = torch.cuda.device_count()
    
    def memory_bandwidth_test(self, device_id=0, size_mb=1024):
        """内存带宽测试"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            print("PyTorch或DCU不可用")
            return
        
        print(f"📊 DCU {device_id} 内存带宽测试 (数据大小: {size_mb} MB)")
        
        device = torch.device(f'cuda:{device_id}')
        size = size_mb * 1024 * 1024 // 4  # 转换为float32数量
        
        # 主机到设备
        host_data = torch.randn(size)
        
        torch.cuda.synchronize()
        start_time = time.time()
        device_data = host_data.to(device)
        torch.cuda.synchronize()
        h2d_time = time.time() - start_time
        
        h2d_bandwidth = (size_mb / h2d_time) * 1000  # MB/s
        
        # 设备到主机
        torch.cuda.synchronize()
        start_time = time.time()
        host_result = device_data.cpu()
        torch.cuda.synchronize()
        d2h_time = time.time() - start_time
        
        d2h_bandwidth = (size_mb / d2h_time) * 1000  # MB/s
        
        print(f"   主机到设备: {h2d_bandwidth:.1f} MB/s")
        print(f"   设备到主机: {d2h_bandwidth:.1f} MB/s")
        
        return h2d_bandwidth, d2h_bandwidth
    
    def compute_performance_test(self, device_id=0, matrix_size=2048):
        """计算性能测试"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            print("PyTorch或DCU不可用")
            return
        
        print(f"🧮 DCU {device_id} 计算性能测试 (矩阵大小: {matrix_size}x{matrix_size})")
        
        device = torch.device(f'cuda:{device_id}')
        
        # 生成测试数据
        a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
        b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
        
        # 预热
        for _ in range(10):
            _ = torch.mm(a, b)
        torch.cuda.synchronize()
        
        # 性能测试
        iterations = 100
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(iterations):
            c = torch.mm(a, b)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        # 计算FLOPS
        flops = 2 * matrix_size**3  # 矩阵乘法的浮点运算数
        gflops = flops / (avg_time * 1e9)
        
        print(f"   平均时间: {avg_time*1000:.2f} ms")
        print(f"   性能: {gflops:.1f} GFLOPS")
        
        return gflops
    
    def run_full_benchmark(self):
        """运行完整的基准测试"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            print("❌ PyTorch或DCU不可用，无法运行基准测试")
            return
        
        print("🚀 开始DCU完整基准测试...")
        print(f"检测到 {self.device_count} 个DCU设备\n")
        
        for device_id in range(self.device_count):
            print(f"{'='*60}")
            print(f"设备 {device_id}: {torch.cuda.get_device_name(device_id)}")
            print(f"{'='*60}")
            
            # 内存带宽测试
            self.memory_bandwidth_test(device_id)
            print()
            
            # 计算性能测试 - 不同矩阵大小
            for size in [1024, 2048, 4096]:
                try:
                    self.compute_performance_test(device_id, size)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"   矩阵大小 {size}: 内存不足，跳过")
                        break
                    else:
                        raise e
            print()

def main():
    parser = argparse.ArgumentParser(description="海光DCU性能监控工具")
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 监控命令
    monitor_parser = subparsers.add_parser('monitor', help='实时监控DCU状态')
    monitor_parser.add_argument('-i', '--interval', type=float, default=2.0, 
                              help='监控间隔(秒), 默认2秒')
    monitor_parser.add_argument('-l', '--log', type=str, 
                              help='日志文件路径')
    monitor_parser.add_argument('-j', '--json', action='store_true',
                              help='输出JSON格式')
    
    # 基准测试命令
    benchmark_parser = subparsers.add_parser('benchmark', help='运行性能基准测试')
    benchmark_parser.add_argument('-d', '--device', type=int, default=0,
                                help='测试的设备ID')
    benchmark_parser.add_argument('-s', '--size', type=int, default=2048,
                                help='矩阵大小')
    benchmark_parser.add_argument('--memory-test', action='store_true',
                                help='仅运行内存带宽测试')
    benchmark_parser.add_argument('--compute-test', action='store_true',
                                help='仅运行计算性能测试')
    
    # 信息命令
    info_parser = subparsers.add_parser('info', help='显示DCU设备信息')
    
    args = parser.parse_args()
    
    if args.command == 'monitor':
        monitor = DCUMonitor(
            interval=args.interval,
            log_file=args.log,
            json_output=args.json
        )
        monitor.run()
        
    elif args.command == 'benchmark':
        benchmark = DCUBenchmark()
        
        if args.memory_test:
            benchmark.memory_bandwidth_test(args.device)
        elif args.compute_test:
            benchmark.compute_performance_test(args.device, args.size)
        else:
            benchmark.run_full_benchmark()
            
    elif args.command == 'info':
        monitor = DCUMonitor()
        devices = monitor.get_dcu_info_via_torch()
        
        if devices:
            print("🖥️  DCU设备信息:")
            print("="*50)
            for device in devices:
                print(f"设备 {device['device_id']}: {device['name']}")
                print(f"  总内存: {monitor.format_bytes(device['memory_total'])}")
                print(f"  计算能力: {device['compute_capability']}")
                print(f"  多处理器数量: {device['multiprocessor_count']}")
                print()
        else:
            print("❌ 未检测到DCU设备")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 