#!/usr/bin/env python3
"""
DCU性能基准测试程序
全面测试DCU在各种计算任务下的性能表现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import json
import sys
import os
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from common.utils.logger import get_logger
from common.dcu.performance_profiler import DCUProfiler

logger = get_logger(__name__)

class BenchmarkSuite:
    """DCU性能基准测试套件"""
    
    def __init__(self, device='cuda:0', warmup_runs=3, test_runs=10):
        self.device = torch.device(device)
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs
        self.results = {}
        
        print(f"🚀 初始化DCU基准测试 (设备: {device})")
        print(f"   预热次数: {warmup_runs}, 测试次数: {test_runs}")
    
    def warmup(self, func, *args, **kwargs):
        """预热函数"""
        for _ in range(self.warmup_runs):
            func(*args, **kwargs)
        torch.cuda.synchronize()
    
    def benchmark(self, func, *args, **kwargs):
        """基准测试函数"""
        # 预热
        self.warmup(func, *args, **kwargs)
        
        # 测试
        start_time = time.time()
        for _ in range(self.test_runs):
            result = func(*args, **kwargs)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / self.test_runs
        return avg_time, result
    
    def test_matrix_operations(self):
        """测试矩阵运算性能"""
        print("\n📊 矩阵运算性能测试")
        print("-" * 50)
        
        sizes = [512, 1024, 2048, 4096, 8192]
        operations = {
            'matmul': lambda a, b: torch.mm(a, b),
            'add': lambda a, b: torch.add(a, b),
            'mul': lambda a, b: torch.mul(a, b),
            'transpose': lambda a, b: torch.transpose(a, 0, 1),
        }
        
        results = {}
        
        for size in sizes:
            print(f"\n矩阵大小: {size}x{size}")
            a = torch.randn(size, size, device=self.device, dtype=torch.float32)
            b = torch.randn(size, size, device=self.device, dtype=torch.float32)
            
            size_results = {}
            
            for op_name, op_func in operations.items():
                try:
                    avg_time, _ = self.benchmark(op_func, a, b)
                    
                    if op_name == 'matmul':
                        flops = 2 * size**3
                        gflops = flops / avg_time / 1e9
                        print(f"   {op_name:10s}: {avg_time*1000:8.2f} ms ({gflops:6.1f} GFLOPS)")
                        size_results[op_name] = {'time': avg_time, 'gflops': gflops}
                    else:
                        print(f"   {op_name:10s}: {avg_time*1000:8.2f} ms")
                        size_results[op_name] = {'time': avg_time}
                        
                except Exception as e:
                    print(f"   {op_name:10s}: 失败 ({e})")
                    size_results[op_name] = {'error': str(e)}
            
            results[size] = size_results
        
        self.results['matrix_operations'] = results
    
    def test_neural_network_operations(self):
        """测试神经网络操作性能"""
        print("\n🧠 神经网络操作性能测试")
        print("-" * 50)
        
        batch_sizes = [1, 8, 32, 128]
        seq_lengths = [128, 512, 1024, 2048]
        hidden_sizes = [768, 1024, 2048, 4096]
        
        results = {}
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                for hidden_size in hidden_sizes:
                    if batch_size * seq_len * hidden_size > 100000000:  # 跳过过大的配置
                        continue
                    
                    print(f"\n配置: batch={batch_size}, seq_len={seq_len}, hidden={hidden_size}")
                    
                    # 创建输入数据
                    x = torch.randn(batch_size, seq_len, hidden_size, device=self.device)
                    
                    # 测试不同的神经网络层
                    layers = {
                        'linear': nn.Linear(hidden_size, hidden_size).to(self.device),
                        'layernorm': nn.LayerNorm(hidden_size).to(self.device),
                        'gelu': nn.GELU(),
                        'softmax': nn.Softmax(dim=-1),
                    }
                    
                    config_results = {}
                    
                    for layer_name, layer in layers.items():
                        try:
                            avg_time, _ = self.benchmark(layer, x)
                            print(f"   {layer_name:10s}: {avg_time*1000:8.2f} ms")
                            config_results[layer_name] = {'time': avg_time}
                        except Exception as e:
                            print(f"   {layer_name:10s}: 失败 ({e})")
                            config_results[layer_name] = {'error': str(e)}
                    
                    key = f"b{batch_size}_s{seq_len}_h{hidden_size}"
                    results[key] = config_results
        
        self.results['neural_network_operations'] = results
    
    def test_memory_bandwidth(self):
        """测试显存带宽"""
        print("\n💾 显存带宽测试")
        print("-" * 50)
        
        sizes_mb = [1, 10, 100, 1000, 5000]  # MB
        results = {}
        
        for size_mb in sizes_mb:
            size_elements = size_mb * 1024 * 1024 // 4  # float32 = 4 bytes
            
            print(f"\n数据大小: {size_mb} MB")
            
            # 创建数据
            data = torch.randn(size_elements, device='cpu')
            
            # 测试 CPU -> DCU 传输
            try:
                start_time = time.time()
                for _ in range(self.test_runs):
                    gpu_data = data.to(self.device)
                torch.cuda.synchronize()
                end_time = time.time()
                
                avg_time = (end_time - start_time) / self.test_runs
                bandwidth = size_mb / avg_time / 1024  # GB/s
                print(f"   CPU->DCU: {avg_time*1000:8.2f} ms ({bandwidth:6.1f} GB/s)")
                
                # 测试 DCU -> CPU 传输
                start_time = time.time()
                for _ in range(self.test_runs):
                    cpu_data = gpu_data.to('cpu')
                torch.cuda.synchronize()
                end_time = time.time()
                
                avg_time = (end_time - start_time) / self.test_runs
                bandwidth = size_mb / avg_time / 1024  # GB/s
                print(f"   DCU->CPU: {avg_time*1000:8.2f} ms ({bandwidth:6.1f} GB/s)")
                
                results[size_mb] = {
                    'cpu_to_dcu': {'time': avg_time, 'bandwidth': bandwidth},
                    'dcu_to_cpu': {'time': avg_time, 'bandwidth': bandwidth}
                }
                
            except Exception as e:
                print(f"   传输失败: {e}")
                results[size_mb] = {'error': str(e)}
        
        self.results['memory_bandwidth'] = results
    
    def test_transformer_attention(self):
        """测试Transformer注意力机制性能"""
        print("\n🔍 Transformer注意力机制测试")
        print("-" * 50)
        
        configs = [
            {'batch': 1, 'seq_len': 512, 'heads': 8, 'dim': 64},
            {'batch': 4, 'seq_len': 1024, 'heads': 12, 'dim': 64},
            {'batch': 8, 'seq_len': 2048, 'heads': 16, 'dim': 64},
        ]
        
        results = {}
        
        for config in configs:
            batch_size = config['batch']
            seq_len = config['seq_len']
            num_heads = config['heads']
            head_dim = config['dim']
            
            print(f"\n配置: batch={batch_size}, seq_len={seq_len}, heads={num_heads}, dim={head_dim}")
            
            # 创建输入
            hidden_size = num_heads * head_dim
            x = torch.randn(batch_size, seq_len, hidden_size, device=self.device)
            
            # 创建注意力层
            attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                batch_first=True
            ).to(self.device)
            
            try:
                # 测试注意力计算
                def attention_forward():
                    return attention(x, x, x)
                
                avg_time, _ = self.benchmark(attention_forward)
                
                # 计算理论FLOPS
                flops = 4 * batch_size * seq_len * seq_len * hidden_size
                gflops = flops / avg_time / 1e9
                
                print(f"   注意力计算: {avg_time*1000:8.2f} ms ({gflops:6.1f} GFLOPS)")
                
                config_key = f"b{batch_size}_s{seq_len}_h{num_heads}_d{head_dim}"
                results[config_key] = {
                    'time': avg_time,
                    'gflops': gflops,
                    'config': config
                }
                
            except Exception as e:
                print(f"   注意力计算失败: {e}")
                config_key = f"b{batch_size}_s{seq_len}_h{num_heads}_d{head_dim}"
                results[config_key] = {'error': str(e)}
        
        self.results['transformer_attention'] = results
    
    def test_mixed_precision(self):
        """测试混合精度性能"""
        print("\n⚡ 混合精度性能测试")
        print("-" * 50)
        
        size = 2048
        dtypes = [torch.float32, torch.float16]
        
        results = {}
        
        for dtype in dtypes:
            print(f"\n数据类型: {dtype}")
            
            a = torch.randn(size, size, device=self.device, dtype=dtype)
            b = torch.randn(size, size, device=self.device, dtype=dtype)
            
            # 矩阵乘法测试
            avg_time, _ = self.benchmark(torch.mm, a, b)
            flops = 2 * size**3
            gflops = flops / avg_time / 1e9
            
            print(f"   矩阵乘法: {avg_time*1000:8.2f} ms ({gflops:6.1f} GFLOPS)")
            
            results[str(dtype)] = {
                'time': avg_time,
                'gflops': gflops
            }
        
        self.results['mixed_precision'] = results
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始DCU性能基准测试")
        print("=" * 60)
        
        # 显示设备信息
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self.device)
            print(f"设备: {props.name}")
            print(f"显存: {props.total_memory / 1024**3:.1f} GB")
            print(f"计算能力: {props.major}.{props.minor}")
        
        # 运行各项测试
        test_functions = [
            self.test_matrix_operations,
            self.test_neural_network_operations,
            self.test_memory_bandwidth,
            self.test_transformer_attention,
            self.test_mixed_precision,
        ]
        
        for test_func in test_functions:
            try:
                test_func()
            except Exception as e:
                logger.error(f"测试 {test_func.__name__} 失败: {e}", exc_info=True)
                print(f"❌ 测试 {test_func.__name__} 失败: {e}")
        
        # 生成总结报告
        self.generate_summary()
    
    def generate_summary(self):
        """生成测试总结"""
        print("\n📋 测试总结")
        print("=" * 60)
        
        # 保存结果到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dcu_benchmark_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 测试完成！结果已保存到: {filename}")
        
        # 显示关键性能指标
        if 'matrix_operations' in self.results:
            matrix_results = self.results['matrix_operations']
            if '2048' in matrix_results and 'matmul' in matrix_results['2048']:
                gflops = matrix_results['2048']['matmul'].get('gflops', 0)
                print(f"🔥 矩阵乘法性能 (2048x2048): {gflops:.1f} GFLOPS")
        
        print("\n🎯 性能建议:")
        print("   - 如果性能低于预期，请检查DCU驱动版本")
        print("   - 考虑使用混合精度训练提升性能")
        print("   - 优化批次大小以充分利用DCU资源")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DCU性能基准测试')
    parser.add_argument('--device', default='cuda:0', help='DCU设备ID')
    parser.add_argument('--warmup', type=int, default=3, help='预热次数')
    parser.add_argument('--runs', type=int, default=10, help='测试次数')
    parser.add_argument('--profile', action='store_true', help='启用性能分析')
    
    args = parser.parse_args()
    
    # 检查DCU可用性
    if not torch.cuda.is_available():
        print("❌ DCU设备不可用！请检查驱动安装。")
        sys.exit(1)
    
    # 创建基准测试套件
    benchmark = BenchmarkSuite(
        device=args.device,
        warmup_runs=args.warmup,
        test_runs=args.runs
    )
    
    # 运行测试
    if args.profile:
        with DCUProfiler() as profiler:
            benchmark.run_all_tests()
        profiler.generate_report('dcu_benchmark_profile.html')
        print(f"📊 性能分析报告已生成: dcu_benchmark_profile.html")
    else:
        benchmark.run_all_tests()

if __name__ == "__main__":
    main() 