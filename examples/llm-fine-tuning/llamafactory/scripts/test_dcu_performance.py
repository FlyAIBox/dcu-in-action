#!/usr/bin/env python3
"""
DCU k100-AI 性能测试脚本
用于测试海光DCU k100-AI加速卡的计算性能和显存带宽
"""

import torch
import time
import gc
import argparse
import sys
from typing import List, Dict, Any
import numpy as np

def check_dcu_availability() -> bool:
    """检查DCU是否可用"""
    if not torch.cuda.is_available():
        print("❌ DCU不可用，请检查驱动安装")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"✅ 发现 {device_count} 个DCU设备")
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        device_props = torch.cuda.get_device_properties(i)
        memory_gb = device_props.total_memory / (1024**3)
        print(f"  DCU {i}: {device_name}")
        print(f"    显存容量: {memory_gb:.1f} GB")
        print(f"    计算能力: {device_props.major}.{device_props.minor}")
    
    return True

def test_matrix_performance(device_id: int = 0, sizes: List[int] = None) -> Dict[str, Any]:
    """测试矩阵运算性能"""
    if sizes is None:
        sizes = [1024, 2048, 4096, 8192]
    
    print(f"\n🧮 DCU {device_id} 矩阵运算性能测试:")
    print("=" * 60)
    
    results = {}
    device = f'cuda:{device_id}'
    
    for size in sizes:
        try:
            # 生成随机矩阵
            a = torch.randn(size, size, device=device, dtype=torch.float16)
            b = torch.randn(size, size, device=device, dtype=torch.float16)
            
            # 预热GPU
            for _ in range(10):
                _ = torch.mm(a, b)
            torch.cuda.synchronize()
            
            # 性能测试
            iterations = 100
            start_time = time.time()
            for _ in range(iterations):
                c = torch.mm(a, b)
            torch.cuda.synchronize()
            end_time = time.time()
            
            elapsed = end_time - start_time
            # 计算TFLOPS: 每次矩阵乘法的运算量为 2*size^3
            flops = 2 * size**3 * iterations
            tflops = flops / (elapsed * 1e12)
            
            results[size] = {
                'elapsed_time': elapsed,
                'tflops': tflops,
                'memory_used': torch.cuda.memory_allocated(device_id) / (1024**3)
            }
            
            print(f"  {size:4d}x{size:4d}: {elapsed:6.3f}s, {tflops:6.2f} TFLOPS, "
                  f"显存: {results[size]['memory_used']:.2f} GB")
            
            # 清理内存
            del a, b, c
            gc.collect()
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"  {size:4d}x{size:4d}: ❌ 显存不足")
            results[size] = {'error': 'OutOfMemoryError'}
        except Exception as e:
            print(f"  {size:4d}x{size:4d}: ❌ 错误: {str(e)}")
            results[size] = {'error': str(e)}
    
    return results

def test_memory_bandwidth(device_id: int = 0, sizes_mb: List[int] = None) -> Dict[str, Any]:
    """测试显存带宽"""
    if sizes_mb is None:
        sizes_mb = [64, 128, 256, 512, 1024, 2048]
    
    print(f"\n💾 DCU {device_id} 显存带宽测试:")
    print("=" * 60)
    
    results = {}
    device = f'cuda:{device_id}'
    
    for size_mb in sizes_mb:
        try:
            elements = size_mb * 1024 * 1024 // 4  # float32
            data = torch.randn(elements, device=device, dtype=torch.float32)
            
            # 预热
            for _ in range(10):
                _ = data.sum()
            torch.cuda.synchronize()
            
            # 带宽测试
            iterations = 100
            start_time = time.time()
            for _ in range(iterations):
                result = data.sum()
            torch.cuda.synchronize()
            end_time = time.time()
            
            elapsed = end_time - start_time
            # 计算带宽: 每次操作读取size_mb数据
            bandwidth_gbps = (size_mb * iterations) / elapsed / 1024
            
            results[size_mb] = {
                'elapsed_time': elapsed,
                'bandwidth_gbps': bandwidth_gbps,
                'memory_used': torch.cuda.memory_allocated(device_id) / (1024**3)
            }
            
            print(f"  {size_mb:4d}MB: {bandwidth_gbps:6.2f} GB/s, "
                  f"显存: {results[size_mb]['memory_used']:.2f} GB")
            
            del data, result
            gc.collect()
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"  {size_mb:4d}MB: ❌ 显存不足")
            results[size_mb] = {'error': 'OutOfMemoryError'}
        except Exception as e:
            print(f"  {size_mb:4d}MB: ❌ 错误: {str(e)}")
            results[size_mb] = {'error': str(e)}
    
    return results

def test_mixed_precision(device_id: int = 0) -> Dict[str, Any]:
    """测试混合精度性能"""
    print(f"\n🎯 DCU {device_id} 混合精度测试:")
    print("=" * 60)
    
    results = {}
    device = f'cuda:{device_id}'
    size = 4096
    iterations = 50
    
    # 测试不同精度
    dtypes = [
        ('FP32', torch.float32),
        ('FP16', torch.float16),
        ('BF16', torch.bfloat16),
    ]
    
    for dtype_name, dtype in dtypes:
        try:
            # 生成矩阵
            a = torch.randn(size, size, device=device, dtype=dtype)
            b = torch.randn(size, size, device=device, dtype=dtype)
            
            # 预热
            for _ in range(5):
                _ = torch.mm(a, b)
            torch.cuda.synchronize()
            
            # 性能测试
            start_time = time.time()
            for _ in range(iterations):
                c = torch.mm(a, b)
            torch.cuda.synchronize()
            end_time = time.time()
            
            elapsed = end_time - start_time
            flops = 2 * size**3 * iterations
            tflops = flops / (elapsed * 1e12)
            memory_used = torch.cuda.memory_allocated(device_id) / (1024**3)
            
            results[dtype_name] = {
                'elapsed_time': elapsed,
                'tflops': tflops,
                'memory_used': memory_used
            }
            
            print(f"  {dtype_name}: {elapsed:6.3f}s, {tflops:6.2f} TFLOPS, "
                  f"显存: {memory_used:.2f} GB")
            
            del a, b, c
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  {dtype_name}: ❌ 错误: {str(e)}")
            results[dtype_name] = {'error': str(e)}
    
    return results

def benchmark_llm_workload(device_id: int = 0) -> Dict[str, Any]:
    """模拟LLM工作负载性能测试"""
    print(f"\n🤖 DCU {device_id} LLM工作负载模拟:")
    print("=" * 60)
    
    device = f'cuda:{device_id}'
    results = {}
    
    # 模拟不同的LLM参数配置
    configs = [
        {'name': '3B模型', 'seq_len': 2048, 'hidden_size': 2048, 'batch_size': 8},
        {'name': '7B模型', 'seq_len': 2048, 'hidden_size': 4096, 'batch_size': 4},
        {'name': '14B模型', 'seq_len': 2048, 'hidden_size': 5120, 'batch_size': 2},
    ]
    
    for config in configs:
        try:
            seq_len = config['seq_len']
            hidden_size = config['hidden_size']
            batch_size = config['batch_size']
            
            # 模拟注意力机制计算
            query = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
            key = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
            value = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
            
            # 预热
            for _ in range(5):
                # 注意力计算
                scores = torch.matmul(query, key.transpose(-2, -1)) / (hidden_size ** 0.5)
                attn_weights = torch.softmax(scores, dim=-1)
                output = torch.matmul(attn_weights, value)
            torch.cuda.synchronize()
            
            # 性能测试
            iterations = 20
            start_time = time.time()
            for _ in range(iterations):
                scores = torch.matmul(query, key.transpose(-2, -1)) / (hidden_size ** 0.5)
                attn_weights = torch.softmax(scores, dim=-1)
                output = torch.matmul(attn_weights, value)
            torch.cuda.synchronize()
            end_time = time.time()
            
            elapsed = end_time - start_time
            tokens_per_second = (batch_size * seq_len * iterations) / elapsed
            memory_used = torch.cuda.memory_allocated(device_id) / (1024**3)
            
            results[config['name']] = {
                'elapsed_time': elapsed,
                'tokens_per_second': tokens_per_second,
                'memory_used': memory_used
            }
            
            print(f"  {config['name']:8s}: {tokens_per_second:6.0f} tok/s, "
                  f"显存: {memory_used:.2f} GB")
            
            del query, key, value, scores, attn_weights, output
            gc.collect()
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"  {config['name']:8s}: ❌ 显存不足")
            results[config['name']] = {'error': 'OutOfMemoryError'}
        except Exception as e:
            print(f"  {config['name']:8s}: ❌ 错误: {str(e)}")
            results[config['name']] = {'error': str(e)}
    
    return results

def generate_report(all_results: Dict[str, Any], device_id: int = 0):
    """生成性能测试报告"""
    print(f"\n📊 DCU {device_id} 性能测试报告")
    print("=" * 80)
    
    # 设备信息
    device_name = torch.cuda.get_device_name(device_id)
    device_props = torch.cuda.get_device_properties(device_id)
    memory_gb = device_props.total_memory / (1024**3)
    
    print(f"设备名称: {device_name}")
    print(f"显存容量: {memory_gb:.1f} GB")
    print(f"计算能力: {device_props.major}.{device_props.minor}")
    
    # 峰值性能
    if 'matrix_results' in all_results:
        matrix_results = all_results['matrix_results']
        max_tflops = max([r.get('tflops', 0) for r in matrix_results.values() if 'tflops' in r])
        print(f"峰值算力: {max_tflops:.2f} TFLOPS")
    
    if 'bandwidth_results' in all_results:
        bandwidth_results = all_results['bandwidth_results']
        max_bandwidth = max([r.get('bandwidth_gbps', 0) for r in bandwidth_results.values() if 'bandwidth_gbps' in r])
        print(f"峰值带宽: {max_bandwidth:.2f} GB/s")
    
    # LLM性能
    if 'llm_results' in all_results:
        print(f"\nLLM工作负载性能:")
        for model_name, result in all_results['llm_results'].items():
            if 'tokens_per_second' in result:
                print(f"  {model_name}: {result['tokens_per_second']:.0f} tokens/s")
    
    # 推荐配置
    print(f"\n💡 基于测试结果的推荐配置:")
    print(f"  - 推荐精度: BF16 (DCU k100-AI原生支持)")
    print(f"  - 推荐batch_size: 4-8 (根据模型大小)")
    print(f"  - 推荐sequence_length: 2048")
    print(f"  - 显存利用建议: 使用80-85%显存以保持稳定性")

def main():
    parser = argparse.ArgumentParser(description='DCU k100-AI 性能测试工具')
    parser.add_argument('--device', type=int, default=0, help='DCU设备ID (默认: 0)')
    parser.add_argument('--matrix-only', action='store_true', help='仅运行矩阵性能测试')
    parser.add_argument('--bandwidth-only', action='store_true', help='仅运行带宽测试')
    parser.add_argument('--llm-only', action='store_true', help='仅运行LLM工作负载测试')
    parser.add_argument('--quick', action='store_true', help='快速测试模式')
    
    args = parser.parse_args()
    
    print("🚀 DCU k100-AI 性能测试工具")
    print("=" * 80)
    
    # 检查DCU可用性
    if not check_dcu_availability():
        sys.exit(1)
    
    device_id = args.device
    if device_id >= torch.cuda.device_count():
        print(f"❌ 设备ID {device_id} 超出范围")
        sys.exit(1)
    
    torch.cuda.set_device(device_id)
    all_results = {}
    
    try:
        # 根据参数运行不同测试
        if args.matrix_only:
            sizes = [2048, 4096] if args.quick else [1024, 2048, 4096, 8192]
            all_results['matrix_results'] = test_matrix_performance(device_id, sizes)
        elif args.bandwidth_only:
            sizes = [256, 512, 1024] if args.quick else [64, 128, 256, 512, 1024, 2048]
            all_results['bandwidth_results'] = test_memory_bandwidth(device_id, sizes)
        elif args.llm_only:
            all_results['llm_results'] = benchmark_llm_workload(device_id)
        else:
            # 完整测试
            sizes = [2048, 4096] if args.quick else [1024, 2048, 4096, 8192]
            bandwidth_sizes = [256, 512, 1024] if args.quick else [128, 256, 512, 1024]
            
            all_results['matrix_results'] = test_matrix_performance(device_id, sizes)
            all_results['bandwidth_results'] = test_memory_bandwidth(device_id, bandwidth_sizes)
            all_results['mixed_precision_results'] = test_mixed_precision(device_id)
            all_results['llm_results'] = benchmark_llm_workload(device_id)
        
        # 生成报告
        generate_report(all_results, device_id)
        
    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        sys.exit(1)
    
    print(f"\n✅ 测试完成！")

if __name__ == "__main__":
    main() 