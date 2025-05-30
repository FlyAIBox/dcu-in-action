#!/usr/bin/env python3
"""
DCU环境验证程序
验证DCU设备是否正常工作，并展示基本的DCU操作
"""

import torch
import time
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from common.dcu.device_manager import DCUDeviceManager
from common.utils.logger import get_logger

logger = get_logger(__name__)

def check_dcu_availability():
    """检查DCU设备可用性"""
    print("🔍 检查DCU设备可用性...")
    
    # 检查PyTorch是否支持CUDA（DCU）
    if not torch.cuda.is_available():
        print("❌ DCU设备不可用！请检查驱动安装。")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"✅ 检测到 {device_count} 个DCU设备")
    
    # 显示每个设备的信息
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"   DCU {i}: {props.name}")
        print(f"   - 显存: {props.total_memory / 1024**3:.1f} GB")
        print(f"   - 计算能力: {props.major}.{props.minor}")
    
    return True

def test_basic_operations():
    """测试基本的DCU操作"""
    print("\n🧪 测试基本DCU操作...")
    
    device = torch.device('cuda:0')
    
    # 创建张量
    print("   创建张量...")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    
    # 矩阵乘法
    print("   执行矩阵乘法...")
    start_time = time.time()
    c = torch.mm(a, b)
    torch.cuda.synchronize()  # 等待DCU操作完成
    end_time = time.time()
    
    print(f"   ✅ 矩阵乘法完成，耗时: {(end_time - start_time)*1000:.2f} ms")
    print(f"   结果形状: {c.shape}")
    print(f"   结果均值: {c.mean().item():.4f}")

def test_memory_management():
    """测试显存管理"""
    print("\n💾 测试显存管理...")
    
    device = torch.device('cuda:0')
    
    # 显示初始显存状态
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    cached = torch.cuda.memory_reserved(device) / 1024**3
    print(f"   初始显存 - 已分配: {allocated:.2f} GB, 已缓存: {cached:.2f} GB")
    
    # 分配大量显存
    print("   分配显存...")
    tensors = []
    for i in range(10):
        tensor = torch.randn(1000, 1000, device=device)
        tensors.append(tensor)
    
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    cached = torch.cuda.memory_reserved(device) / 1024**3
    print(f"   分配后显存 - 已分配: {allocated:.2f} GB, 已缓存: {cached:.2f} GB")
    
    # 清理显存
    print("   清理显存...")
    del tensors
    torch.cuda.empty_cache()
    
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    cached = torch.cuda.memory_reserved(device) / 1024**3
    print(f"   清理后显存 - 已分配: {allocated:.2f} GB, 已缓存: {cached:.2f} GB")

def test_multi_device():
    """测试多设备操作"""
    device_count = torch.cuda.device_count()
    if device_count < 2:
        print("\n⚠️  只有一个DCU设备，跳过多设备测试")
        return
    
    print(f"\n🔗 测试多设备操作（{device_count}个设备）...")
    
    # 在不同设备上创建张量
    tensors = []
    for i in range(min(device_count, 4)):  # 最多测试4个设备
        device = torch.device(f'cuda:{i}')
        tensor = torch.randn(500, 500, device=device)
        tensors.append(tensor)
        print(f"   在DCU {i}上创建张量: {tensor.shape}")
    
    # 设备间数据传输
    if len(tensors) >= 2:
        print("   测试设备间数据传输...")
        start_time = time.time()
        tensor_copy = tensors[0].to('cuda:1')
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"   ✅ 数据传输完成，耗时: {(end_time - start_time)*1000:.2f} ms")

def benchmark_performance():
    """性能基准测试"""
    print("\n📊 性能基准测试...")
    
    device = torch.device('cuda:0')
    sizes = [512, 1024, 2048, 4096]
    
    print("   矩阵大小 | 耗时(ms) | GFLOPS")
    print("   ---------|----------|--------")
    
    for size in sizes:
        # 创建随机矩阵
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # 预热
        for _ in range(3):
            torch.mm(a, b)
        torch.cuda.synchronize()
        
        # 性能测试
        start_time = time.time()
        for _ in range(10):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10 * 1000  # ms
        flops = 2 * size**3  # 矩阵乘法的浮点运算次数
        gflops = flops / (avg_time / 1000) / 1e9
        
        print(f"   {size:8d} | {avg_time:8.2f} | {gflops:7.1f}")

def main():
    """主函数"""
    print("🚀 DCU-in-Action 环境验证程序")
    print("=" * 50)
    
    # 检查DCU可用性
    if not check_dcu_availability():
        sys.exit(1)
    
    try:
        # 基本操作测试
        test_basic_operations()
        
        # 显存管理测试
        test_memory_management()
        
        # 多设备测试
        test_multi_device()
        
        # 性能基准测试
        benchmark_performance()
        
        print("\n🎉 所有测试通过！DCU环境工作正常。")
        print("\n📚 接下来您可以：")
        print("   1. 运行训练示例: cd examples/training && python train_example.py")
        print("   2. 尝试模型微调: cd examples/finetuning && python finetune_example.py")
        print("   3. 启动推理服务: cd examples/inference && python inference_server.py")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        logger.error(f"DCU测试失败: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 