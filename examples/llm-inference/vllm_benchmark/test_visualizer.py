#!/usr/bin/env python3
"""
可视化工具测试脚本
验证BenchmarkVisualizer类的各项功能是否正常工作
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """创建示例数据用于测试"""
    print("📝 创建示例数据...")
    
    # 模拟基准测试数据
    np.random.seed(42)
    n_samples = 20
    
    data = {
        'date': [f"20250102-{str(10 + i).zfill(4)}00" for i in range(n_samples)],
        'backend': ['vllm'] * n_samples,
        'model_id': ['test-model'] * n_samples,
        'tokenizer_id': ['test-model'] * n_samples,
        'num_prompts': np.random.choice([10, 40, 80, 160, 320], n_samples),
        'request_rate': ['inf'] * n_samples,
        'burstiness': [1.0] * n_samples,
        'max_concurrency': np.random.choice([1, 4, 8, 16, 32, 48], n_samples),
        'duration': np.random.uniform(100, 2000, n_samples),
        'completed': np.random.choice([10, 40, 80, 160, 320], n_samples),
        'total_input_tokens': np.random.uniform(5000, 50000, n_samples),
        'total_output_tokens': np.random.uniform(5000, 50000, n_samples),
        'request_throughput': np.random.uniform(0.1, 1.5, n_samples),
        'request_goodput:': [''] * n_samples,
        'output_throughput': np.random.uniform(50, 400, n_samples),
        'total_token_throughput': np.random.uniform(100, 800, n_samples),
        'mean_ttft_ms': np.random.uniform(100, 5000, n_samples),
        'median_ttft_ms': np.random.uniform(100, 5000, n_samples),
        'std_ttft_ms': np.random.uniform(10, 1000, n_samples),
        'p99_ttft_ms': np.random.uniform(200, 8000, n_samples),
        'mean_tpot_ms': np.random.uniform(30, 100, n_samples),
        'median_tpot_ms': np.random.uniform(30, 100, n_samples),
        'std_tpot_ms': np.random.uniform(1, 50, n_samples),
        'p99_tpot_ms': np.random.uniform(40, 150, n_samples),
        'mean_itl_ms': np.random.uniform(30, 100, n_samples),
        'median_itl_ms': np.random.uniform(30, 100, n_samples),
        'std_itl_ms': np.random.uniform(1, 50, n_samples),
        'p99_itl_ms': np.random.uniform(40, 150, n_samples),
        'mean_e2el_ms': np.random.uniform(5000, 50000, n_samples),
        'median_e2el_ms': np.random.uniform(5000, 50000, n_samples),
        'std_e2el_ms': np.random.uniform(100, 5000, n_samples),
        'p99_e2el_ms': np.random.uniform(10000, 80000, n_samples),
        'input_len': np.random.choice([256, 512, 1024, 2048], n_samples),
        'output_len': np.random.choice([256, 512, 1024, 2048], n_samples),
        'filename': [f'test_run_{i}.json' for i in range(n_samples)]
    }
    
    df = pd.DataFrame(data)
    
    # 确保results目录存在
    os.makedirs('results', exist_ok=True)
    
    # 保存测试数据
    test_csv = 'results/test_aggregate_results.csv'
    df.to_csv(test_csv, index=False)
    
    print(f"✅ 示例数据已创建: {test_csv}")
    return test_csv

def test_dependencies():
    """测试依赖包"""
    print("📦 测试依赖包...")
    
    required_packages = [
        'pandas',
        'numpy', 
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - 未安装")
            missing_packages.append(package)
    
    # 测试plotly
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print(f"  ✅ plotly")
    except ImportError:
        print(f"  ❌ plotly - 未安装")
        missing_packages.append('plotly')
    
    if missing_packages:
        print(f"\n❌ 缺少依赖包: {missing_packages}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ 所有依赖包正常")
    return True

def test_visualizer_basic():
    """测试基础功能"""
    print("\n🔧 测试基础功能...")
    
    try:
        from benchmark_visualizer import BenchmarkVisualizer
        
        # 创建示例数据
        test_csv = create_sample_data()
        
        # 初始化可视化器
        visualizer = BenchmarkVisualizer(csv_path=test_csv)
        
        if visualizer.df is None:
            print("❌ 数据加载失败")
            return False
        
        print(f"✅ 数据加载成功: {len(visualizer.df)} 条记录")
        return True
        
    except Exception as e:
        print(f"❌ 基础功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 vLLM基准测试可视化工具 - 快速测试")
    print("=" * 50)
    
    tests = [
        ("依赖包检查", test_dependencies),
        ("基础功能", test_visualizer_basic)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        
        try:
            if test_func():
                print(f"✅ {test_name} - 通过")
                passed += 1
            else:
                print(f"❌ {test_name} - 失败")
        except Exception as e:
            print(f"❌ {test_name} - 异常: {e}")
    
    print(f"\n🏁 测试完成: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 基础测试通过！可视化工具已就绪。")
        print("\n💡 使用提示:")
        print("   python3 visualize.py              # 生成所有图表")
        print("   python3 example_usage.py          # 查看使用示例")
    else:
        print(f"⚠️ {total - passed} 个测试失败，请安装依赖包。")

if __name__ == "__main__":
    main() 