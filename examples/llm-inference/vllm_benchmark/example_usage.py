#!/usr/bin/env python3
"""
vLLM基准测试可视化工具使用示例

演示如何使用BenchmarkVisualizer类进行数据分析和可视化
"""

from benchmark_visualizer import BenchmarkVisualizer
import pandas as pd

def demo_basic_usage():
    """基础使用演示"""
    print("=" * 60)
    print("🎯 vLLM基准测试可视化工具 - 基础使用演示")
    print("=" * 60)
    
    # 创建可视化器实例
    visualizer = BenchmarkVisualizer()
    
    if visualizer.df is not None:
        # 显示数据摘要
        print(visualizer.get_data_summary())
        
        # 生成单个图表
        print("\n📊 生成吞吐量分析图...")
        visualizer.plot_throughput_analysis()
        
        print("\n⏱️ 生成延迟分析图...")
        visualizer.plot_latency_analysis()
        
        print("\n🚀 生成交互式仪表板...")
        visualizer.plot_interactive_dashboard()
        
        # 生成性能报告
        print("\n📋 生成性能报告...")
        report = visualizer.generate_performance_report()
        print(report)

def demo_custom_analysis():
    """自定义分析演示"""
    print("\n" + "=" * 60)
    print("🔬 自定义分析演示")
    print("=" * 60)
    
    visualizer = BenchmarkVisualizer()
    
    if visualizer.df is not None:
        df = visualizer.df
        
        # 自定义分析1：找出最佳性能配置
        print("\n🏆 最佳性能配置分析:")
        best_configs = df.nlargest(5, 'output_throughput')[
            ['config', 'output_throughput', 'mean_ttft_ms', 'max_concurrency']
        ]
        print(best_configs.to_string(index=False))
        
        # 自定义分析2：并发性能效率分析
        print("\n⚡ 并发效率分析:")
        efficiency_analysis = df.groupby('max_concurrency').agg({
            'output_throughput': ['mean', 'std'],
            'efficiency': 'mean',
            'mean_ttft_ms': 'mean'
        }).round(2)
        print(efficiency_analysis)
        
        # 自定义分析3：输入输出长度影响分析
        print("\n📏 输入输出长度影响分析:")
        length_analysis = df.groupby(['input_len', 'output_len']).agg({
            'output_throughput': 'mean',
            'mean_ttft_ms': 'mean',
            'count': 'size'
        }).round(2)
        print(length_analysis)

def demo_advanced_features():
    """高级功能演示"""
    print("\n" + "=" * 60)
    print("🚀 高级功能演示")
    print("=" * 60)
    
    visualizer = BenchmarkVisualizer()
    
    if visualizer.df is not None:
        # 一键生成所有图表
        print("📊 一键生成所有可视化图表...")
        visualizer.generate_all_charts()
        
        print("\n✅ 高级功能演示完成！")
        print(f"📁 所有文件已保存到: {visualizer.figures_dir}/")

def performance_insights():
    """性能洞察分析"""
    print("\n" + "=" * 60)
    print("💡 性能洞察分析")
    print("=" * 60)
    
    visualizer = BenchmarkVisualizer()
    
    if visualizer.df is not None:
        df = visualizer.df
        
        # 计算关键洞察
        print("📈 关键性能洞察:")
        
        # 1. 吞吐量与并发数的关系
        throughput_corr = df['output_throughput'].corr(df['max_concurrency'])
        print(f"1. 吞吐量与并发数相关性: {throughput_corr:.3f}")
        
        if throughput_corr > 0.7:
            print("   ✅ 强正相关 - 增加并发数可有效提升吞吐量")
        elif throughput_corr > 0.3:
            print("   ⚠️  中等相关 - 增加并发数有一定效果但可能存在瓶颈")
        else:
            print("   ❌ 弱相关 - 继续增加并发数可能无效或有害")
        
        # 2. 延迟稳定性分析
        ttft_cv = df['mean_ttft_ms'].std() / df['mean_ttft_ms'].mean()
        print(f"\n2. TTFT变异系数: {ttft_cv:.3f}")
        
        if ttft_cv < 0.2:
            print("   ✅ 延迟非常稳定")
        elif ttft_cv < 0.5:
            print("   ⚠️  延迟基本稳定")
        else:
            print("   ❌ 延迟波动较大，需要优化")
        
        # 3. 最优配置推荐
        print(f"\n3. 配置推荐:")
        
        # 找到效率最高的配置
        best_efficiency_idx = df['efficiency'].idxmax()
        best_config = df.loc[best_efficiency_idx]
        
        print(f"   推荐配置: {best_config['config']}")
        print(f"   - 并发数: {best_config['max_concurrency']}")
        print(f"   - 输入长度: {best_config['input_len']}")
        print(f"   - 输出长度: {best_config['output_len']}")
        print(f"   - 效率: {best_config['efficiency']:.2f} tokens/s per concurrency")
        print(f"   - 吞吐量: {best_config['output_throughput']:.2f} tokens/s")
        print(f"   - TTFT: {best_config['mean_ttft_ms']:.2f} ms")

def main():
    """主函数"""
    print("🎯 vLLM基准测试可视化工具完整演示")
    print("=" * 80)
    
    try:
        # 基础使用演示
        demo_basic_usage()
        
        # 自定义分析
        demo_custom_analysis()
        
        # 性能洞察
        performance_insights()
        
        # 高级功能（放在最后，因为会生成所有图表）
        demo_advanced_features()
        
        print("\n" + "=" * 80)
        print("🎉 所有演示完成！")
        print("💡 提示：")
        print("   - 所有图表文件保存在 'figures/' 目录")
        print("   - 可以单独调用各个方法进行特定分析")
        print("   - 支持保存为PNG和HTML格式")
        print("   - 查看 'figures/performance_report.txt' 获取详细报告")
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        print("💡 请确保已安装所需依赖: pip install -r requirements.txt")
        print("💡 请确保存在聚合结果文件: python3 aggregate_result.py")

if __name__ == "__main__":
    main() 