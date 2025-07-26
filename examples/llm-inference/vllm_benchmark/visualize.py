#!/usr/bin/env python3
"""
vLLM基准测试可视化工具 - 快速启动脚本

使用方法:
    python3 visualize.py                    # 生成所有图表
    python3 visualize.py --throughput       # 只生成吞吐量分析
    python3 visualize.py --latency          # 只生成延迟分析
    python3 visualize.py --interactive      # 只生成交互式仪表板
    python3 visualize.py --report           # 只生成性能报告
    python3 visualize.py --demo             # 运行完整演示
"""

import argparse
import sys
from benchmark_visualizer import BenchmarkVisualizer

def main():
    parser = argparse.ArgumentParser(
        description='vLLM基准测试可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    python3 visualize.py                    # 生成所有图表
    python3 visualize.py --throughput       # 只生成吞吐量分析
    python3 visualize.py --latency          # 只生成延迟分析
    python3 visualize.py --interactive      # 只生成交互式仪表板
    python3 visualize.py --report           # 只生成性能报告
    python3 visualize.py --demo             # 运行完整演示
    python3 visualize.py --csv custom.csv   # 指定自定义CSV文件
        """
    )
    
    parser.add_argument('--csv', '-c', 
                       default='results/aggregate_results.csv',
                       help='指定CSV文件路径 (默认: results/aggregate_results.csv)')
    
    parser.add_argument('--throughput', '-t', 
                       action='store_true',
                       help='生成吞吐量分析图')
    
    parser.add_argument('--latency', '-l', 
                       action='store_true',
                       help='生成延迟分析图')
    
    parser.add_argument('--interactive', '-i', 
                       action='store_true',
                       help='生成交互式仪表板')
    
    parser.add_argument('--report', '-r', 
                       action='store_true',
                       help='生成性能报告')
    
    parser.add_argument('--demo', '-d', 
                       action='store_true',
                       help='运行完整演示')
    
    parser.add_argument('--no-save', '-n', 
                       action='store_true',
                       help='不保存图表文件（仅显示）')
    
    args = parser.parse_args()
    
    print("🎯 vLLM基准测试可视化工具")
    print("=" * 50)
    
    # 创建可视化器实例
    try:
        visualizer = BenchmarkVisualizer(csv_path=args.csv)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return 1
    
    if visualizer.df is None:
        print("❌ 无法加载数据，请检查文件路径")
        return 1
    
    save = not args.no_save
    
    # 如果运行演示
    if args.demo:
        print("🚀 运行完整演示...")
        try:
            from example_usage import main as demo_main
            demo_main()
        except ImportError:
            print("❌ 找不到演示脚本 example_usage.py")
        return 0
    
    # 检查是否指定了特定功能
    any_specific = args.throughput or args.latency or args.interactive or args.report
    
    if not any_specific:
        # 默认生成所有图表
        print("📊 生成所有可视化图表...")
        visualizer.generate_all_charts()
    else:
        # 按需生成
        if args.throughput:
            print("📈 生成吞吐量分析图...")
            visualizer.plot_throughput_analysis(save=save)
        
        if args.latency:
            print("⏱️ 生成延迟分析图...")
            visualizer.plot_latency_analysis(save=save)
        
        if args.interactive:
            print("🚀 生成交互式仪表板...")
            visualizer.plot_interactive_dashboard(save=save)
        
        if args.report:
            print("📋 生成性能报告...")
            report = visualizer.generate_performance_report(save=save)
            print(report)
    
    # 显示数据摘要
    print("\n" + "=" * 50)
    print("📊 数据摘要")
    print("=" * 50)
    try:
        print(visualizer.get_data_summary())
    except:
        pass
    
    if save:
        print(f"\n✅ 图表已保存到: {visualizer.figures_dir}/")
        print("💡 提示:")
        print("   - PNG图片可用于报告和演示")
        print("   - HTML文件支持交互式浏览")
        print("   - 性能报告包含详细分析建议")
    
    print("\n🎉 可视化完成！")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 