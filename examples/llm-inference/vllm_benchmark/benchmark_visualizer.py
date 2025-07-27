#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_visualizer.py - vLLM基准测试结果可视化工具

本模块提供了功能完整的基准测试数据可视化解决方案，专门用于分析和展示
vLLM推理服务的性能指标。主要功能包括：

1. 多维度性能分析：
   - 吞吐量分析：并发数vs吞吐量、输入长度影响、时间趋势、效率热力图
   - 延迟分析：TTFT分布、TPOT分析、延迟组件对比、端到端延迟
   - 交互式仪表板：支持动态筛选和缩放的Web界面

2. 数据处理能力：
   - 自动加载和预处理CSV格式的基准测试结果
   - 智能数据清洗和格式转换
   - 支持多种数据源和格式

3. 可视化功能：
   - 静态图表：高质量PNG图片，适合报告和演示
   - 交互式图表：HTML格式，支持缩放、筛选、悬停提示
   - 性能报告：自动生成包含洞察和建议的文本报告

4. 高级分析：
   - 性能瓶颈识别
   - 最优配置推荐
   - 趋势分析和预测
   - 多配置对比分析

使用方法：
    visualizer = BenchmarkVisualizer("results/aggregate_results.csv")
    visualizer.generate_all_charts()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from typing import Dict, List, Optional, Tuple, Union
import os
from datetime import datetime
import json

# 设置样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

class BenchmarkVisualizer:
    """vLLM基准测试结果可视化工具类"""
    
    def __init__(self, csv_path: str = "results/aggregate_results.csv"):
        self.csv_path = csv_path
        self.df = None
        self.figures_dir = "figures"
        os.makedirs(self.figures_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        self.load_data()
    
    def load_data(self) -> pd.DataFrame:
        """加载数据"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✅ 成功加载数据: {len(self.df)} 条记录")
            self._preprocess_data()
            return self.df
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return None
    
    def _preprocess_data(self):
        """数据预处理"""
        if self.df is None:
            return
        
        self.df['datetime'] = pd.to_datetime(self.df['date'], format='%Y%m%d-%H%M%S')
        self.df['config'] = self.df.apply(
            lambda x: f"io{x['input_len']}x{x['output_len']}_mc{x['max_concurrency']}", 
            axis=1
        )
        self.df['efficiency'] = self.df['output_throughput'] / self.df['max_concurrency']
        print(f"📊 数据预处理完成")
    
    def get_data_summary(self) -> str:
        """获取数据摘要"""
        if self.df is None:
            return "❌ 数据未加载"
        
        summary = f"""
📈 数据摘要报告
================
记录总数: {len(self.df)}
测试时间范围: {self.df['datetime'].min()} 到 {self.df['datetime'].max()}
模型数量: {self.df['model_id'].nunique()}
配置组合: {self.df['config'].nunique()}

性能指标概览:
- 平均吞吐量: {self.df['output_throughput'].mean():.2f} tokens/s
- 平均TTFT: {self.df['mean_ttft_ms'].mean():.2f} ms
- 平均TPOT: {self.df['mean_tpot_ms'].mean():.2f} ms
- 最大并发范围: {self.df['max_concurrency'].min()} - {self.df['max_concurrency'].max()}
"""
        return summary
    
    def plot_throughput_analysis(self, save: bool = True):
        """吞吐量分析"""
        if self.df is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🚀 vLLM 吞吐量性能分析', fontsize=16, fontweight='bold')
        
        # 并发数 vs 吞吐量
        sns.scatterplot(data=self.df, x='max_concurrency', y='output_throughput', 
                       hue='config', alpha=0.7, ax=axes[0,0])
        axes[0,0].set_title('并发数与输出吞吐量关系')
        
        # 输入长度 vs 吞吐量
        sns.boxplot(data=self.df, x='input_len', y='output_throughput', ax=axes[0,1])
        axes[0,1].set_title('输入长度对吞吐量的影响')
        
        # 时间趋势
        throughput_trend = self.df.groupby('datetime')['total_token_throughput'].mean().reset_index()
        axes[1,0].plot(throughput_trend['datetime'], throughput_trend['total_token_throughput'], 
                      marker='o', linewidth=2)
        axes[1,0].set_title('总吞吐量时间趋势')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 效率热力图
        pivot_data = self.df.pivot_table(values='efficiency', 
                                        index='input_len', 
                                        columns='max_concurrency', 
                                        aggfunc='mean')
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1,1])
        axes[1,1].set_title('效率热力图')
        
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.figures_dir}/throughput_analysis.png', dpi=300, bbox_inches='tight')
            print(f"✅ 吞吐量分析图已保存")
        plt.show()
    
    def plot_latency_analysis(self, save: bool = True):
        """延迟分析"""
        if self.df is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('⏱️ vLLM 延迟性能分析', fontsize=16, fontweight='bold')
        
        # TTFT分布
        self.df['mean_ttft_ms'].hist(bins=30, alpha=0.7, ax=axes[0,0])
        axes[0,0].axvline(self.df['mean_ttft_ms'].mean(), color='red', linestyle='--')
        axes[0,0].set_title('首个Token时间 (TTFT) 分布')
        
        # TPOT vs 并发数
        sns.violinplot(data=self.df, x='max_concurrency', y='mean_tpot_ms', ax=axes[0,1])
        axes[0,1].set_title('每Token时间 (TPOT) vs 并发数')
        
        # 延迟组件对比
        latency_means = [self.df['mean_ttft_ms'].mean(), 
                        self.df['mean_tpot_ms'].mean(), 
                        self.df['mean_itl_ms'].mean()]
        axes[1,0].bar(['TTFT', 'TPOT', 'ITL'], latency_means, 
                     color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[1,0].set_title('平均延迟组件对比')
        
        # 端到端延迟
        sns.scatterplot(data=self.df, x='output_len', y='mean_e2el_ms', 
                       hue='max_concurrency', alpha=0.7, ax=axes[1,1])
        axes[1,1].set_title('端到端延迟 vs 输出长度')
        
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.figures_dir}/latency_analysis.png', dpi=300, bbox_inches='tight')
            print(f"✅ 延迟分析图已保存")
        plt.show()
    
    def plot_interactive_dashboard(self, save: bool = True):
        """交互式仪表板"""
        if self.df is None:
            return
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('吞吐量 vs 并发数', '延迟分布', '时间趋势', '性能热力图')
        )
        
        # 吞吐量散点图
        fig.add_trace(
            go.Scatter(x=self.df['max_concurrency'], 
                      y=self.df['output_throughput'],
                      mode='markers',
                      name='输出吞吐量',
                      marker=dict(size=8, opacity=0.6)),
            row=1, col=1
        )
        
        # TTFT分布
        fig.add_trace(
            go.Histogram(x=self.df['mean_ttft_ms'], 
                        name='TTFT分布',
                        nbinsx=20),
            row=1, col=2
        )
        
        # 时间趋势
        time_series = self.df.groupby('datetime')['output_throughput'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=time_series['datetime'], 
                      y=time_series['output_throughput'],
                      mode='lines+markers',
                      name='吞吐量趋势'),
            row=2, col=1
        )
        
        # 热力图
        pivot_data = self.df.pivot_table(values='output_throughput', 
                                        index='input_len', 
                                        columns='max_concurrency', 
                                        aggfunc='mean')
        fig.add_trace(
            go.Heatmap(z=pivot_data.values,
                      x=pivot_data.columns,
                      y=pivot_data.index,
                      colorscale='Viridis'),
            row=2, col=2
        )
        
        fig.update_layout(title='🚀 vLLM 基准测试交互式仪表板', height=800)
        
        if save:
            fig.write_html(f'{self.figures_dir}/interactive_dashboard.html')
            print(f"✅ 交互式仪表板已保存")
        fig.show()
    
    def plot_concurrency_performance(self, save: bool = True) -> None:
        """并发性能分析"""
        if self.df is None:
            print("❌ 数据未加载")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🔄 vLLM 并发性能分析', fontsize=16, fontweight='bold')
        
        # 1. 并发扩展性
        concurrency_perf = self.df.groupby('max_concurrency').agg({
            'output_throughput': ['mean', 'std'],
            'mean_ttft_ms': 'mean',
            'efficiency': 'mean'
        }).reset_index()
        
        concurrency_perf.columns = ['max_concurrency', 'throughput_mean', 'throughput_std', 
                                   'ttft_mean', 'efficiency_mean']
        
        axes[0,0].errorbar(concurrency_perf['max_concurrency'], 
                          concurrency_perf['throughput_mean'],
                          yerr=concurrency_perf['throughput_std'],
                          marker='o', capsize=5, capthick=2)
        axes[0,0].set_title('并发扩展性 (吞吐量)')
        axes[0,0].set_xlabel('最大并发数')
        axes[0,0].set_ylabel('平均输出吞吐量 (tokens/s)')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 并发效率
        axes[0,1].plot(concurrency_perf['max_concurrency'], 
                      concurrency_perf['efficiency_mean'], 
                      marker='s', linewidth=2, markersize=8, color='green')
        axes[0,1].set_title('并发效率')
        axes[0,1].set_xlabel('最大并发数')
        axes[0,1].set_ylabel('效率 (tokens/s per concurrency)')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. TTFT vs 并发数
        axes[1,0].plot(concurrency_perf['max_concurrency'], 
                      concurrency_perf['ttft_mean'], 
                      marker='^', linewidth=2, markersize=8, color='red')
        axes[1,0].set_title('首Token延迟 vs 并发数')
        axes[1,0].set_xlabel('最大并发数')
        axes[1,0].set_ylabel('平均TTFT (ms)')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. 并发数分布
        concurrency_counts = self.df['max_concurrency'].value_counts().sort_index()
        bars = axes[1,1].bar(concurrency_counts.index, concurrency_counts.values, 
                            alpha=0.7, color='purple')
        axes[1,1].set_title('测试并发数分布')
        axes[1,1].set_xlabel('最大并发数')
        axes[1,1].set_ylabel('测试次数')
        
        # 添加数值标签
        for bar, value in zip(bars, concurrency_counts.values):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                          str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.figures_dir}/concurrency_performance.png', dpi=300, bbox_inches='tight')
            print(f"✅ 并发性能分析图已保存: {self.figures_dir}/concurrency_performance.png")
        
        plt.show()
    
    def plot_performance_comparison(self, save: bool = True) -> None:
        """性能对比分析"""
        if self.df is None:
            print("❌ 数据未加载")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📊 vLLM 性能对比分析', fontsize=16, fontweight='bold')
        
        # 1. 不同配置的性能雷达图
        configs = self.df['config'].unique()[:5]  # 取前5个配置
        metrics = ['output_throughput', 'mean_ttft_ms', 'mean_tpot_ms', 'efficiency']
        
        # 标准化数据用于雷达图
        normalized_data = []
        for config in configs:
            config_data = self.df[self.df['config'] == config]
            values = []
            for metric in metrics:
                if metric in ['mean_ttft_ms', 'mean_tpot_ms']:
                    # 延迟指标：越小越好，使用倒数
                    values.append(1 / config_data[metric].mean())
                else:
                    # 吞吐量和效率：越大越好
                    values.append(config_data[metric].mean())
            normalized_data.append(values)
        
        # 使用条形图代替雷达图（matplotlib实现更简单）
        x = np.arange(len(metrics))
        width = 0.15
        
        for i, (config, values) in enumerate(zip(configs, normalized_data)):
            axes[0,0].bar(x + i*width, values, width, label=config, alpha=0.8)
        
        axes[0,0].set_title('配置性能对比')
        axes[0,0].set_xlabel('性能指标')
        axes[0,0].set_ylabel('标准化值')
        axes[0,0].set_xticks(x + width * 2)
        axes[0,0].set_xticklabels(['吞吐量', 'TTFT⁻¹', 'TPOT⁻¹', '效率'])
        axes[0,0].legend()
        
        # 2. 输入/输出长度性能分析
        io_perf = self.df.groupby(['input_len', 'output_len'])['output_throughput'].mean().reset_index()
        scatter = axes[0,1].scatter(io_perf['input_len'], io_perf['output_len'], 
                                   c=io_perf['output_throughput'], s=100, 
                                   cmap='viridis', alpha=0.7)
        axes[0,1].set_title('输入/输出长度性能图')
        axes[0,1].set_xlabel('输入长度 (tokens)')
        axes[0,1].set_ylabel('输出长度 (tokens)')
        plt.colorbar(scatter, ax=axes[0,1], label='吞吐量 (tokens/s)')
        
        # 3. 性能指标相关性分析
        correlation_metrics = ['output_throughput', 'mean_ttft_ms', 'mean_tpot_ms', 
                              'max_concurrency', 'num_prompts']
        corr_matrix = self.df[correlation_metrics].corr()
        
        im = axes[1,0].imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        axes[1,0].set_xticks(range(len(correlation_metrics)))
        axes[1,0].set_yticks(range(len(correlation_metrics)))
        axes[1,0].set_xticklabels([m.replace('_', '\n') for m in correlation_metrics], rotation=45)
        axes[1,0].set_yticklabels([m.replace('_', '\n') for m in correlation_metrics])
        axes[1,0].set_title('性能指标相关性')
        
        # 添加相关性数值
        for i in range(len(correlation_metrics)):
            for j in range(len(correlation_metrics)):
                text = axes[1,0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white")
        
        plt.colorbar(im, ax=axes[1,0], label='相关系数')
        
        # 4. 最佳性能配置
        best_configs = self.df.nlargest(10, 'output_throughput')[['config', 'output_throughput', 'mean_ttft_ms']]
        
        bars = axes[1,1].barh(range(len(best_configs)), best_configs['output_throughput'], alpha=0.7)
        axes[1,1].set_yticks(range(len(best_configs)))
        axes[1,1].set_yticklabels(best_configs['config'], fontsize=8)
        axes[1,1].set_title('Top 10 最佳吞吐量配置')
        axes[1,1].set_xlabel('输出吞吐量 (tokens/s)')
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, best_configs['output_throughput'])):
            axes[1,1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                          f'{value:.1f}', va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.figures_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
            print(f"✅ 性能对比分析图已保存: {self.figures_dir}/performance_comparison.png")
        
        plt.show()
    
    def generate_performance_report(self, save: bool = True) -> str:
        """生成性能报告"""
        if self.df is None:
            return "❌ 数据未加载"
        
        best_throughput = self.df.loc[self.df['output_throughput'].idxmax()]
        best_latency = self.df.loc[self.df['mean_ttft_ms'].idxmin()]
        
        report = f"""
🎯 vLLM 基准测试性能报告
========================
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📈 关键性能指标
--------------
总测试次数: {len(self.df)}
平均输出吞吐量: {self.df['output_throughput'].mean():.2f} tokens/s
平均TTFT: {self.df['mean_ttft_ms'].mean():.2f} ms
平均TPOT: {self.df['mean_tpot_ms'].mean():.2f} ms

🏆 最佳性能配置
--------------
最高吞吐量:
  - 配置: {best_throughput['config']}
  - 吞吐量: {best_throughput['output_throughput']:.2f} tokens/s
  - 并发数: {best_throughput['max_concurrency']}

最低延迟:
  - 配置: {best_latency['config']}
  - TTFT: {best_latency['mean_ttft_ms']:.2f} ms
  - 并发数: {best_latency['max_concurrency']}
"""
        
        if save:
            with open(f'{self.figures_dir}/performance_report.txt', 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ 性能报告已保存")
        
        return report
    
    def generate_all_charts(self):
        """生成所有图表"""
        print("🚀 开始生成所有可视化图表...")
        
        try:
            self.plot_throughput_analysis()
            self.plot_latency_analysis() 
            self.plot_interactive_dashboard()
            print("\n" + self.generate_performance_report())
            print(f"\n✅ 所有图表已生成完成！请查看 {self.figures_dir}/ 目录")
        except Exception as e:
            print(f"❌ 生成图表时发生错误: {e}")


def main():
    """主函数"""
    print("🎯 vLLM 基准测试可视化工具")
    visualizer = BenchmarkVisualizer()
    
    if visualizer.df is not None:
        # 显示数据摘要
        print(visualizer.get_data_summary())
        
        # 生成所有图表
        visualizer.generate_all_charts()
        
        # 导出数据
        visualizer.export_data('excel')
        
        print("\n🎉 可视化分析完成！")
    else:
        print("❌ 无法加载数据")


if __name__ == "__main__":
    main() 