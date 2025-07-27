# SPDX-License-Identifier: Apache-2.0
"""
可视化模块 - 生成基准测试结果的图表和报告

这个模块提供了完整的基准测试结果可视化功能，包括：
1. 性能指标的图表生成（柱状图、直方图、饼图等）
2. 多种测试结果的对比分析
3. HTML格式的详细报告生成
4. 图表的自动保存和管理

主要功能：
- 在线服务测试结果可视化
- 离线吞吐量测试结果可视化
- 多次测试结果的对比报告
- 自定义图表样式和中文支持
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings

# 忽略matplotlib的警告信息，保持输出清洁
warnings.filterwarnings('ignore')

# 设置中文字体支持，确保图表中的中文能正确显示
# 优先使用SimHei（黑体），如果不可用则使用DejaVu Sans
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class BenchmarkVisualizer:
    """
    基准测试可视化器

    这个类负责将基准测试的数字结果转换为直观的图表和报告。
    支持多种图表类型和输出格式，帮助用户更好地理解测试结果。

    主要特性：
    - 自动化图表生成
    - 多种图表类型支持
    - 中文界面友好
    - 结果自动保存
    - HTML报告生成
    """

    def __init__(self, result_dir: str = "./benchmark_results"):
        """
        初始化可视化器

        Args:
            result_dir: 结果保存目录，所有生成的图表和报告都会保存在这里
        """
        self.result_dir = result_dir
        # 确保结果目录存在
        os.makedirs(result_dir, exist_ok=True)

        # 设置绘图样式，使用seaborn的白色网格样式
        sns.set_style("whitegrid")
        plt.style.use('seaborn-v0_8')
    
    def visualize_serving_results(self, metrics: Dict[str, Any],
                                config: Dict[str, Any],
                                save_path: Optional[str] = None) -> str:
        """
        可视化在线服务测试结果

        生成一个包含6个子图的综合性能分析图表：
        1. 吞吐量指标柱状图 - 显示请求和token吞吐量
        2. TTFT分布直方图 - 首token时间分布
        3. TPOT分布直方图 - 每token时间分布
        4. 延迟百分位数对比 - P50/P90/P95/P99对比
        5. 请求统计饼图 - 成功请求和token统计
        6. 配置信息表格 - 测试配置概览

        Args:
            metrics: 包含性能指标的字典
            config: 包含测试配置的字典
            save_path: 图表保存路径，None则自动生成

        Returns:
            str: 保存的图表文件路径
        """
        # 如果未指定保存路径，则自动生成带时间戳的文件名
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.result_dir, f"serving_results_{timestamp}.png")

        # 创建2x3的子图布局，总图尺寸为18x12英寸
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'vLLM在线服务基准测试结果\n模型: {config.get("model", "Unknown")}',
                    fontsize=16, fontweight='bold')

        # ==================== 子图1: 吞吐量指标柱状图 ====================
        throughput_data = {
            '请求吞吐量\n(req/s)': metrics.get('request_throughput', 0),      # 每秒处理的请求数
            '输出token吞吐量\n(tok/s)': metrics.get('output_throughput', 0),   # 每秒生成的token数
            '总token吞吐量\n(tok/s)': metrics.get('total_token_throughput', 0) # 输入+输出token总吞吐量
        }

        # 使用不同颜色的柱状图显示三种吞吐量指标
        bars = axes[0, 0].bar(throughput_data.keys(), throughput_data.values(),
                             color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].set_title('吞吐量指标', fontweight='bold')
        axes[0, 0].set_ylabel('数值')

        # 在每个柱子顶部添加数值标签，便于精确读取
        for bar, value in zip(bars, throughput_data.values()):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2,
                           bar.get_height() + max(throughput_data.values())*0.01,
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # ==================== 子图2: TTFT延迟分布直方图 ====================
        if 'ttft_ms' in metrics and metrics['ttft_ms']:
            # 绘制TTFT（首token时间）的分布直方图
            axes[0, 1].hist(metrics['ttft_ms'], bins=20, alpha=0.7, color='#FF6B6B', edgecolor='black')

            # 添加平均值和中位数的垂直参考线
            axes[0, 1].axvline(metrics.get('mean_ttft_ms', 0), color='red', linestyle='--',
                              label=f'平均值: {metrics.get("mean_ttft_ms", 0):.2f}ms')
            axes[0, 1].axvline(metrics.get('median_ttft_ms', 0), color='blue', linestyle='--',
                              label=f'中位数: {metrics.get("median_ttft_ms", 0):.2f}ms')

            # 设置图表标题和坐标轴标签
            axes[0, 1].set_title('首Token时间分布 (TTFT)', fontweight='bold')
            axes[0, 1].set_xlabel('时间 (ms)')
            axes[0, 1].set_ylabel('频次')
            axes[0, 1].legend()

        # ==================== 子图3: TPOT延迟分布直方图 ====================
        if 'tpot_ms' in metrics and metrics['tpot_ms']:
            # 绘制TPOT（每token时间）的分布直方图
            axes[0, 2].hist(metrics['tpot_ms'], bins=20, alpha=0.7, color='#4ECDC4', edgecolor='black')

            # 添加平均值和中位数的垂直参考线
            axes[0, 2].axvline(metrics.get('mean_tpot_ms', 0), color='red', linestyle='--',
                              label=f'平均值: {metrics.get("mean_tpot_ms", 0):.2f}ms')
            axes[0, 2].axvline(metrics.get('median_tpot_ms', 0), color='blue', linestyle='--',
                              label=f'中位数: {metrics.get("median_tpot_ms", 0):.2f}ms')

            # 设置图表标题和坐标轴标签
            axes[0, 2].set_title('每Token时间分布 (TPOT)', fontweight='bold')
            axes[0, 2].set_xlabel('时间 (ms)')
            axes[0, 2].set_ylabel('频次')
            axes[0, 2].legend()
        
        # ==================== 子图4: 延迟百分位数对比柱状图 ====================
        # 百分位数标签和对应的数值
        percentiles = ['P50', 'P90', 'P95', 'P99']
        ttft_percentiles = [metrics.get(f'p{p}_ttft_ms', 0) for p in [50, 90, 95, 99]]
        tpot_percentiles = [metrics.get(f'p{p}_tpot_ms', 0) for p in [50, 90, 95, 99]]

        # 设置柱状图的位置和宽度
        x = np.arange(len(percentiles))
        width = 0.35

        # 绘制并排的柱状图，比较TTFT和TPOT的百分位数
        axes[1, 0].bar(x - width/2, ttft_percentiles, width, label='TTFT', color='#FF6B6B', alpha=0.8)
        axes[1, 0].bar(x + width/2, tpot_percentiles, width, label='TPOT', color='#4ECDC4', alpha=0.8)

        # 设置图表标题、坐标轴标签和刻度
        axes[1, 0].set_title('延迟百分位数对比', fontweight='bold')
        axes[1, 0].set_xlabel('百分位数')
        axes[1, 0].set_ylabel('时间 (ms)')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(percentiles)
        axes[1, 0].legend()

        # ==================== 子图5: 请求统计饼图 ====================
        request_stats = {
            '成功请求': metrics.get('completed', 0),           # 成功处理的请求数
            '总输入tokens': metrics.get('total_input', 0),     # 所有请求的输入token总数
            '总输出tokens': metrics.get('total_output', 0)     # 所有请求的输出token总数
        }

        # 使用不同颜色绘制饼图
        colors = ['#96CEB4', '#FFEAA7', '#DDA0DD']
        _, _, _ = axes[1, 1].pie(request_stats.values(), labels=request_stats.keys(),
                                autopct='%1.0f', colors=colors, startangle=90)
        axes[1, 1].set_title('请求统计', fontweight='bold')
        
        # ==================== 子图6: 配置信息表格 ====================
        # 准备要显示的配置信息
        config_info = [
            ['数据集', config.get('dataset_name', 'Unknown')],
            ['请求数量', str(config.get('num_prompts', 'Unknown'))],
            ['请求速率', f"{config.get('request_rate', 'Unknown')} req/s"],
            ['后端', config.get('backend', 'Unknown')],
            ['测试时长', f"{metrics.get('benchmark_duration_s', 0):.2f}s"]
        ]

        # 隐藏坐标轴，创建纯表格显示
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')

        # 创建表格，显示测试的关键配置信息
        table = axes[1, 2].table(cellText=config_info,
                                colLabels=['配置项', '值'],
                                cellLoc='center',
                                loc='center',
                                colWidths=[0.4, 0.6])

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)  # 调整表格大小
        axes[1, 2].set_title('测试配置', fontweight='bold')

        # 调整子图间距并保存图片
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 高分辨率保存
        plt.close()  # 关闭图形以释放内存

        return save_path
    
    def visualize_throughput_results(self, metrics: Dict[str, Any],
                                   config: Dict[str, Any],
                                   save_path: Optional[str] = None) -> str:
        """可视化离线吞吐量测试结果"""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.result_dir, f"throughput_results_{timestamp}.png")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'vLLM离线吞吐量基准测试结果\n模型: {config.get("model", "Unknown")}', 
                    fontsize=16, fontweight='bold')
        
        # 1. 吞吐量指标
        throughput_metrics = {
            '请求吞吐量\n(req/s)': metrics.get('requests_per_second', 0),
            '总token吞吐量\n(tok/s)': metrics.get('total_tokens_per_second', 0),
            '输出token吞吐量\n(tok/s)': metrics.get('output_tokens_per_second', 0)
        }
        
        bars = axes[0, 0].bar(throughput_metrics.keys(), throughput_metrics.values(),
                             color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].set_title('吞吐量指标', fontweight='bold')
        axes[0, 0].set_ylabel('数值')
        
        for bar, value in zip(bars, throughput_metrics.values()):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(throughput_metrics.values())*0.01,
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Token统计
        token_stats = {
            '输入tokens': metrics.get('total_input_tokens', 0),
            '输出tokens': metrics.get('total_output_tokens', 0)
        }
        
        colors = ['#FFEAA7', '#96CEB4']
        wedges, texts, autotexts = axes[0, 1].pie(token_stats.values(), labels=token_stats.keys(),
                                                 autopct='%1.0f', colors=colors, startangle=90)
        axes[0, 1].set_title('Token分布', fontweight='bold')
        
        # 3. 配置对比
        config_data = [
            ['请求数量', config.get('num_prompts', 0)],
            ['输入长度', config.get('input_len', 0)],
            ['输出长度', config.get('output_len', 0)],
            ['并行度', config.get('tensor_parallel_size', 1)]
        ]
        
        axes[1, 0].axis('tight')
        axes[1, 0].axis('off')
        table = axes[1, 0].table(cellText=config_data,
                                colLabels=['配置项', '值'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 0].set_title('测试配置', fontweight='bold')
        
        # 4. 性能总结
        perf_summary = f"""
        测试总结:
        
        • 总耗时: {metrics.get('elapsed_time_s', 0):.2f}秒
        • 平均每请求: {1000 * metrics.get('elapsed_time_s', 0) / max(config.get('num_prompts', 1), 1):.2f}ms
        • 内存使用: {metrics.get('gpu_memory_usage', 'N/A')}
        • 数据集: {config.get('dataset_name', 'Unknown')}
        """
        
        axes[1, 1].text(0.1, 0.5, perf_summary, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        axes[1, 1].set_title('性能总结', fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_comparison_report(self, results_list: List[Dict], 
                               save_path: Optional[str] = None) -> str:
        """创建多次测试结果对比报告"""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.result_dir, f"comparison_report_{timestamp}.png")
        
        if not results_list:
            print("没有结果数据用于对比")
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('基准测试结果对比报告', fontsize=16, fontweight='bold')
        
        # 提取对比数据
        models = [r.get('config', {}).get('model', f'Test_{i}') for i, r in enumerate(results_list)]
        throughputs = [r.get('metrics', {}).get('request_throughput', 0) for r in results_list]
        ttfts = [r.get('metrics', {}).get('mean_ttft_ms', 0) for r in results_list]
        tpots = [r.get('metrics', {}).get('mean_tpot_ms', 0) for r in results_list]
        
        # 1. 吞吐量对比
        bars = axes[0, 0].bar(range(len(models)), throughputs, color='#4ECDC4')
        axes[0, 0].set_title('请求吞吐量对比', fontweight='bold')
        axes[0, 0].set_ylabel('请求/秒')
        axes[0, 0].set_xticks(range(len(models)))
        axes[0, 0].set_xticklabels([m.split('/')[-1] for m in models], rotation=45)
        
        # 2. TTFT对比
        axes[0, 1].bar(range(len(models)), ttfts, color='#FF6B6B')
        axes[0, 1].set_title('首Token时间对比', fontweight='bold')
        axes[0, 1].set_ylabel('毫秒')
        axes[0, 1].set_xticks(range(len(models)))
        axes[0, 1].set_xticklabels([m.split('/')[-1] for m in models], rotation=45)
        
        # 3. TPOT对比
        axes[1, 0].bar(range(len(models)), tpots, color='#45B7D1')
        axes[1, 0].set_title('每Token时间对比', fontweight='bold')
        axes[1, 0].set_ylabel('毫秒')
        axes[1, 0].set_xticks(range(len(models)))
        axes[1, 0].set_xticklabels([m.split('/')[-1] for m in models], rotation=45)
        
        # 4. 综合评分雷达图
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.5, '综合性能评分\n(功能开发中)', 
                        ha='center', va='center', transform=axes[1, 1].transAxes,
                        fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path


def create_html_report(results: Dict[str, Any], image_paths: List[str], 
                      save_path: Optional[str] = None) -> str:
    """创建HTML格式的详细报告"""
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"./benchmark_results/report_{timestamp}.html"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>vLLM基准测试报告</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; color: #333; }}
            .section {{ margin: 20px 0; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
            .metric-card {{ background: #f5f5f5; padding: 15px; border-radius: 8px; }}
            .chart {{ text-align: center; margin: 20px 0; }}
            .chart img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>vLLM基准测试报告</h1>
            <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <div class="section">
            <h2>测试配置</h2>
            <div class="metrics">
                <div class="metric-card">
                    <h3>模型信息</h3>
                    <p>模型: {results.get('config', {}).get('model', 'Unknown')}</p>
                    <p>后端: {results.get('config', {}).get('backend', 'Unknown')}</p>
                </div>
                <div class="metric-card">
                    <h3>数据集信息</h3>
                    <p>数据集: {results.get('config', {}).get('dataset_name', 'Unknown')}</p>
                    <p>请求数: {results.get('config', {}).get('num_prompts', 'Unknown')}</p>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>性能指标</h2>
            <div class="metrics">
                <div class="metric-card">
                    <h3>吞吐量</h3>
                    <p>请求吞吐量: {results.get('metrics', {}).get('request_throughput', 0):.2f} req/s</p>
                    <p>Token吞吐量: {results.get('metrics', {}).get('output_throughput', 0):.2f} tok/s</p>
                </div>
                <div class="metric-card">
                    <h3>延迟</h3>
                    <p>平均TTFT: {results.get('metrics', {}).get('mean_ttft_ms', 0):.2f} ms</p>
                    <p>平均TPOT: {results.get('metrics', {}).get('mean_tpot_ms', 0):.2f} ms</p>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>可视化图表</h2>
            {''.join([f'<div class="chart"><img src="{path}" alt="测试结果图表"></div>' for path in image_paths])}
        </div>
    </body>
    </html>
    """
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return save_path
