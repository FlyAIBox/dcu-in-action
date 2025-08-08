#!/usr/bin/env python3
"""
简化版可视化脚本，解决中文字体显示问题
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import argparse
import os
from pathlib import Path

# 设置中文字体支持
def setup_chinese_font():
    """设置中文字体支持"""
    import os
    import shutil
    
    # 清理matplotlib字体缓存
    try:
        cache_dir = os.path.join(os.path.expanduser('~'), '.matplotlib')
        if os.path.exists(cache_dir):
            print("Clearing matplotlib font cache...")
            shutil.rmtree(cache_dir, ignore_errors=True)
    except Exception as e:
        print(f"Failed to clear cache (ignored): {e}")
    
    # 重新加载字体管理器
    fm.fontManager.__init__()
    
    # 直接指定字体文件路径，优先使用字符集更完整的中文字体
    font_files = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # Noto CJK - 优先
        '/usr/share/fonts/truetype/arphic/uming.ttc',      # AR PL UMing CN
        '/usr/share/fonts/truetype/arphic/ukai.ttc',       # AR PL UKai CN
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # WenQuanYi Micro Hei
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',    # WenQuanYi Zen Hei
    ]
    
    # 尝试添加字体文件
    selected_font = None
    available_chinese_fonts = []
    
    for font_file in font_files:
        if os.path.exists(font_file):
            try:
                # 添加字体到matplotlib
                fm.fontManager.addfont(font_file)
                
                # 获取字体名称
                for font in fm.fontManager.ttflist:
                    if font.fname == font_file:
                        font_name = font.name
                        if any(keyword in font_name for keyword in ['CJK', 'CN', 'UMing', 'UKai', 'WenQuanYi']):
                            available_chinese_fonts.append(font_name)
                            if not selected_font:
                                selected_font = font_name
                                print(f"✓ Successfully added Chinese font: {selected_font} ({font_file})")
            except Exception as e:
                print(f"✗ Failed to add font {font_file}: {e}")
    
    # 备用字体名称列表（系统预装的字体）
    fallback_fonts = [
        'Noto Sans CJK SC',
        'Noto Serif CJK SC', 
        'AR PL UMing CN',
        'AR PL UKai CN',
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'SimHei',
        'Microsoft YaHei',
        'PingFang SC',
        'Hiragino Sans GB'
    ]
    
    # 如果直接添加字体失败，尝试使用系统字体名称
    if not selected_font:
        print("Searching for system pre-installed Chinese fonts...")
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        for font in fallback_fonts:
            if font in available_fonts:
                selected_font = font
                print(f"✓ Using system Chinese font: {selected_font}")
                break
    
    # 强制设置字体参数
    if selected_font:
        # 设置多个备选字体确保覆盖
        font_list = [selected_font] + available_chinese_fonts + ['DejaVu Sans', 'Arial', 'Liberation Sans']
        # 去重
        font_list = list(dict.fromkeys(font_list))
        
        plt.rcParams['font.sans-serif'] = font_list
        plt.rcParams['font.family'] = 'sans-serif'
        
        # 强制设置所有相关的字体参数
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.titlesize'] = 'large'
        
        print(f"✓ Font setup completed, font list: {font_list[:3]}...")

        # 验证字体设置
        current_font = plt.rcParams['font.sans-serif'][0]
        print(f"✓ Current font in use: {current_font}")

    else:
        print("⚠️  Warning: No Chinese font found, Chinese characters may display as boxes")
        print("Recommended: sudo apt-get install fonts-noto-cjk")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    return selected_font

# 初始化字体设置
font_status = setup_chinese_font()

def create_throughput_comparison(df, output_dir):
    """Create throughput comparison charts"""
    print("Generating throughput comparison charts...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Throughput Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. 总吞吐量对比
    bars1 = ax1.bar(range(len(df)), df['total_token_throughput'].astype(float),
                    color='skyblue', alpha=0.8, edgecolor='navy')
    ax1.set_title('Total Token Throughput (tokens/s)', fontweight='bold')
    ax1.set_ylabel('Throughput (tokens/s)')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['config'], rotation=45, ha='right')

    # 添加数值标签
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

    # 2. 生成吞吐量对比
    bars2 = ax2.bar(range(len(df)), df['output_throughput'].astype(float),
                    color='lightgreen', alpha=0.8, edgecolor='darkgreen')
    ax2.set_title('Output Throughput (tokens/s)', fontweight='bold')
    ax2.set_ylabel('Throughput (tokens/s)')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['config'], rotation=45, ha='right')

    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

    # 3. 令牌数量对比
    x = np.arange(len(df))
    width = 0.35

    bars3 = ax3.bar(x - width/2, df['total_input_tokens'].astype(float), width,
                    label='Input Tokens', color='orange', alpha=0.8)
    bars4 = ax3.bar(x + width/2, df['total_output_tokens'].astype(float), width,
                    label='Output Tokens', color='purple', alpha=0.8)

    ax3.set_title('Token Count Comparison', fontweight='bold')
    ax3.set_ylabel('Token Count')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df['config'], rotation=45, ha='right')
    ax3.legend()

    # 4. 请求处理速度
    bars5 = ax4.bar(range(len(df)), df['request_throughput'].astype(float),
                    color='coral', alpha=0.8, edgecolor='red')
    ax4.set_title('Request Processing Rate (req/s)', fontweight='bold')
    ax4.set_ylabel('Requests per Second')
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels(df['config'], rotation=45, ha='right')

    for i, bar in enumerate(bars5):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Throughput comparison chart generated")

def create_latency_comparison(df, output_dir):
    """Create latency comparison charts"""
    print("Generating latency comparison charts...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Latency Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. 平均延迟对比
    bars1 = ax1.bar(range(len(df)), df['mean_ttft_ms'].astype(float),
                    color='lightcoral', alpha=0.8, edgecolor='darkred')
    ax1.set_title('Mean Time to First Token (ms)', fontweight='bold')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['config'], rotation=45, ha='right')

    # 2. P99延迟对比
    bars2 = ax2.bar(range(len(df)), df['p99_ttft_ms'].astype(float),
                    color='gold', alpha=0.8, edgecolor='orange')
    ax2.set_title('P99 Time to First Token (ms)', fontweight='bold')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['config'], rotation=45, ha='right')

    # 3. 生成延迟对比
    bars3 = ax3.bar(range(len(df)), df['mean_tpot_ms'].astype(float),
                    color='lightblue', alpha=0.8, edgecolor='blue')
    ax3.set_title('Mean Time per Output Token (ms)', fontweight='bold')
    ax3.set_ylabel('Latency (ms)')
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels(df['config'], rotation=45, ha='right')

    # 4. 端到端延迟对比
    bars4 = ax4.bar(range(len(df)), df['mean_e2el_ms'].astype(float),
                    color='plum', alpha=0.8, edgecolor='purple')
    ax4.set_title('Mean End-to-End Latency (ms)', fontweight='bold')
    ax4.set_ylabel('Latency (ms)')
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels(df['config'], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Latency comparison chart generated")

def create_performance_heatmap(df, output_dir):
    """Create performance heatmap"""
    print("Generating performance heatmap...")
    
    # 选择关键性能指标
    metrics = [
        'total_token_throughput', 'output_throughput', 'request_throughput',
        'mean_ttft_ms', 'p99_ttft_ms', 'mean_tpot_ms', 'mean_e2el_ms'
    ]
    
    # 创建热力图数据
    heatmap_data = df[['config'] + metrics].set_index('config')
    
    # 标准化数据（0-1范围）
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(
        scaler.fit_transform(heatmap_data),
        index=heatmap_data.index,
        columns=heatmap_data.columns
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(normalized_data.T, annot=True, cmap='RdYlBu_r', 
                cbar_kws={'label': 'Normalized Performance Score'})
    plt.title('Performance Heatmap (Normalized)', fontweight='bold')
    plt.xlabel('Configuration')
    plt.ylabel('Performance Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_heatmap.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Performance heatmap generated")

def main():
    parser = argparse.ArgumentParser(description='Generate performance visualization charts (Simplified)')
    parser.add_argument('--csv', required=True, help='CSV result file path')
    parser.add_argument('--output', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data: {args.csv}")
    df = pd.read_csv(args.csv)

    # Check if first row is Chinese header, skip if so
    if df.iloc[0]['date'] == '日期':
        df = df.iloc[1:].reset_index(drop=True)
        print("Detected Chinese header row, skipped")

    # Create configuration names
    df['config'] = df.apply(lambda row: f"io{row['input_len']}x{row['output_len']}_mc{row['max_concurrency']}_np{row['num_prompts']}", axis=1)

    print(f"Found {len(df)} test configurations")

    print("Generating visualization charts...")

    # Generate various charts
    create_throughput_comparison(df, output_dir)
    create_latency_comparison(df, output_dir)
    create_performance_heatmap(df, output_dir)
    
    print(f"\nAll visualization files saved to: {output_dir}/")
    print("Generated files:")
    print("  - throughput_comparison.png    (Throughput comparison)")
    print("  - latency_comparison.png       (Latency comparison)")
    print("  - performance_heatmap.png      (Performance heatmap)")

if __name__ == "__main__":
    main()
