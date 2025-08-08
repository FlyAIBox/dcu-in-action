#!/usr/bin/env python3
# 文件: main.py
# 功能: vLLM推理服务压测工具统一入口脚本
# 说明: 集成批量压测、单次压测和结果聚合功能的完整工具

import sys
import os
import argparse
import yaml
import subprocess
import glob
import json
import pandas as pd
import re
from datetime import datetime

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 常量定义
BENCHMARK_SCRIPT = "src/core/benchmark_serving.py"
DEFAULT_CONFIG_FILE = "config.yaml"
RESULT_DIR = "results"


def parse_input_output_lengths(filename):
    """
    从文件名中解析输入和输出token长度

    参数:
        filename (str): 结果文件名，格式如 "bench_io256x256_mc32_np128.json"

    返回:
        tuple: (input_len, output_len) 输入和输出token长度

    示例:
        "bench_io256x256_mc32_np128.json" → input_len=256, output_len=256
    """
    # 使用正则表达式匹配文件名中的io{数字}x{数字}模式
    match = re.search(r"io(\d+)x(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def save_bilingual_csv(df, output_path):
    """
    保存双语CSV文件，第一行英文列名，第二行中文列名

    参数:
        df (DataFrame): 包含数据的DataFrame
        output_path (str): 输出文件路径
    """
    # 定义英文到中文的列名映射
    column_mapping = {
        'date': '日期',
        'backend': '后端',
        'model_id': '模型ID',
        'tokenizer_id': '分词器ID',
        'num_prompts': '提示数量',
        'request_rate': '请求速率',
        'burstiness': '突发性',
        'max_concurrency': '最大并发数',
        'duration': '持续时间',
        'completed': '完成数量',
        'total_input_tokens': '总输入令牌数',
        'total_output_tokens': '总输出令牌数',
        'request_throughput': '请求吞吐量',
        'request_goodput:': '请求有效吞吐量',
        'output_throughput': '输出吞吐量',
        'total_token_throughput': '总令牌吞吐量',
        'mean_ttft_ms': '平均首令牌时间(ms)',
        'median_ttft_ms': '中位首令牌时间(ms)',
        'std_ttft_ms': '首令牌时间标准差(ms)',
        'p99_ttft_ms': '首令牌时间P99(ms)',
        'mean_tpot_ms': '平均每令牌时间(ms)',
        'median_tpot_ms': '中位每令牌时间(ms)',
        'std_tpot_ms': '每令牌时间标准差(ms)',
        'p99_tpot_ms': '每令牌时间P99(ms)',
        'mean_itl_ms': '平均令牌间延迟(ms)',
        'median_itl_ms': '中位令牌间延迟(ms)',
        'std_itl_ms': '令牌间延迟标准差(ms)',
        'p99_itl_ms': '令牌间延迟P99(ms)',
        'mean_e2el_ms': '平均端到端延迟(ms)',
        'median_e2el_ms': '中位端到端延迟(ms)',
        'std_e2el_ms': '端到端延迟标准差(ms)',
        'p99_e2el_ms': '端到端延迟P99(ms)',
        'input_len': '输入长度',
        'output_len': '输出长度',
        'filename': '文件名'
    }

    # 获取中文列名
    chinese_columns = []
    for col in df.columns:
        chinese_columns.append(column_mapping.get(col, col))

    # 创建双语CSV内容
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        # 写入英文列名（第一行）
        f.write(','.join(df.columns) + '\n')
        # 写入中文列名（第二行）
        f.write(','.join(chinese_columns) + '\n')
        # 写入数据（从第三行开始）
        df.to_csv(f, index=False, header=False)


def run_benchmark(common_args, input_len, output_len, concurrency, num_prompts, result_subdir):
    """
    执行单个参数组合的压测

    参数:
        common_args (list): 公共的命令行参数列表
        input_len (int): 输入token长度
        output_len (int): 输出token长度
        concurrency (int): 最大并发请求数
        num_prompts (int): 总请求数量
        result_subdir (str): 结果保存的子目录路径
    """
    # 复制公共参数，避免修改原始列表
    args = common_args.copy()

    # 添加输入和输出token长度参数
    args += ["--random-input-len", str(input_len), "--random-output-len", str(output_len)]

    # 添加并发控制和请求数量参数
    args += ["--max-concurrency", str(concurrency)]
    args += ["--num-prompts", str(num_prompts)]

    # 创建结果保存目录
    os.makedirs(result_subdir, exist_ok=True)

    # 生成结果文件名，格式: bench_io{输入长度}x{输出长度}_mc{并发数}_np{请求数}.json
    outfile = os.path.join(
        result_subdir,
        f"bench_io{input_len}x{output_len}_mc{concurrency}_np{num_prompts}.json"
    )
    args += ["--save-result", "--result-filename", outfile]

    # 打印即将执行的命令
    print(f"正在执行: {' '.join(args)}")

    # 执行压测命令
    ret = subprocess.run(args, capture_output=True, text=True)

    # 检查执行结果
    if ret.returncode != 0:
        print(f"参数组合 io=({input_len},{output_len}), mc={concurrency}, np={num_prompts} 执行失败: {ret.stderr}")
    else:
        print(f"参数组合 io=({input_len},{output_len}), mc={concurrency}, np={num_prompts} 执行完成，结果已保存: {outfile}")


def get_model_short_name(model_path):
    """
    从模型路径中提取简短的模型名称
    
    参数:
        model_path (str): 模型路径或名称
        
    返回:
        str: 简化的模型名称
    """
    # 如果是路径，取最后一部分
    model_name = os.path.basename(model_path)
    
    # 移除常见的前缀和后缀
    model_name = model_name.replace("cognitivecomputations/", "")
    model_name = model_name.replace("-awq", "")
    model_name = model_name.replace("-gptq", "")
    model_name = model_name.replace("-gguf", "")
    
    return model_name


def batch_benchmark(config_file=DEFAULT_CONFIG_FILE):
    """
    批量执行压测实验

    参数:
        config_file (str): 配置文件路径

    处理流程:
    1. 从YAML配置文件加载参数设置
    2. 构建公共的命令行参数
    3. 创建按模型名称和时间组织的结果目录
    4. 遍历所有参数组合执行压测
    """
    # 从YAML配置文件加载设置和参数列表
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    # 提取配置参数
    model = cfg["model"]                              # 模型名称
    base_url = cfg["base_url"]                        # vLLM服务器地址
    tokenizer = cfg["tokenizer"]                      # 分词器路径
    io_pairs = cfg.get("input_output", [])            # 输入输出长度组合列表
    cp_pairs = cfg.get("concurrency_prompts", [])     # 并发数和请求数组合列表

    # 创建按模型名称和测试时间组织的结果目录
    model_short_name = get_model_short_name(model)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_subdir = os.path.join(RESULT_DIR, f"{model_short_name}_{current_time}")
    
    print(f"测试结果将保存到: {result_subdir}")

    # 构建所有压测共用的命令行参数
    common_args = [
        "python3", "-m", "src.core.benchmark_serving",
        "--backend", "vllm",                          # 使用vLLM后端
        "--model", model,                             # 指定模型
        "--base-url", base_url,                       # vLLM服务器URL
        "--tokenizer", tokenizer,                     # 分词器
        "--dataset-name", "random",                   # 使用随机生成的数据集
        "--percentile-metrics", "ttft,tpot,itl,e2el"  # 要统计的性能指标百分位数
    ]

    # 执行所有参数组合的压测（笛卡尔积）
    # 对每个输入输出长度组合，测试所有的并发数和请求数组合
    for input_len, output_len in io_pairs:
        for concurrency, num_prompts in cp_pairs:
            run_benchmark(common_args, input_len, output_len, concurrency, num_prompts, result_subdir)
    
    return result_subdir


def single_benchmark(model, base_url, num_prompts=100, max_concurrency=10,
                    random_input_len=256, random_output_len=256, tokenizer=None):
    """
    执行单次压测

    参数:
        model (str): 模型名称
        base_url (str): vLLM服务器地址
        num_prompts (int): 请求数量
        max_concurrency (int): 最大并发数
        random_input_len (int): 输入长度
        random_output_len (int): 输出长度
        tokenizer (str): 分词器路径（可选）
    """
    # 构建命令行参数
    cmd = [
        'python3', '-m', 'src.core.benchmark_serving',
        '--backend', 'vllm',
        '--model', model,
        '--base-url', base_url,
        '--dataset-name', 'random',
        '--random-input-len', str(random_input_len),
        '--random-output-len', str(random_output_len),
        '--num-prompts', str(num_prompts),
        '--max-concurrency', str(max_concurrency),
        '--percentile-metrics', 'ttft,tpot,itl,e2el'
    ]

    # 如果指定了分词器，添加到命令中
    if tokenizer:
        cmd.extend(['--tokenizer', tokenizer])

    # 创建按模型名称和测试时间组织的结果目录
    model_short_name = get_model_short_name(model)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_subdir = os.path.join(RESULT_DIR, f"{model_short_name}_{current_time}")
    os.makedirs(result_subdir, exist_ok=True)

    # 生成结果文件名
    outfile = os.path.join(result_subdir, f"single_bench_{current_time}.json")
    cmd.extend(['--save-result', '--result-filename', outfile])

    print(f"测试结果将保存到: {result_subdir}")
    print(f"正在执行单次压测: {' '.join(cmd)}")

    # 执行压测命令
    ret = subprocess.run(cmd)

    # 检查执行结果
    if ret.returncode != 0:
        print(f"单次压测执行失败，返回码: {ret.returncode}")
    else:
        print(f"单次压测执行完成，结果已保存: {outfile}")
    
    return result_subdir


def get_available_result_dirs():
    """
    获取results目录下所有可用的结果子目录
    
    返回:
        list: 按时间排序的结果目录列表（最新的在前）
    """
    if not os.path.exists(RESULT_DIR):
        return []
    
    subdirs = []
    for item in os.listdir(RESULT_DIR):
        item_path = os.path.join(RESULT_DIR, item)
        if os.path.isdir(item_path):
            # 检查目录中是否有JSON文件
            json_files = glob.glob(os.path.join(item_path, "*.json"))
            if json_files:
                subdirs.append(item)
    
    # 按目录名排序（包含时间戳，所以可以按字典序排序）
    subdirs.sort(reverse=True)
    return subdirs


def aggregate_results(target_dir=None):
    """
    聚合指定目录下的压测结果

    参数:
        target_dir (str): 要聚合的结果目录，如果为None则使用最新的结果目录

    处理流程:
    1. 确定要聚合的目录
    2. 扫描目录下的所有JSON文件
    3. 读取每个JSON文件的内容
    4. 从文件名解析输入输出长度信息
    5. 将所有数据合并到pandas DataFrame
    6. 导出为双语CSV文件（英文+中文列名）
    """
    # 1) 确定要聚合的目录
    if target_dir is None:
        # 获取可用的结果目录
        available_dirs = get_available_result_dirs()
        if not available_dirs:
            # 检查根目录下是否有JSON文件（兼容旧版本）
            json_paths = glob.glob(os.path.join(RESULT_DIR, "*.json"))
            if json_paths:
                target_path = RESULT_DIR
                print(f"在根目录找到 {len(json_paths)} 个JSON文件，将进行聚合")
            else:
                print("在results/目录中未找到任何JSON文件或结果子目录。")
                return
        else:
            # 使用最新的结果目录
            target_dir = available_dirs[0]
            target_path = os.path.join(RESULT_DIR, target_dir)
            print(f"使用最新的结果目录: {target_dir}")
            print(f"可用的结果目录: {', '.join(available_dirs)}")
    else:
        target_path = os.path.join(RESULT_DIR, target_dir)
        if not os.path.exists(target_path):
            print(f"指定的目录不存在: {target_path}")
            return

    # 2) 获取目标目录下所有JSON文件的路径列表
    json_paths = glob.glob(os.path.join(target_path, "*.json"))
    if not json_paths:
        print(f"在目录 {target_path} 中未找到JSON文件。")
        return

    print(f"找到 {len(json_paths)} 个JSON文件进行聚合")

    # 3) 遍历每个JSON文件，读取数据并添加元信息
    records = []
    for p in json_paths:
        try:
            # 读取JSON文件内容
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 获取文件名并解析输入输出长度
            filename = os.path.basename(p)
            input_len, output_len = parse_input_output_lengths(filename)

            # 向数据中添加额外的元信息
            data["input_len"] = input_len      # 输入token长度
            data["output_len"] = output_len    # 输出token长度
            data["filename"] = filename        # 原始文件名
            data["result_dir"] = target_dir or "root"  # 结果目录名

            records.append(data)
        except Exception as e:
            print(f"读取文件 {p} 时出错: {e}")
            continue

    if not records:
        print("没有成功读取到任何有效的测试结果。")
        return

    # 4) 使用pandas将所有记录转换为DataFrame
    df = pd.json_normalize(records)  # 将嵌套的JSON数据扁平化

    # 5) 生成输出CSV文件路径并保存为双语CSV文件
    current_date = datetime.now().strftime("%Y%m%d")
    if target_dir:
        out_csv = os.path.join(target_path, f"aggregate_results_{current_date}.csv")
    else:
        out_csv = os.path.join(RESULT_DIR, f"aggregate_results_{current_date}.csv")
    
    save_bilingual_csv(df, out_csv)

    print(f"已聚合 {len(records)} 个测试结果 → {out_csv}")
    print("CSV文件格式：第一行为英文列名，第二行为中文列名，第三行开始为数据")
    
    # 自动生成可视化报告
    print("\n正在生成可视化报告...")
    try:
        import subprocess
        result = subprocess.run([
            'python3', 'src/visualize/visualize_simple.py', 
            '--csv', out_csv,
            '--output', 'reports'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ 可视化报告生成成功")
            print("报告文件位置: reports/")
        else:
            print(f"可视化报告生成失败: {result.stderr}")
    except Exception as e:
        print(f"可视化报告生成出错: {e}")
        print("可手动运行: python3 src/visualize/visualize_simple.py")


def visualize_results(csv_path=None, output_dir="charts", mode="advanced"):
    """
    生成可视化报告
    
    参数:
        csv_path (str): CSV文件路径，如果为None则自动查找最新的
        output_dir (str): 输出目录
        mode (str): 模式，"simple"或"advanced"
    """
    
    def find_latest_csv():
        """查找最新的聚合结果CSV文件"""
        csv_files = glob.glob("results/*/aggregate_results_*.csv")
        if not csv_files:
            csv_files = glob.glob("results/aggregate_results_*.csv")
        
        if not csv_files:
            return None
        
        # 按修改时间排序，返回最新的
        csv_files.sort(key=os.path.getmtime, reverse=True)
        return csv_files[0]
    
    # 确定CSV文件路径
    if csv_path is None:
        csv_path = find_latest_csv()
        if not csv_path:
            print("错误: 未找到聚合结果CSV文件")
            print("请先运行 'python main.py aggregate' 生成聚合结果")
            return False
        print(f"使用最新的CSV文件: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"错误: CSV文件不存在: {csv_path}")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在生成可视化报告...")
    print(f"模式: {mode}")
    print(f"输入文件: {csv_path}")
    print(f"输出目录: {output_dir}")
    
    try:
        if mode == "simple":
            # 使用简化版可视化脚本
            result = subprocess.run([
                'python3', 'src/visualize/visualize_simple.py', 
                '--csv', csv_path,
                '--output', output_dir
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ 简化版可视化报告生成成功")
                print("生成的文件:")
                print("  - throughput_comparison.png    (吞吐量对比)")
                print("  - latency_comparison.png       (延迟对比)")
                print("  - performance_heatmap.png      (性能热力图)")
            else:
                print(f"简化版可视化报告生成失败: {result.stderr}")
                return False
                
        elif mode == "advanced":
            # 使用完整版可视化脚本
            result = subprocess.run([
                'python3', 'src/visualize/visualize_results.py',
                '--csv', csv_path,
                '--output', output_dir
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ 完整版可视化报告生成成功")
                print("生成的文件:")
                print("  - throughput_comparison.png    (吞吐量对比)")
                print("  - latency_comparison.png       (延迟对比)")
                print("  - performance_heatmap.png      (性能热力图)")
                print("  - comprehensive_dashboard.png  (综合仪表板)")
                print("  - performance_report.txt       (性能报告)")
            else:
                print(f"完整版可视化报告生成失败: {result.stderr}")
                return False
                
        elif mode == "both":
            # 生成两种模式的报告
            simple_dir = os.path.join(output_dir, "simple")
            advanced_dir = os.path.join(output_dir, "advanced")
            
            print("正在生成简化版报告...")
            visualize_results(csv_path, simple_dir, "simple")
            
            print("正在生成完整版报告...")
            visualize_results(csv_path, advanced_dir, "advanced")
            
        else:
            print(f"错误: 不支持的模式 '{mode}'，支持的模式: simple, advanced, both")
            return False
    
    except Exception as e:
        print(f"可视化报告生成出错: {e}")
        return False
    
    print(f"\n所有可视化文件已保存到: {output_dir}/")
    return True


def main():
    """主函数：解析命令行参数并调用相应的功能"""
    parser = argparse.ArgumentParser(
        description="vLLM推理服务压测工具 - 集成批量压测、单次压测和结果聚合功能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        使用示例:
        # 批量压测（根据config.yaml配置）
        python main.py batch
        python main.py batch --config custom_config.yaml

        # 单次压测
        python main.py single --model deepseek-ai/DeepSeek-R1 --base-url http://localhost:8010 --num-prompts 100
        python main.py single --model /path/to/model --base-url http://localhost:8010 --max-concurrency 16 --random-input-len 512 --random-output-len 512

        # 聚合结果
        python main.py aggregate                                    # 聚合最新的结果目录
        python main.py aggregate --list                            # 列出所有可用的结果目录
        python main.py aggregate --dir DeepSeek-R1_20250728_145302 # 聚合指定的结果目录

        # 生成可视化报告
        python main.py visualize                                    # 自动查找最新CSV文件，生成完整版报告
        python main.py visualize --csv results/aggregate_results_20250728.csv  # 指定CSV文件
        python main.py visualize --mode simple --output simple_charts          # 生成简化版报告
        python main.py visualize --mode both --output all_charts               # 生成两种模式的报告

        功能说明:
        batch     - 根据config.yaml配置文件执行批量压测，结果按模型名称和时间组织到子目录
        single    - 执行单次压测，结果按模型名称和时间组织到子目录
        aggregate - 聚合指定目录下的JSON结果文件，生成双语CSV报告
                   支持 --list 查看可用目录，--dir 指定目录（默认使用最新的）
        visualize - 生成可视化性能报告，支持simple(基础图表)、advanced(完整报告)、both(两种模式)
                   支持自动查找最新CSV文件，或手动指定CSV文件路径
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 批量压测子命令
    batch_parser = subparsers.add_parser('batch', help='批量压测（根据config.yaml配置）')
    batch_parser.add_argument('--config', default=DEFAULT_CONFIG_FILE,
                             help=f'配置文件路径 (默认: {DEFAULT_CONFIG_FILE})')

    # 单次压测子命令
    single_parser = subparsers.add_parser('single', help='单次压测')
    single_parser.add_argument('--model', required=True, help='模型名称或路径')
    single_parser.add_argument('--base-url', required=True, help='vLLM服务器地址 (例如: http://localhost:8010)')
    single_parser.add_argument('--num-prompts', type=int, default=100, help='请求数量 (默认: 100)')
    single_parser.add_argument('--max-concurrency', type=int, default=10, help='最大并发数 (默认: 10)')
    single_parser.add_argument('--random-input-len', type=int, default=256, help='输入token长度 (默认: 256)')
    single_parser.add_argument('--random-output-len', type=int, default=256, help='输出token长度 (默认: 256)')
    single_parser.add_argument('--tokenizer', help='分词器路径 (可选，默认使用模型路径)')

    # 聚合结果子命令
    agg_parser = subparsers.add_parser('aggregate', help='聚合压测结果')
    agg_parser.add_argument('--dir', help='指定要聚合的结果目录名（不指定则使用最新的）')
    agg_parser.add_argument('--list', action='store_true', help='列出所有可用的结果目录')
    
    # 可视化子命令
    viz_parser = subparsers.add_parser('visualize', help='生成可视化报告')
    viz_parser.add_argument('--csv', help='指定CSV文件路径（不指定则自动查找最新的聚合结果）')
    viz_parser.add_argument('--output', default='charts', help='输出目录 (默认: charts)')
    viz_parser.add_argument('--mode', choices=['simple', 'advanced', 'both'], default='advanced',
                           help='可视化模式: simple(基础图表), advanced(完整报告), both(生成两种模式) (默认: advanced)')

    args = parser.parse_args()

    if args.command == 'batch':
        print("=== 开始批量压测 ===")
        print(f"使用配置文件: {args.config}")
        result_dir = batch_benchmark(args.config)
        print("=== 批量压测完成 ===")
        print(f"结果已保存到: {result_dir}")
        print(f"可使用以下命令聚合结果: python main.py aggregate --dir {os.path.basename(result_dir)}")

    elif args.command == 'single':
        print("=== 开始单次压测 ===")
        print(f"模型: {args.model}")
        print(f"服务器地址: {args.base_url}")
        print(f"请求数量: {args.num_prompts}")
        print(f"最大并发数: {args.max_concurrency}")
        print(f"输入长度: {args.random_input_len}")
        print(f"输出长度: {args.random_output_len}")

        result_dir = single_benchmark(
            model=args.model,
            base_url=args.base_url,
            num_prompts=args.num_prompts,
            max_concurrency=args.max_concurrency,
            random_input_len=args.random_input_len,
            random_output_len=args.random_output_len,
            tokenizer=args.tokenizer
        )
        print("=== 单次压测完成 ===")
        print(f"结果已保存到: {result_dir}")
        print(f"可使用以下命令聚合结果: python main.py aggregate --dir {os.path.basename(result_dir)}")

    elif args.command == 'aggregate':
        if args.list:
            print("=== 可用的结果目录 ===")
            available_dirs = get_available_result_dirs()
            if available_dirs:
                for i, dir_name in enumerate(available_dirs, 1):
                    dir_path = os.path.join(RESULT_DIR, dir_name)
                    json_count = len(glob.glob(os.path.join(dir_path, "*.json")))
                    print(f"{i}. {dir_name} ({json_count} 个JSON文件)")
            else:
                print("未找到任何结果目录")
        else:
            print("=== 开始聚合结果 ===")
            if args.dir:
                print(f"指定聚合目录: {args.dir}")
            aggregate_results(args.dir)
            print("=== 结果聚合完成 ===")

    elif args.command == 'visualize':
        print("=== 开始生成可视化报告 ===")
        success = visualize_results(
            csv_path=args.csv,
            output_dir=args.output,
            mode=args.mode
        )
        if success:
            print("=== 可视化报告生成完成 ===")
            print(f"报告已保存到: {args.output}")
        else:
            print("=== 可视化报告生成失败 ===")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()