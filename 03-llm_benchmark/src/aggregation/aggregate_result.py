# 文件: aggregate_result.py
# 功能: 聚合vLLM推理压测结果，将多个JSON结果文件合并为一个CSV文件
# 依赖: pip install pandas

import glob
import json
import pandas as pd
import os
import re
from datetime import datetime

# 结果文件存储目录
RESULT_DIR = "results"
# 生成带日期的输出CSV文件路径
current_date = datetime.now().strftime("%Y%m%d")
OUT_CSV = os.path.join(RESULT_DIR, f"aggregate_results_{current_date}.csv")

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

def main():
    """
    主函数：聚合所有压测结果
    
    处理流程:
    1. 扫描results目录下的所有JSON文件
    2. 读取每个JSON文件的内容
    3. 从文件名解析输入输出长度信息
    4. 将所有数据合并到pandas DataFrame
    5. 导出为双语CSV文件（英文+中文列名）
    """
    # 1) 获取results目录下所有JSON文件的路径列表
    json_paths = glob.glob(os.path.join(RESULT_DIR, "*.json"))
    if not json_paths:
        print("在results/目录中未找到JSON文件。")
        return

    # 2) 遍历每个JSON文件，读取数据并添加元信息
    records = []
    for p in json_paths:
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

        records.append(data)

    # 3) 使用pandas将所有记录转换为DataFrame
    df = pd.json_normalize(records)  # 将嵌套的JSON数据扁平化
    
    # 4) 保存为双语CSV文件
    save_bilingual_csv(df, OUT_CSV)
    
    print(f"已聚合 {len(records)} 个测试结果 → {OUT_CSV}")
    print("CSV文件格式：第一行为英文列名，第二行为中文列名，第三行开始为数据")

if __name__ == "__main__":
    main()
