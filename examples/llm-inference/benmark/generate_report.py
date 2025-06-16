#!/usr/bin/env python3
"""
生成大模型推理性能测评报告
"""

import os
import json
import argparse
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ReportGenerator:
    def __init__(self, results_dir: str, timestamp: str):
        self.results_dir = results_dir
        self.timestamp = timestamp
        self.results = []
        
    def load_results(self):
        """加载所有测试结果"""
        for filename in os.listdir(self.results_dir):
            if filename.endswith(f"{self.timestamp}.json"):
                filepath = os.path.join(self.results_dir, filename)
                with open(filepath, "r") as f:
                    results = json.load(f)
                    self.results.extend(results)
    
    def load_system_info(self) -> str:
        """加载系统信息"""
        system_info_file = os.path.join(
            self.results_dir, 
            f"system_info_{self.timestamp}.txt"
        )
        with open(system_info_file, "r") as f:
            return f.read()
    
    def generate_summary_table(self) -> str:
        """生成性能总结表格"""
        df = pd.DataFrame(self.results)
        
        # 按模型和框架分组计算平均值
        summary = df.groupby(["model_name", "framework"]).agg({
            "latency_ms": "mean",
            "throughput": "mean",
            "memory_used_gb": "mean",
            "gpu_util_percent": "mean"
        }).round(2)
        
        return summary.to_markdown()
    
    def generate_batch_size_analysis(self) -> str:
        """分析batch size对性能的影响"""
        df = pd.DataFrame(self.results)
        
        # 创建吞吐量vs批次大小图
        plt.figure(figsize=(12, 6))
        for model in df["model_name"].unique():
            model_data = df[df["model_name"] == model]
            plt.plot(
                model_data["batch_size"], 
                model_data["throughput"],
                marker="o",
                label=model
            )
        
        plt.xlabel("Batch Size")
        plt.ylabel("Throughput (tokens/s)")
        plt.title("Throughput vs Batch Size")
        plt.legend()
        plt.grid(True)
        
        # 保存图片
        plot_path = os.path.join(
            self.results_dir,
            f"batch_size_analysis_{self.timestamp}.png"
        )
        plt.savefig(plot_path)
        plt.close()
        
        return f"![Batch Size Analysis]({plot_path})"
    
    def generate_sequence_length_analysis(self) -> str:
        """分析序列长度对性能的影响"""
        df = pd.DataFrame(self.results)
        
        # 创建延迟vs序列长度图
        plt.figure(figsize=(12, 6))
        for model in df["model_name"].unique():
            model_data = df[df["model_name"] == model]
            plt.plot(
                model_data["sequence_length"],
                model_data["latency_ms"],
                marker="o",
                label=model
            )
        
        plt.xlabel("Sequence Length")
        plt.ylabel("Latency (ms)")
        plt.title("Latency vs Sequence Length")
        plt.legend()
        plt.grid(True)
        
        # 保存图片
        plot_path = os.path.join(
            self.results_dir,
            f"sequence_length_analysis_{self.timestamp}.png"
        )
        plt.savefig(plot_path)
        plt.close()
        
        return f"![Sequence Length Analysis]({plot_path})"
    
    def generate_memory_analysis(self) -> str:
        """分析显存使用情况"""
        df = pd.DataFrame(self.results)
        
        # 创建显存使用热图
        plt.figure(figsize=(10, 8))
        pivot_table = df.pivot_table(
            values="memory_used_gb",
            index="model_name",
            columns="batch_size",
            aggfunc="mean"
        )
        
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd"
        )
        
        plt.title("Memory Usage (GB) by Model and Batch Size")
        
        # 保存图片
        plot_path = os.path.join(
            self.results_dir,
            f"memory_analysis_{self.timestamp}.png"
        )
        plt.savefig(plot_path)
        plt.close()
        
        return f"![Memory Analysis]({plot_path})"
    
    def generate_optimization_suggestions(self) -> str:
        """生成优化建议"""
        df = pd.DataFrame(self.results)
        
        suggestions = []
        
        # 分析GPU利用率
        low_util_cases = df[df["gpu_util_percent"] < 70]
        if not low_util_cases.empty:
            suggestions.append(
                "### GPU利用率优化\n"
                "- 发现以下情况GPU利用率较低：\n" +
                "\n".join([
                    f"  - {row['model_name']} (利用率: {row['gpu_util_percent']:.1f}%)"
                    for _, row in low_util_cases.iterrows()
                ]) +
                "\n- 建议：\n"
                "  - 增加batch size以提高GPU利用率\n"
                "  - 检查数据加载是否存在瓶颈\n"
                "  - 考虑使用混合精度训练"
            )
        
        # 分析显存使用
        high_mem_cases = df[df["memory_used_gb"] > 50]  # 假设50GB为警戒线
        if not high_mem_cases.empty:
            suggestions.append(
                "### 显存优化\n"
                "- 发现以下情况显存使用接近上限：\n" +
                "\n".join([
                    f"  - {row['model_name']} (使用: {row['memory_used_gb']:.1f}GB)"
                    for _, row in high_mem_cases.iterrows()
                ]) +
                "\n- 建议：\n"
                "  - 使用模型量化减少显存占用\n"
                "  - 优化batch size和序列长度\n"
                "  - 考虑使用梯度检查点技术"
            )
        
        # 分析延迟
        high_latency_cases = df[df["latency_ms"] > 1000]  # 假设1秒为警戒线
        if not high_latency_cases.empty:
            suggestions.append(
                "### 延迟优化\n"
                "- 发现以下情况推理延迟较高：\n" +
                "\n".join([
                    f"  - {row['model_name']} (延迟: {row['latency_ms']:.1f}ms)"
                    for _, row in high_latency_cases.iterrows()
                ]) +
                "\n- 建议：\n"
                "  - 使用更高效的推理框架\n"
                "  - 优化模型架构\n"
                "  - 使用模型蒸馏技术\n"
                "  - 考虑使用更多的DCU卡进行并行推理"
            )
        
        return "\n\n".join(suggestions)
    
    def generate_report(self, output_file: str):
        """生成完整的测试报告"""
        # 加载数据
        self.load_results()
        system_info = self.load_system_info()
        
        # 生成报告内容
        report_content = f"""# 海光DCU K100-AI大模型推理性能测评报告

## 测试环境
```
{system_info}
```

## 性能总结
{self.generate_summary_table()}

## Batch Size分析
{self.generate_batch_size_analysis()}

## 序列长度分析
{self.generate_sequence_length_analysis()}

## 显存使用分析
{self.generate_memory_analysis()}

## 优化建议
{self.generate_optimization_suggestions()}

## 测试方法说明
1. 测试环境
   - 硬件：海光DCU K100-AI加速卡（64GB显存）× 8
   - 操作系统：Linux
   - 推理框架：vLLM、Xinference

2. 测试模型
   - DeepSeek-7B
   - Qwen-7B
   - ChatGLM3-6B

3. 测试参数
   - Batch sizes: 1, 4, 8, 16, 32
   - 序列长度: 128, 256, 512, 1024, 2048
   - 每个配置重复测试5次取平均值
   - 使用相同的测试数据和采样参数

4. 测试指标
   - 推理延迟（ms）
   - 吞吐量（tokens/s）
   - 显存使用（GB）
   - GPU利用率（%）

5. 测试流程
   - 每次测试前清空显存
   - 进行2轮预热
   - 收集5轮有效测试数据
   - 测试间隔30秒避免过热

## 结论
1. 性能表现
   - 单卡最大吞吐量
   - 多卡扩展性
   - 显存效率

2. 框架对比
   - vLLM vs Xinference性能差异
   - 各自优势和局限性

3. 模型对比
   - 不同模型在相同硬件下的表现
   - 性能和资源消耗的平衡

4. 最佳实践建议
   - 推荐的批次大小和序列长度配置
   - 硬件资源分配建议
   - 性能优化方向
"""
        
        # 保存报告
        with open(output_file, "w") as f:
            f.write(report_content)
        
        print(f"✅ 测试报告已生成: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="生成大模型推理性能测评报告")
    parser.add_argument("--results-dir", required=True,
                      help="测试结果目录")
    parser.add_argument("--timestamp", required=True,
                      help="测试时间戳")
    parser.add_argument("--output", required=True,
                      help="输出报告文件路径")
    
    args = parser.parse_args()
    
    generator = ReportGenerator(args.results_dir, args.timestamp)
    generator.generate_report(args.output)

if __name__ == "__main__":
    main()