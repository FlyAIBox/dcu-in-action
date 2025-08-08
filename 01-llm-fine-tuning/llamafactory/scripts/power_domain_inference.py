#!/usr/bin/env python3
"""
电力领域模型推理示例
展示如何使用微调后的Qwen3-32B电力领域模型
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import json
from typing import List, Dict
import time

class PowerDomainAssistant:
    """电力领域助手"""
    
    def __init__(self, 
                 base_model_path: str = "Qwen/Qwen2.5-32B-Instruct",
                 adapter_path: str = "saves/qwen3-32b-power-domain-8dcu",
                 device: str = "dcu"):
        """
        初始化电力领域助手
        
        Args:
            base_model_path: 基础模型路径
            adapter_path: LoRA适配器路径
            device: 设备类型 (dcu/cuda/cpu)
        """
        print(f"加载模型: {base_model_path}")
        print(f"加载适配器: {adapter_path}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载LoRA适配器
        print("加载LoRA适配器...")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        
        # 设置系统提示词
        self.system_prompt = """你是一个电力系统专家，精通电力系统分析、设备运维和故障诊断。
请基于专业知识，准确、详细地回答用户的问题。对于计算题，请给出详细的计算过程和步骤。"""
        
    def generate_response(self, 
                         question: str, 
                         max_length: int = 2048,
                         temperature: float = 0.7,
                         top_p: float = 0.9) -> str:
        """
        生成回答
        
        Args:
            question: 用户问题
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: Top-p采样参数
            
        Returns:
            生成的回答
        """
        # 构建对话
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question}
        ]
        
        # 应用对话模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取助手回答
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        else:
            # 移除输入部分
            response = response[len(text):].strip()
        
        return response
    
    def batch_inference(self, questions: List[str]) -> List[str]:
        """批量推理"""
        responses = []
        for i, question in enumerate(questions, 1):
            print(f"\n处理问题 {i}/{len(questions)}...")
            start_time = time.time()
            response = self.generate_response(question)
            elapsed_time = time.time() - start_time
            responses.append(response)
            print(f"生成完成，耗时: {elapsed_time:.2f}秒")
        return responses


def test_power_domain_model():
    """测试电力领域模型"""
    # 测试问题
    test_questions = [
        "什么是电力系统的N-1准则？请详细说明其重要性。",
        "一条220kV输电线路，输送功率150MW，功率因数0.9，请计算线路电流。",
        "变压器油中乙炔含量超标说明什么问题？应该如何处理？",
        "请解释什么是电力系统的暂态稳定，与静态稳定有何区别？",
        "10kV配电线路长度15km，导线电阻0.3Ω/km，负荷电流300A，计算线损。"
    ]
    
    # 创建助手实例
    assistant = PowerDomainAssistant()
    
    print("="*60)
    print("电力领域模型测试")
    print("="*60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        print("-"*60)
        
        start_time = time.time()
        response = assistant.generate_response(question)
        elapsed_time = time.time() - start_time
        
        print(f"回答:\n{response}")
        print(f"\n生成耗时: {elapsed_time:.2f}秒")
        print("="*60)


def interactive_mode():
    """交互式问答模式"""
    # 创建助手实例
    assistant = PowerDomainAssistant()
    
    print("="*60)
    print("电力领域智能助手 - 交互模式")
    print("输入 'quit' 或 'exit' 退出")
    print("="*60)
    
    while True:
        try:
            # 获取用户输入
            question = input("\n请输入您的问题: ").strip()
            
            # 检查退出命令
            if question.lower() in ['quit', 'exit', 'q']:
                print("感谢使用，再见！")
                break
            
            if not question:
                continue
            
            # 生成回答
            print("\n正在思考...")
            start_time = time.time()
            response = assistant.generate_response(question)
            elapsed_time = time.time() - start_time
            
            # 显示回答
            print(f"\n回答:\n{response}")
            print(f"\n(生成耗时: {elapsed_time:.2f}秒)")
            
        except KeyboardInterrupt:
            print("\n\n程序被中断，再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")


def benchmark_mode(questions_file: str, output_file: str):
    """基准测试模式"""
    # 加载测试问题
    with open(questions_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    questions = [item['instruction'] for item in test_data]
    
    # 创建助手实例
    assistant = PowerDomainAssistant()
    
    print(f"开始基准测试，共 {len(questions)} 个问题...")
    
    # 批量推理
    start_time = time.time()
    responses = assistant.batch_inference(questions)
    total_time = time.time() - start_time
    
    # 保存结果
    results = []
    for i, (item, response) in enumerate(zip(test_data, responses)):
        results.append({
            "id": i,
            "instruction": item['instruction'],
            "expected_output": item.get('output', ''),
            "model_output": response,
            "input": item.get('input', '')
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n基准测试完成!")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均每题: {total_time/len(questions):.2f}秒")
    print(f"结果保存至: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="电力领域模型推理工具")
    parser.add_argument("--mode", choices=['test', 'interactive', 'benchmark'], 
                       default='test', help="运行模式")
    parser.add_argument("--base_model", type=str, 
                       default="Qwen/Qwen2.5-32B-Instruct",
                       help="基础模型路径")
    parser.add_argument("--adapter_path", type=str,
                       default="saves/qwen3-32b-power-domain-8dcu",
                       help="LoRA适配器路径")
    parser.add_argument("--questions_file", type=str,
                       default="data/power_domain_samples.json",
                       help="测试问题文件（benchmark模式）")
    parser.add_argument("--output_file", type=str,
                       default="results/power_domain_benchmark.json",
                       help="结果输出文件（benchmark模式）")
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        test_power_domain_model()
    elif args.mode == 'interactive':
        interactive_mode()
    elif args.mode == 'benchmark':
        benchmark_mode(args.questions_file, args.output_file)


if __name__ == "__main__":
    main() 