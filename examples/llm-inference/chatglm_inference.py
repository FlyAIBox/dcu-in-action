#!/usr/bin/env python3
"""
ChatGLM模型在海光DCU上的推理示例
支持ChatGLM2-6B和ChatGLM3-6B模型
"""

import torch
import time
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

class ChatGLMInference:
    def __init__(self, model_name="THUDM/chatglm3-6b", device="auto"):
        """
        初始化ChatGLM推理引擎
        
        Args:
            model_name: 模型名称或路径
            device: 设备类型，"auto"为自动选择
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.chat_history = []
        
        self._load_model()
    
    def _load_model(self):
        """加载模型和分词器"""
        print(f"🔄 正在加载模型: {self.model_name}")
        
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
            
            # 设置为评估模式
            self.model.eval()
            
            print("✅ 模型加载成功！")
            
            # 打印模型信息
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"📊 模型参数量: {total_params/1e9:.2f}B")
            
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"💾 显存占用: {memory_used:.2f} GB")
                
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise e
    
    def generate_response(self, prompt, max_length=2048, temperature=0.7, top_p=0.95):
        """
        生成回复
        
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 采样温度
            top_p: nucleus采样参数
            
        Returns:
            str: 生成的回复
        """
        try:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 记录推理时间
            start_time = time.time()
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # 解码输出
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 计算推理时间和速度
            inference_time = time.time() - start_time
            output_tokens = len(outputs[0]) - len(inputs['input_ids'][0])
            tokens_per_second = output_tokens / inference_time
            
            print(f"⏱️  推理时间: {inference_time:.2f}s")
            print(f"🚀 生成速度: {tokens_per_second:.1f} tokens/s")
            
            return response, {
                'inference_time': inference_time,
                'tokens_per_second': tokens_per_second,
                'output_tokens': output_tokens
            }
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return None, None
    
    def chat(self, user_input, history_length=10):
        """
        对话模式，维护对话历史
        
        Args:
            user_input: 用户输入
            history_length: 保留的历史对话轮数
            
        Returns:
            str: 助手回复
        """
        # 添加用户输入到历史
        self.chat_history.append(f"用户: {user_input}")
        
        # 构建对话上下文（保留最近的对话）
        recent_history = self.chat_history[-history_length*2:]  # 每轮包含用户和助手
        context = "\n".join(recent_history)
        
        # 构建提示
        prompt = f"{context}\n助手: "
        
        # 生成回复
        full_response, stats = self.generate_response(prompt)
        
        if full_response:
            # 提取助手回复部分
            assistant_reply = full_response.split("助手:")[-1].strip()
            
            # 添加助手回复到历史
            self.chat_history.append(f"助手: {assistant_reply}")
            
            return assistant_reply, stats
        
        return None, None
    
    def batch_inference(self, prompts, batch_size=4):
        """
        批量推理
        
        Args:
            prompts: 提示列表
            batch_size: 批次大小
            
        Returns:
            list: 回复列表
        """
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            
            # 批量编码
            inputs = self.tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 批量生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7
                )
            
            # 解码结果
            batch_results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(batch_results)
            
            print(f"✅ 完成批次 {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}")
        
        return results
    
    def benchmark(self, test_prompts=None, iterations=5):
        """
        性能基准测试
        
        Args:
            test_prompts: 测试提示列表
            iterations: 测试迭代次数
        """
        if test_prompts is None:
            test_prompts = [
                "介绍一下人工智能的发展历史",
                "解释什么是深度学习",
                "Python编程语言有什么优势？",
                "描述机器学习的应用场景",
                "什么是海光DCU加速卡？"
            ]
        
        print(f"🏃 开始性能基准测试 ({iterations}轮)...")
        
        total_time = 0
        total_tokens = 0
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n测试 {i+1}/{len(test_prompts)}: {prompt[:30]}...")
            
            iter_times = []
            iter_tokens = []
            
            for j in range(iterations):
                response, stats = self.generate_response(prompt)
                if stats:
                    iter_times.append(stats['inference_time'])
                    iter_tokens.append(stats['output_tokens'])
            
            # 计算平均值
            avg_time = sum(iter_times) / len(iter_times)
            avg_tokens = sum(iter_tokens) / len(iter_tokens)
            avg_speed = avg_tokens / avg_time
            
            total_time += avg_time
            total_tokens += avg_tokens
            
            print(f"   平均推理时间: {avg_time:.2f}s")
            print(f"   平均生成速度: {avg_speed:.1f} tokens/s")
        
        # 总体统计
        overall_speed = total_tokens / total_time
        print(f"\n📊 基准测试结果:")
        print(f"   总推理时间: {total_time:.2f}s")
        print(f"   总生成token: {total_tokens:.0f}")
        print(f"   平均生成速度: {overall_speed:.1f} tokens/s")

def interactive_chat(model_name="THUDM/chatglm3-6b"):
    """交互式对话模式"""
    print("🤖 ChatGLM交互式对话模式")
    print("输入 'exit', 'quit' 或 '退出' 来结束对话\n")
    
    # 初始化推理引擎
    engine = ChatGLMInference(model_name)
    
    while True:
        user_input = input("👤 用户: ").strip()
        
        if user_input.lower() in ['exit', 'quit', '退出', '']:
            print("👋 再见！")
            break
        
        print("🤖 助手: ", end="", flush=True)
        
        # 生成回复
        reply, stats = engine.chat(user_input)
        
        if reply:
            print(reply)
            if stats:
                print(f"   (⏱️ {stats['inference_time']:.1f}s, 🚀 {stats['tokens_per_second']:.1f} tokens/s)")
        else:
            print("抱歉，生成回复时出现错误。")
        
        print()

def main():
    parser = argparse.ArgumentParser(description="ChatGLM模型推理")
    parser.add_argument("--model", default="THUDM/chatglm3-6b", help="模型名称或路径")
    parser.add_argument("--mode", choices=["chat", "benchmark", "test"], default="chat", help="运行模式")
    parser.add_argument("--prompt", type=str, help="单次推理的提示")
    
    args = parser.parse_args()
    
    if args.mode == "chat":
        interactive_chat(args.model)
    elif args.mode == "benchmark":
        engine = ChatGLMInference(args.model)
        engine.benchmark()
    elif args.mode == "test":
        if args.prompt:
            engine = ChatGLMInference(args.model)
            response, stats = engine.generate_response(args.prompt)
            print(f"回复: {response}")
            if stats:
                print(f"统计: {stats}")
        else:
            print("测试模式需要提供 --prompt 参数")

if __name__ == "__main__":
    main() 