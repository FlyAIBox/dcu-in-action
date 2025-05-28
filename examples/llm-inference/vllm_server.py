#!/usr/bin/env python3
"""
基于vLLM的高性能推理服务器
支持多模型并发推理和REST API接口
"""

import asyncio
import time
import json
from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
except ImportError:
    print("❌ 请先安装vLLM: pip install vllm")
    exit(1)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

@dataclass
class ModelConfig:
    """模型配置类"""
    name: str
    model_path: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 2048
    dtype: str = "float16"

class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False

class ChatResponse(BaseModel):
    """聊天响应模型"""
    response: str
    model: str
    usage: Dict[str, int]
    timing: Dict[str, float]

class BatchRequest(BaseModel):
    """批量请求模型"""
    prompts: List[str]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95

class vLLMServer:
    def __init__(self, models_config: List[ModelConfig]):
        """
        初始化vLLM服务器
        
        Args:
            models_config: 模型配置列表
        """
        self.models_config = models_config
        self.engines = {}
        self.app = FastAPI(title="海光DCU vLLM推理服务", version="1.0.0")
        
        self._setup_routes()
    
    async def _load_models(self):
        """异步加载所有模型"""
        print("🔄 正在加载模型...")
        
        for config in self.models_config:
            try:
                print(f"   加载模型: {config.name}")
                
                # 配置引擎参数
                engine_args = AsyncEngineArgs(
                    model=config.model_path,
                    tensor_parallel_size=config.tensor_parallel_size,
                    gpu_memory_utilization=config.gpu_memory_utilization,
                    max_model_len=config.max_model_len,
                    dtype=config.dtype,
                    trust_remote_code=True
                )
                
                # 创建异步引擎
                engine = AsyncLLMEngine.from_engine_args(engine_args)
                self.engines[config.name] = engine
                
                print(f"   ✅ {config.name} 加载成功")
                
            except Exception as e:
                print(f"   ❌ {config.name} 加载失败: {e}")
    
    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.on_event("startup")
        async def startup_event():
            await self._load_models()
        
        @self.app.get("/")
        async def root():
            return {
                "message": "海光DCU vLLM推理服务",
                "models": list(self.engines.keys()),
                "status": "running"
            }
        
        @self.app.get("/models")
        async def list_models():
            """列出所有可用模型"""
            models_info = []
            for name, engine in self.engines.items():
                config = next(c for c in self.models_config if c.name == name)
                models_info.append({
                    "name": name,
                    "model_path": config.model_path,
                    "tensor_parallel_size": config.tensor_parallel_size,
                    "max_model_len": config.max_model_len
                })
            return {"models": models_info}
        
        @self.app.post("/chat/{model_name}")
        async def chat(model_name: str, request: ChatRequest) -> ChatResponse:
            """单次对话接口"""
            if model_name not in self.engines:
                raise HTTPException(status_code=404, detail=f"模型 {model_name} 未找到")
            
            engine = self.engines[model_name]
            
            # 设置采样参数
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens
            )
            
            # 记录开始时间
            start_time = time.time()
            
            try:
                # 异步生成
                results = await engine.generate(request.message, sampling_params, request_id=f"chat_{time.time()}")
                
                # 提取结果
                output = results.outputs[0]
                response_text = output.text
                
                # 计算统计信息
                end_time = time.time()
                inference_time = end_time - start_time
                
                return ChatResponse(
                    response=response_text,
                    model=model_name,
                    usage={
                        "prompt_tokens": len(results.prompt_token_ids),
                        "completion_tokens": len(output.token_ids),
                        "total_tokens": len(results.prompt_token_ids) + len(output.token_ids)
                    },
                    timing={
                        "inference_time": inference_time,
                        "tokens_per_second": len(output.token_ids) / inference_time
                    }
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")
        
        @self.app.post("/batch/{model_name}")
        async def batch_inference(model_name: str, request: BatchRequest):
            """批量推理接口"""
            if model_name not in self.engines:
                raise HTTPException(status_code=404, detail=f"模型 {model_name} 未找到")
            
            engine = self.engines[model_name]
            
            # 设置采样参数
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens
            )
            
            start_time = time.time()
            
            try:
                # 批量生成
                results = []
                for i, prompt in enumerate(request.prompts):
                    result = await engine.generate(prompt, sampling_params, request_id=f"batch_{i}_{time.time()}")
                    results.append(result)
                
                end_time = time.time()
                
                # 整理结果
                responses = []
                total_tokens = 0
                
                for result in results:
                    output = result.outputs[0]
                    responses.append({
                        "prompt": result.prompt,
                        "response": output.text,
                        "tokens": len(output.token_ids)
                    })
                    total_tokens += len(output.token_ids)
                
                return {
                    "results": responses,
                    "model": model_name,
                    "timing": {
                        "total_time": end_time - start_time,
                        "average_tokens_per_second": total_tokens / (end_time - start_time)
                    }
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"批量推理失败: {str(e)}")
        
        @self.app.get("/health")
        async def health_check():
            """健康检查接口"""
            return {
                "status": "healthy",
                "models_loaded": len(self.engines),
                "timestamp": time.time()
            }
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """启动服务器"""
        print(f"🚀 启动vLLM推理服务...")
        print(f"   地址: http://{host}:{port}")
        print(f"   文档: http://{host}:{port}/docs")
        
        uvicorn.run(self.app, host=host, port=port)

class HighPerformanceInference:
    """高性能推理工具类"""
    
    def __init__(self, model_name: str, **kwargs):
        """
        初始化高性能推理引擎
        
        Args:
            model_name: 模型名称或路径
            **kwargs: vLLM引擎参数
        """
        default_kwargs = {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "dtype": "float16",
            "trust_remote_code": True
        }
        default_kwargs.update(kwargs)
        
        print(f"🔄 初始化vLLM引擎: {model_name}")
        self.llm = LLM(model=model_name, **default_kwargs)
        print("✅ 引擎初始化完成")
    
    def generate(self, prompts, **sampling_kwargs):
        """
        生成文本
        
        Args:
            prompts: 单个提示或提示列表
            **sampling_kwargs: 采样参数
            
        Returns:
            生成结果
        """
        default_sampling = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 512
        }
        default_sampling.update(sampling_kwargs)
        
        sampling_params = SamplingParams(**default_sampling)
        
        # 确保prompts是列表
        if isinstance(prompts, str):
            prompts = [prompts]
        
        start_time = time.time()
        outputs = self.llm.generate(prompts, sampling_params)
        end_time = time.time()
        
        # 处理结果
        results = []
        total_tokens = 0
        
        for output in outputs:
            generated_text = output.outputs[0].text
            output_tokens = len(output.outputs[0].token_ids)
            total_tokens += output_tokens
            
            results.append({
                "prompt": output.prompt,
                "generated_text": generated_text,
                "tokens": output_tokens
            })
        
        # 计算性能指标
        total_time = end_time - start_time
        tokens_per_second = total_tokens / total_time
        
        print(f"⚡ 生成完成:")
        print(f"   总时间: {total_time:.2f}s")
        print(f"   总tokens: {total_tokens}")
        print(f"   生成速度: {tokens_per_second:.1f} tokens/s")
        
        return results if len(results) > 1 else results[0]
    
    def benchmark(self, test_prompts=None, iterations=3):
        """性能基准测试"""
        if test_prompts is None:
            test_prompts = [
                "介绍一下海光DCU的技术特点",
                "解释深度学习的基本原理",
                "Python编程语言的优势有哪些？",
                "描述人工智能在医疗领域的应用",
                "什么是大规模语言模型？"
            ]
        
        print(f"🏃 开始vLLM性能基准测试...")
        print(f"   测试提示数量: {len(test_prompts)}")
        print(f"   测试轮数: {iterations}")
        
        all_results = []
        
        for i in range(iterations):
            print(f"\n轮次 {i+1}/{iterations}:")
            
            start_time = time.time()
            results = self.generate(test_prompts, max_tokens=256)
            end_time = time.time()
            
            total_time = end_time - start_time
            total_tokens = sum(r["tokens"] for r in results)
            tokens_per_second = total_tokens / total_time
            
            all_results.append({
                "time": total_time,
                "tokens": total_tokens,
                "tps": tokens_per_second
            })
            
            print(f"   时间: {total_time:.2f}s")
            print(f"   tokens: {total_tokens}")
            print(f"   速度: {tokens_per_second:.1f} tokens/s")
        
        # 计算平均值
        avg_time = sum(r["time"] for r in all_results) / len(all_results)
        avg_tokens = sum(r["tokens"] for r in all_results) / len(all_results)
        avg_tps = sum(r["tps"] for r in all_results) / len(all_results)
        
        print(f"\n📊 基准测试结果 (平均值):")
        print(f"   平均时间: {avg_time:.2f}s")
        print(f"   平均tokens: {avg_tokens:.0f}")
        print(f"   平均速度: {avg_tps:.1f} tokens/s")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM高性能推理")
    parser.add_argument("--mode", choices=["server", "client", "benchmark"], default="client", help="运行模式")
    parser.add_argument("--model", default="Qwen/Qwen-7B-Chat", help="模型名称或路径")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--prompt", type=str, help="测试提示")
    
    args = parser.parse_args()
    
    if args.mode == "server":
        # 启动服务器模式
        config = ModelConfig(
            name="default",
            model_path=args.model,
            tensor_parallel_size=1
        )
        server = vLLMServer([config])
        server.run(host=args.host, port=args.port)
        
    elif args.mode == "client":
        # 客户端推理模式
        engine = HighPerformanceInference(args.model)
        
        if args.prompt:
            result = engine.generate(args.prompt)
            print(f"输入: {result['prompt']}")
            print(f"输出: {result['generated_text']}")
        else:
            # 交互模式
            print("🤖 vLLM交互式推理 (输入'exit'退出)")
            while True:
                prompt = input("\n👤 输入: ").strip()
                if prompt.lower() in ['exit', 'quit', '退出']:
                    break
                
                result = engine.generate(prompt)
                print(f"🤖 输出: {result['generated_text']}")
    
    elif args.mode == "benchmark":
        # 基准测试模式
        engine = HighPerformanceInference(args.model)
        engine.benchmark()

if __name__ == "__main__":
    main() 