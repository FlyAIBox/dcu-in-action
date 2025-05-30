#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLaMA Factory模型推理服务
作者: DCU实战项目组
"""

import os
import sys
import json
import asyncio
import argparse
import logging
import time
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil
import GPUtil

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量
inference_engine = None

# API请求/响应模型
class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    history: Optional[List[List[str]]] = Field(default=[], description="对话历史")
    max_length: Optional[int] = Field(default=2048, description="最大生成长度")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0, description="top-p参数")
    do_sample: Optional[bool] = Field(default=True, description="是否采样")

class ChatResponse(BaseModel):
    response: str = Field(..., description="模型回复")
    history: List[List[str]] = Field(..., description="更新后的对话历史")
    timestamp: float = Field(..., description="响应时间戳")
    inference_time: float = Field(..., description="推理耗时(秒)")

class BatchRequest(BaseModel):
    messages: List[str] = Field(..., description="消息列表")
    max_length: Optional[int] = Field(default=2048, description="最大生成长度")
    temperature: Optional[float] = Field(default=0.7, description="温度参数")
    top_p: Optional[float] = Field(default=0.9, description="top-p参数")

class BatchResponse(BaseModel):
    responses: List[str] = Field(..., description="回复列表")
    total_time: float = Field(..., description="总耗时(秒)")
    avg_time_per_request: float = Field(..., description="平均单条耗时(秒)")

class ModelInfo(BaseModel):
    model_name: str = Field(..., description="模型名称")
    model_path: str = Field(..., description="模型路径")
    device: str = Field(..., description="运行设备")
    memory_usage: Dict[str, float] = Field(..., description="内存使用情况")
    total_requests: int = Field(..., description="总请求数")
    avg_response_time: float = Field(..., description="平均响应时间")

class HealthResponse(BaseModel):
    status: str = Field(..., description="服务状态")
    uptime: float = Field(..., description="运行时间(秒)")
    model_loaded: bool = Field(..., description="模型是否已加载")
    system_info: Dict[str, float] = Field(..., description="系统信息")

class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, model_path: str, device: str = "auto", precision: str = "fp16"):
        self.model_path = model_path
        self.device = device
        self.precision = precision
        self.model = None
        self.tokenizer = None
        self.load_time = 0
        
        self.load_model()
    
    def load_model(self):
        """加载模型"""
        start_time = time.time()
        
        print(f"正在加载模型: {self.model_path}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        # 设置模型精度
        torch_dtype = torch.float16 if self.precision == "fp16" else torch.float32
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.model.eval()
        
        self.load_time = time.time() - start_time
        
        print(f"模型加载完成，耗时: {self.load_time:.2f}秒")
        print(f"模型设备: {self.get_model_device()}")
        print(f"模型精度: {self.precision}")
    
    def get_model_device(self) -> str:
        """获取模型设备信息"""
        if hasattr(self.model, 'hf_device_map'):
            devices = set(self.model.hf_device_map.values())
            return f"GPU: {list(devices)}" if 'cuda' in str(devices) else "CPU"
        return "Unknown"
    
    def get_model_info(self) -> ModelInfo:
        """获取模型信息"""
        # 估算模型大小
        param_count = sum(p.numel() for p in self.model.parameters())
        model_size = f"{param_count / 1e9:.1f}B parameters"
        
        return ModelInfo(
            model_name=os.path.basename(self.model_path),
            model_path=self.model_path,
            device=self.get_model_device(),
            memory_usage={},
            total_requests=0,
            avg_response_time=0.0
        )
    
    def chat(self, message: str, history: List[List[str]] = None,
             max_length: int = 2048, temperature: float = 0.7,
             top_p: float = 0.9, top_k: int = 50, 
             repetition_penalty: float = 1.1) -> tuple:
        """聊天推理"""
        if history is None:
            history = []
        
        start_time = time.time()
        
        try:
            # 检查模型是否有chat方法
            if hasattr(self.model, 'chat'):
                # 使用模型自带的chat方法
                response, updated_history = self.model.chat(
                    self.tokenizer,
                    message,
                    history=history,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p
                )
            else:
                # 手动构建对话
                response, updated_history = self._manual_chat(
                    message, history, max_length, temperature, top_p, top_k, repetition_penalty
                )
            
            # 计算token使用量
            prompt_tokens = len(self.tokenizer.encode(message))
            completion_tokens = len(self.tokenizer.encode(response))
            
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "inference_time": time.time() - start_time
            }
            
            return response, updated_history, usage
            
        except Exception as e:
            print(f"推理过程中出错: {str(e)}")
            raise e
    
    def _manual_chat(self, message: str, history: List[List[str]],
                    max_length: int, temperature: float, top_p: float,
                    top_k: int, repetition_penalty: float) -> tuple:
        """手动聊天实现"""
        # 构建对话历史
        conversation = ""
        for user_msg, assistant_msg in history:
            conversation += f"用户: {user_msg}\n助手: {assistant_msg}\n"
        conversation += f"用户: {message}\n助手: "
        
        # 编码输入
        inputs = self.tokenizer(
            conversation,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        )
        
        # 生成回复
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # 更新历史
        updated_history = history + [[message, response]]
        
        return response, updated_history

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载模型
    global inference_engine
    
    model_path = os.environ.get("MODEL_PATH", "/path/to/your/model")
    device = os.environ.get("DEVICE", "auto")
    precision = os.environ.get("PRECISION", "fp16")
    
    if not os.path.exists(model_path):
        print(f"错误：模型路径不存在: {model_path}")
        print("请设置环境变量 MODEL_PATH 或在命令行中指定模型路径")
        sys.exit(1)
    
    inference_engine = InferenceEngine(model_path, device, precision)
    
    yield
    
    # 关闭时清理资源
    if inference_engine and inference_engine.model:
        del inference_engine.model
        del inference_engine.tokenizer
        torch.cuda.empty_cache()

# 创建FastAPI应用
app = FastAPI(
    title="LLaMA Factory 推理服务",
    description="基于LLaMA Factory的大模型推理API服务",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局推理服务实例
inference_service: Optional[InferenceEngine] = None

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    global inference_service
    
    if inference_service is None:
        logger.error("推理服务未初始化")
        return
    
    try:
        inference_service.load_model()
        logger.info("🚀 推理服务启动成功")
    except Exception as e:
        logger.error(f"❌ 推理服务启动失败: {e}")
        sys.exit(1)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """聊天对话接口"""
    if inference_service is None or inference_service.model is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    try:
        response, history, usage = inference_service.chat(
            request.message,
            request.history,
            request.max_length,
            request.temperature,
            request.top_p,
            request.do_sample
        )
        
        return ChatResponse(
            response=response,
            history=history,
            timestamp=time.time(),
            inference_time=usage["inference_time"]
        )
        
    except Exception as e:
        logger.error(f"聊天接口错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """流式聊天接口"""
    if inference_service is None or inference_service.model is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    async def generate():
        try:
            async for chunk in inference_service.chat_stream(
                request.message,
                request.history,
                request.max_length,
                request.temperature,
                request.top_p
            ):
                yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/batch", response_model=BatchResponse)
async def batch_inference_endpoint(request: BatchRequest):
    """批量推理接口"""
    if inference_service is None or inference_service.model is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    try:
        responses, total_time = await inference_service.batch_inference(
            request.messages,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return BatchResponse(
            responses=responses,
            total_time=total_time,
            avg_time_per_request=total_time / len(request.messages)
        )
        
    except Exception as e:
        logger.error(f"批量推理错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info", response_model=ModelInfo)
async def model_info_endpoint():
    """模型信息接口"""
    if inference_service is None:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    try:
        info = inference_service.get_model_info()
        return ModelInfo(**info)
    except Exception as e:
        logger.error(f"获取模型信息错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    try:
        system_info = {}
        uptime = 0.0
        model_loaded = False
        
        if inference_service is not None:
            system_info = inference_service.get_system_info()
            uptime = system_info.get("uptime", 0.0)
            model_loaded = inference_service.model is not None
        
        return HealthResponse(
            status="healthy" if model_loaded else "loading",
            uptime=uptime,
            model_loaded=model_loaded,
            system_info=system_info
        )
        
    except Exception as e:
        logger.error(f"健康检查错误: {e}")
        return HealthResponse(
            status="error",
            uptime=0.0,
            model_loaded=False,
            system_info={}
        )

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "LLaMA Factory 推理服务",
        "docs": "/docs",
        "health": "/health",
        "info": "/info"
    }

def create_client_example():
    """创建客户端使用示例"""
    example_code = '''
import requests
import json

# 推理服务基础URL
BASE_URL = "http://localhost:8000"

class LLaMAClient:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.history = []
    
    def chat(self, message: str, **kwargs) -> str:
        """单轮对话"""
        url = f"{self.base_url}/chat"
        payload = {
            "message": message,
            "history": self.history,
            **kwargs
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            self.history = result["history"]
            return result["response"]
            
        except requests.exceptions.RequestException as e:
            return f"请求失败: {e}"
    
    def batch_inference(self, messages: list, **kwargs) -> list:
        """批量推理"""
        url = f"{self.base_url}/batch"
        payload = {
            "messages": messages,
            **kwargs
        }
        
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            return result["responses"]
            
        except requests.exceptions.RequestException as e:
            return [f"请求失败: {e}"] * len(messages)
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        url = f"{self.base_url}/info"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def reset_history(self):
        """重置对话历史"""
        self.history = []

# 使用示例
if __name__ == "__main__":
    client = LLaMAClient()
    
    # 检查服务状态
    info = client.get_model_info()
    print(f"模型信息: {info}")
    
    # 单轮对话
    response = client.chat("你好，请介绍一下自己")
    print(f"回复: {response}")
    
    # 批量推理
    questions = [
        "什么是人工智能？",
        "解释一下机器学习的基本概念",
        "深度学习和传统机器学习有什么区别？"
    ]
    
    answers = client.batch_inference(questions)
    for q, a in zip(questions, answers):
        print(f"问题: {q}")
        print(f"回答: {a}\\n")
'''
    
    with open("client_example.py", "w", encoding="utf-8") as f:
        f.write(example_code)
    
    print("客户端示例代码已保存到: client_example.py")

def main():
    parser = argparse.ArgumentParser(description="LLaMA Factory 推理服务器")
    parser.add_argument("--model_path", required=True, help="模型路径")
    parser.add_argument("--host", default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="运行设备")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    parser.add_argument("--create_client", action="store_true", help="创建客户端示例代码")
    parser.add_argument("--log_level", default="info", choices=["debug", "info", "warning", "error"], help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # 创建客户端示例
    if args.create_client:
        create_client_example()
        return
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        logger.error(f"模型路径不存在: {args.model_path}")
        sys.exit(1)
    
    # 初始化推理服务
    global inference_service
    inference_service = InferenceEngine(args.model_path, args.device)
    
    logger.info(f"启动推理服务...")
    logger.info(f"  模型路径: {args.model_path}")
    logger.info(f"  服务地址: http://{args.host}:{args.port}")
    logger.info(f"  运行设备: {args.device}")
    
    # 启动服务
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=args.log_level
        )
    except KeyboardInterrupt:
        logger.info("服务已停止")
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 