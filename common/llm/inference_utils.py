"""
大模型推理工具库
提供模型加载、批量推理、流式推理、vLLM集成、量化推理等功能
"""

import os
import json
import time
import torch
import asyncio
import threading
from typing import Dict, Any, Optional, List, Union, Generator, AsyncGenerator, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from queue import Queue
from contextlib import contextmanager
import gc

try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoConfig,
        GenerationConfig, TextStreamer,
        AutoModelForCausalLM,
        BitsAndBytesConfig
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

try:
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

from ..utils.logger import get_logger, performance_monitor
from ..utils.monitor import SystemMonitor
from ..dcu.device_manager import DCUManager

logger = get_logger(__name__)


@dataclass
class InferenceConfig:
    """推理配置类"""
    # 模型配置
    model_name_or_path: str
    tokenizer_path: Optional[str] = None
    device_map: Union[str, Dict] = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = True
    
    # 生成配置
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    
    # 批量推理配置
    batch_size: int = 1
    max_batch_size: int = 32
    padding_side: str = "left"
    
    # 量化配置
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    
    # vLLM配置
    use_vllm: bool = False
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    
    # 其他配置
    enable_streaming: bool = False
    stream_interval: int = 2
    max_workers: int = 4
    
    def to_generation_config(self) -> Dict[str, Any]:
        """转换为GenerationConfig格式"""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
            "pad_token_id": None,  # 将在运行时设置
            "eos_token_id": None,  # 将在运行时设置
        }


class ModelLoader:
    """模型加载器"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.dcu_manager = DCUManager()
        self.monitor = SystemMonitor()
        
    @performance_monitor
    def load_model_and_tokenizer(self) -> tuple:
        """加载模型和分词器"""
        logger.info(f"开始加载模型: {self.config.model_name_or_path}")
        
        # 加载分词器
        tokenizer_path = self.config.tokenizer_path or self.config.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=self.config.trust_remote_code,
            padding_side=self.config.padding_side
        )
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 量化配置
        quantization_config = None
        if self.config.load_in_4bit or self.config.load_in_8bit:
            if not HAS_TRANSFORMERS:
                raise ImportError("需要安装transformers库进行量化推理")
                
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            )
        
        # 加载模型
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "device_map": self.config.device_map,
            "quantization_config": quantization_config,
        }
        
        if self.config.torch_dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, self.config.torch_dtype)
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                **model_kwargs
            )
            
            # 设置评估模式
            model.eval()
            
            logger.info(f"模型加载成功: {type(model).__name__}")
            logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise


class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.dcu_manager = DCUManager()
        
        # 初始化
        self._setup()
    
    def _setup(self):
        """初始化推理引擎"""
        # 加载模型和分词器
        loader = ModelLoader(self.config)
        self.model, self.tokenizer = loader.load_model_and_tokenizer()
        
        # 创建生成配置
        gen_config = self.config.to_generation_config()
        gen_config["pad_token_id"] = self.tokenizer.pad_token_id
        gen_config["eos_token_id"] = self.tokenizer.eos_token_id
        self.generation_config = GenerationConfig(**gen_config)
        
        logger.info("推理引擎初始化完成")
    
    @performance_monitor
    def generate(self, 
                 prompt: str, 
                 **kwargs) -> str:
        """单条文本生成"""
        # 编码输入
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        
        # 移动到设备
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 合并配置
        generation_config = GenerationConfig(**{
            **asdict(self.generation_config),
            **kwargs
        })
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # 解码输出
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    @performance_monitor
    def batch_generate(self, 
                      prompts: List[str], 
                      **kwargs) -> List[str]:
        """批量生成"""
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # 编码输入
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # 移动到设备
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 合并配置
            generation_config = GenerationConfig(**{
                **asdict(self.generation_config),
                **kwargs
            })
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # 解码输出
            batch_responses = []
            for j, output in enumerate(outputs):
                input_len = inputs["input_ids"][j].shape[0]
                generated_tokens = output[input_len:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                batch_responses.append(response.strip())
            
            results.extend(batch_responses)
            
            # 清理内存
            del inputs, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return results
    
    def stream_generate(self, 
                       prompt: str, 
                       **kwargs) -> Generator[str, None, None]:
        """流式生成"""
        if not self.config.enable_streaming:
            # 如果未启用流式，则返回完整结果
            result = self.generate(prompt, **kwargs)
            yield result
            return
        
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 流式生成参数
        generation_config = GenerationConfig(**{
            **asdict(self.generation_config),
            **kwargs
        })
        
        # 使用TextStreamer进行流式输出
        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # 生成
        with torch.no_grad():
            self.model.generate(
                **inputs,
                generation_config=generation_config,
                streamer=streamer
            )
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             **kwargs) -> str:
        """对话生成（支持多轮对话）"""
        # 构建对话prompt（这里使用简单的格式，实际应根据模型调整）
        prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        
        prompt += "Assistant: "
        
        return self.generate(prompt, **kwargs)
    
    def evaluate_perplexity(self, 
                          texts: List[str]) -> Dict[str, float]:
        """计算困惑度"""
        total_log_likelihood = 0.0
        total_tokens = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for text in texts:
                # 编码
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 前向传播
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                
                # 计算对数似然
                log_likelihood = -outputs.loss.item() * inputs["input_ids"].size(1)
                total_log_likelihood += log_likelihood
                total_tokens += inputs["input_ids"].size(1)
        
        # 计算困惑度
        avg_log_likelihood = total_log_likelihood / total_tokens
        perplexity = torch.exp(torch.tensor(-avg_log_likelihood)).item()
        
        return {
            "perplexity": perplexity,
            "avg_log_likelihood": avg_log_likelihood,
            "total_tokens": total_tokens
        }
    
    @contextmanager
    def inference_mode(self):
        """推理模式上下文管理器"""
        try:
            self.model.eval()
            torch.set_grad_enabled(False)
            yield
        finally:
            torch.set_grad_enabled(True)
    
    def cleanup(self):
        """清理资源"""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        logger.info("推理引擎资源已清理")


class vLLMInferenceEngine:
    """vLLM推理引擎"""
    
    def __init__(self, config: InferenceConfig):
        if not HAS_VLLM:
            raise ImportError("需要安装vllm库: pip install vllm")
        
        self.config = config
        self.engine = None
        self.sampling_params = None
        self._setup()
    
    def _setup(self):
        """初始化vLLM引擎"""
        # 创建vLLM引擎
        self.engine = LLM(
            model=self.config.model_name_or_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        # 创建采样参数
        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_tokens=self.config.max_new_tokens,
        )
        
        logger.info("vLLM推理引擎初始化完成")
    
    @performance_monitor
    def generate(self, prompt: str, **kwargs) -> str:
        """单条生成"""
        # 更新采样参数
        sampling_params = SamplingParams(**{
            **asdict(self.sampling_params),
            **kwargs
        })
        
        # 生成
        outputs = self.engine.generate([prompt], sampling_params)
        
        return outputs[0].outputs[0].text.strip()
    
    @performance_monitor
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """批量生成"""
        # 更新采样参数
        sampling_params = SamplingParams(**{
            **asdict(self.sampling_params),
            **kwargs
        })
        
        # 批量生成
        outputs = self.engine.generate(prompts, sampling_params)
        
        return [output.outputs[0].text.strip() for output in outputs]


class InferenceServer:
    """推理服务器"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.engine = None
        self.request_queue = Queue()
        self.result_cache = {}
        self.is_running = False
        
        # 初始化推理引擎
        if config.use_vllm:
            self.engine = vLLMInferenceEngine(config)
        else:
            self.engine = InferenceEngine(config)
    
    def start_server(self):
        """启动推理服务器"""
        self.is_running = True
        
        # 启动工作线程
        for i in range(self.config.max_workers):
            thread = threading.Thread(target=self._worker, daemon=True)
            thread.start()
        
        logger.info(f"推理服务器已启动，工作线程数: {self.config.max_workers}")
    
    def stop_server(self):
        """停止推理服务器"""
        self.is_running = False
        logger.info("推理服务器已停止")
    
    def _worker(self):
        """工作线程"""
        while self.is_running:
            try:
                # 获取请求
                if not self.request_queue.empty():
                    request_id, prompt, kwargs = self.request_queue.get(timeout=1.0)
                    
                    # 执行推理
                    try:
                        result = self.engine.generate(prompt, **kwargs)
                        self.result_cache[request_id] = {"status": "success", "result": result}
                    except Exception as e:
                        self.result_cache[request_id] = {"status": "error", "error": str(e)}
                    
                    self.request_queue.task_done()
                
            except Exception:
                continue
    
    def submit_request(self, 
                      prompt: str, 
                      request_id: Optional[str] = None, 
                      **kwargs) -> str:
        """提交推理请求"""
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000000)}"
        
        # 添加到队列
        self.request_queue.put((request_id, prompt, kwargs))
        
        return request_id
    
    def get_result(self, request_id: str, timeout: int = 30) -> Dict[str, Any]:
        """获取推理结果"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if request_id in self.result_cache:
                result = self.result_cache.pop(request_id)
                return result
            
            time.sleep(0.1)
        
        return {"status": "timeout", "error": "请求超时"}


# 工具函数
def create_inference_engine(config: Union[InferenceConfig, Dict[str, Any]]) -> Union[InferenceEngine, vLLMInferenceEngine]:
    """创建推理引擎的工厂函数"""
    if isinstance(config, dict):
        config = InferenceConfig(**config)
    
    if config.use_vllm:
        return vLLMInferenceEngine(config)
    else:
        return InferenceEngine(config)


def load_model_with_config(model_path: str, 
                          **config_kwargs) -> InferenceEngine:
    """便捷的模型加载函数"""
    config = InferenceConfig(
        model_name_or_path=model_path,
        **config_kwargs
    )
    
    return InferenceEngine(config)


# 预设配置
INFERENCE_PRESETS = {
    "fast": InferenceConfig(
        model_name_or_path="",
        max_new_tokens=512,
        temperature=0.7,
        batch_size=8,
        load_in_4bit=True,
    ),
    "quality": InferenceConfig(
        model_name_or_path="",
        max_new_tokens=2048,
        temperature=0.7,
        batch_size=4,
        load_in_4bit=False,
    ),
    "vllm_fast": InferenceConfig(
        model_name_or_path="",
        use_vllm=True,
        tensor_parallel_size=1,
        max_new_tokens=1024,
        temperature=0.7,
    ),
}


def get_preset_config(preset_name: str, model_path: str) -> InferenceConfig:
    """获取预设配置"""
    if preset_name not in INFERENCE_PRESETS:
        raise ValueError(f"未知预设: {preset_name}，可用预设: {list(INFERENCE_PRESETS.keys())}")
    
    config = INFERENCE_PRESETS[preset_name]
    config.model_name_or_path = model_path
    
    return config 