#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
backend_request_func.py - 后端请求处理模块

本模块提供了与各种不同后端服务进行交互的异步请求函数，用于LLM(大语言模型)推理基准测试。
主要功能包括：
1. 支持多种后端服务：TGI、TRT-LLM、DeepSpeed MII、OpenAI API等
2. 异步请求处理：提供高性能的并发请求处理能力
3. 性能指标收集：自动收集TTFT(首个token时间)、TPOT(每token时间)、延迟等关键指标
4. 错误处理：完整的异常处理和错误信息记录
5. 多模态支持：支持文本、音频等多种输入模式

类和函数说明：
- RequestFuncInput: 请求输入数据结构
- RequestFuncOutput: 请求输出数据结构  
- async_request_*: 各种后端的异步请求处理函数
- get_model/get_tokenizer: 模型和分词器获取工具函数

作者：vLLM团队
修改：添加详细中文注释
"""

# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional, Union

import aiohttp
import huggingface_hub.constants
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

# 注意：这里不导入vLLM，这样基准测试脚本可以在没有安装vLLM的环境下运行
# NOTE(simon): do not import vLLM here so the benchmark script
# can run without vLLM installed.

# 设置HTTP请求超时时间为6小时，用于处理长时间运行的推理任务
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    """
    请求函数输入数据结构
    
    定义了发送到后端服务的请求所需的所有参数和配置信息。
    这个数据类包含了执行一次推理请求所需的完整信息。
    """
    prompt: str                                    # 输入提示词文本
    api_url: str                                   # 后端API服务的URL地址
    prompt_len: int                                # 提示词的token长度
    output_len: int                                # 期望输出的token长度
    model: str                                     # 模型名称或标识符
    model_name: Optional[str] = None               # 可选的模型显示名称
    logprobs: Optional[int] = None                 # 返回的对数概率数量（可选）
    extra_body: Optional[dict] = None              # 额外的请求体参数（可选）
    multi_modal_content: Optional[dict] = None     # 多模态内容（如图像、音频等）
    ignore_eos: bool = False                       # 是否忽略EOS（结束符）token
    language: Optional[str] = None                 # 语言标识（用于音频等特定任务）


@dataclass
class RequestFuncOutput:
    """
    请求函数输出数据结构
    
    包含了一次推理请求完成后的所有结果和性能指标。
    这些指标用于后续的性能分析和基准测试评估。
    """
    generated_text: str = ""                       # 生成的文本内容
    success: bool = False                          # 请求是否成功完成
    latency: float = 0.0                          # 总延迟时间（秒）
    output_tokens: int = 0                         # 实际输出的token数量
    ttft: float = 0.0                             # 首个token生成时间（Time to First Token）
    itl: list[float] = field(default_factory=list) # 每个token之间的延迟列表（Inter-Token Latencies）
    tpot: float = 0.0                             # 平均每token生成时间（Time Per Output Token）
    prompt_len: int = 0                           # 输入提示词长度
    error: str = ""                               # 错误信息（如果请求失败）


async def async_request_tgi(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    """
    TGI (Text Generation Inference) 后端异步请求处理函数
    
    向TGI服务发送流式生成请求，并收集详细的性能指标。
    TGI是Hugging Face开发的高性能文本生成推理服务。
    
    Args:
        request_func_input: 包含请求参数的输入对象
        pbar: 可选的进度条对象，用于显示进度
        
    Returns:
        RequestFuncOutput: 包含生成结果和性能指标的输出对象
        
    Note:
        - TGI要求API URL以"generate_stream"结尾
        - 使用流式响应来精确测量token级别的延迟
        - 支持温度、top_p等生成参数的配置
    """
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        # 构建TGI请求参数
        params = {
            "max_new_tokens": request_func_input.output_len,  # 最大新生成token数
            "do_sample": True,                               # 启用采样
            "temperature": 0.01,                             # 温度参数（TGI不接受0.0）
            "top_p": 0.99,                                   # Top-p采样参数（TGI不接受1.0）
            "truncate": request_func_input.prompt_len,       # 截断长度
            "ignore_eos_token": request_func_input.ignore_eos, # 是否忽略结束token
        }
        payload = {
            "inputs": request_func_input.prompt,
            "parameters": params,
        }
        
        # 初始化输出对象
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        if request_func_input.ignore_eos:
            output.output_tokens = request_func_input.output_len
        else:
            output.output_tokens = None

        ttft = 0.0  # 首个token时间
        st = time.perf_counter()  # 请求开始时间
        most_recent_timestamp = st
        
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    # 处理流式响应
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        chunk_bytes = chunk_bytes.decode("utf-8")

                        # 跳过TGI的ping响应
                        if chunk_bytes.startswith(":"):
                            continue
                        chunk = chunk_bytes.removeprefix("data:")

                        data = json.loads(chunk)
                        timestamp = time.perf_counter()
                        
                        # 记录首个token时间
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft
                        # 记录token间延迟
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    # 计算总延迟和设置成功状态
                    output.latency = most_recent_timestamp - st
                    output.success = True
                    output.generated_text = data["generated_text"]
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            # 异常处理：记录详细的错误信息
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        # 更新进度条
        if pbar:
            pbar.update(1)
        return output


async def async_request_trt_llm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    """
    TensorRT-LLM 后端异步请求处理函数
    
    向TensorRT-LLM服务发送流式生成请求。TensorRT-LLM是NVIDIA开发的
    高性能GPU推理引擎，专门优化大语言模型推理性能。
    
    Args:
        request_func_input: 包含请求参数的输入对象
        pbar: 可选的进度条对象
        
    Returns:
        RequestFuncOutput: 包含生成结果和性能指标的输出对象
        
    Note:
        - 支持累积token输出模式
        - 针对NVIDIA GPU优化的推理引擎
        - 提供低延迟、高吞吐量的推理性能
    """
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        # 构建TensorRT-LLM请求参数
        payload = {
            "accumulate_tokens": True,                      # 累积token模式
            "text_input": request_func_input.prompt,        # 输入文本
            "temperature": 0.0,                             # 温度参数
            "top_p": 1.0,                                   # Top-p参数
            "max_tokens": request_func_input.output_len,    # 最大token数
            "stream": True,                                 # 启用流式输出
        }
        if request_func_input.ignore_eos:
            payload["min_length"] = request_func_input.output_len
            
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix("data:")
                        data = json.loads(chunk)
                        output.generated_text += data["text_output"]
                        timestamp = time.perf_counter()
                        
                        # 记录首个token时间
                        if ttft == 0.0:
                            ttft = timestamp - st
                            output.ttft = ttft
                        # 记录token间延迟
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_deepspeed_mii(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "max_tokens": request_func_input.output_len,
            "temperature": 0.01,  # deepspeed-mii does not accept 0.0 temp.
            "top_p": 1.0,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        # NOTE: DeepSpeed-MII doesn't support streaming as of Jan 28 2024,
        # will use 0 as placeholder.
        # See https://github.com/microsoft/DeepSpeed-MII/pull/311
        output.ttft = 0

        st = time.perf_counter()
        try:
            async with session.post(
                url=request_func_input.api_url, json=payload
            ) as response:
                if response.status == 200:
                    parsed_resp = await response.json()
                    output.latency = time.perf_counter() - st
                    if "choices" in parsed_resp:
                        output.generated_text = parsed_resp["choices"][0]["text"]
                    elif "text" in parsed_resp:
                        output.generated_text = parsed_resp["text"][0]
                    else:
                        output.error = (
                            "Unexpected response format: "
                            "neither 'choices' nor 'text' found"
                        )
                        output.success = False
                    output.success = True
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(("completions", "profile")), (
        "OpenAI Completions API URL must end with 'completions' or 'profile'."
    )

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        ### Revised: Add min_tokens to the payload
        ### to ensure the model generates at least the specified number of tokens.
        payload = {
            "model": request_func_input.model_name
            if request_func_input.model_name
            else request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "max_tokens": request_func_input.output_len,
            "min_tokens": request_func_input.output_len,
            "logprobs": request_func_input.logprobs,
            "stream": True,
            
            "stream_options": {
                "include_usage": True,
            },
        }
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                        if chunk != "[DONE]":
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if choices := data.get("choices"):
                                # Note that text could be empty here
                                # e.g. for special tokens
                                text = choices[0].get("text")
                                timestamp = time.perf_counter()
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += text or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")
                    if first_chunk_received:
                        output.success = True
                    else:
                        output.success = False
                        output.error = (
                            "Never received a valid chunk to calculate TTFT."
                            "This response will be marked as failed!"
                        )
                    output.generated_text = generated_text
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(("chat/completions", "profile")), (
        "OpenAI Chat Completions API URL must end with 'chat/completions'."
    )

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        content = [{"type": "text", "text": request_func_input.prompt}]
        if request_func_input.multi_modal_content:
            content.append(request_func_input.multi_modal_content)
        payload = {
            "model": request_func_input.model_name
            if request_func_input.model_name
            else request_func_input.model,
            "messages": [
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "max_completion_tokens": request_func_input.output_len,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                        if chunk != "[DONE]":
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            if choices := data.get("choices"):
                                content = choices[0]["delta"].get("content")
                                # First token
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                generated_text += content or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")

                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_audio(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    # Lazy import without PlaceholderModule to avoid vllm dep.
    import soundfile

    api_url = request_func_input.api_url
    assert api_url.endswith(("transcriptions", "translations")), (
        "OpenAI Chat Completions API URL must end with 'transcriptions' "
    )
    "or `translations`."

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        content = [{"type": "text", "text": request_func_input.prompt}]
        payload = {
            "model": request_func_input.model_name
            if request_func_input.model_name
            else request_func_input.model,
            "temperature": 0.0,
            "max_completion_tokens": request_func_input.output_len,
            "stream": True,
            "language": "en",
            # Flattened due to multipart/form-data
            "stream_include_usage": True,
            "stream_continuous_usage_stats": True,
        }
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        # Send audio file
        def to_bytes(y, sr):
            buffer = io.BytesIO()
            soundfile.write(buffer, y, sr, format="WAV")
            buffer.seek(0)
            return buffer

        with to_bytes(*request_func_input.multi_modal_content["audio"]) as f:
            form = aiohttp.FormData()
            form.add_field("file", f, content_type="audio/wav")
            for key, value in payload.items():
                form.add_field(key, str(value))

            output = RequestFuncOutput()
            output.prompt_len = request_func_input.prompt_len

            generated_text = ""
            ttft = 0.0
            st = time.perf_counter()
            most_recent_timestamp = st
            try:
                async with session.post(
                    url=api_url, data=form, headers=headers
                ) as response:
                    if response.status == 200:
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue

                            chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                            if chunk != "[DONE]":
                                timestamp = time.perf_counter()
                                data = json.loads(chunk)

                                if choices := data.get("choices"):
                                    content = choices[0]["delta"].get("content")
                                    # First token
                                    if ttft == 0.0:
                                        ttft = timestamp - st
                                        output.ttft = ttft

                                    # Decoding phase
                                    else:
                                        output.itl.append(
                                            timestamp - most_recent_timestamp
                                        )

                                    generated_text += content or ""
                                elif usage := data.get("usage"):
                                    output.output_tokens = usage.get(
                                        "completion_tokens"
                                    )

                                most_recent_timestamp = timestamp

                        output.generated_text = generated_text
                        output.success = True
                        output.latency = most_recent_timestamp - st
                    else:
                        output.error = response.reason or ""
                        output.success = False
            except Exception:
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


def get_model(pretrained_model_name_or_path: str) -> str:
    if os.getenv("VLLM_USE_MODELSCOPE", "False").lower() == "true":
        from modelscope import snapshot_download

        from vllm.model_executor.model_loader.weight_utils import get_lock

        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(pretrained_model_name_or_path):
            model_path = snapshot_download(
                model_id=pretrained_model_name_or_path,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"],
            )

            return model_path
    return pretrained_model_name_or_path


def get_tokenizer(
    pretrained_model_name_or_path: str,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    if pretrained_model_name_or_path is not None and not os.path.exists(
        pretrained_model_name_or_path
    ):
        pretrained_model_name_or_path = get_model(pretrained_model_name_or_path)
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False
    if tokenizer_mode == "mistral":
        try:
            from vllm.transformers_utils.tokenizer import MistralTokenizer
        except ImportError as e:
            raise ImportError(
                "MistralTokenizer requires vllm package.\n"
                "Please install it with `pip install vllm` "
                "to use mistral tokenizer mode."
            ) from e
        return MistralTokenizer.from_pretrained(str(pretrained_model_name_or_path))
    else:
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )


ASYNC_REQUEST_FUNCS = {
    "tgi": async_request_tgi,
    "vllm": async_request_openai_completions,
    "lmdeploy": async_request_openai_completions,
    "deepspeed-mii": async_request_deepspeed_mii,
    "openai": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
    "openai-audio": async_request_openai_audio,
    "tensorrt-llm": async_request_trt_llm,
    "scalellm": async_request_openai_completions,
    "sglang": async_request_openai_completions,
}

OPENAI_COMPATIBLE_BACKENDS = [
    k
    for k, v in ASYNC_REQUEST_FUNCS.items()
    if v in (async_request_openai_completions, async_request_openai_chat_completions)
]