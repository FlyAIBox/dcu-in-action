# SPDX-License-Identifier: Apache-2.0
"""
后端请求处理模块 - 支持多种大模型推理后端的统一请求接口

这个模块为不同的大模型推理后端提供统一的异步请求接口，支持：
- vLLM (OpenAI兼容API)
- TGI (Text Generation Inference)
- TensorRT-LLM
- DeepSpeed-MII
- OpenAI官方API
- 其他OpenAI兼容的推理服务

主要功能:
1. 统一的请求/响应数据结构
2. 异步HTTP请求处理
3. 流式响应解析
4. 性能指标收集 (TTFT, ITL等)
5. 错误处理和重试机制

针对DCU优化:
- 支持海光DCU的HIP运行时环境
- 优化了大模型推理的网络通信
- 提供详细的性能监控数据
"""

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
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

# 注意: 这里不导入vLLM，使得基准测试脚本可以在没有安装vLLM的环境中运行
# 这样可以测试其他推理后端而不依赖vLLM

# HTTP客户端超时配置: 6小时，适应大模型长时间推理场景
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    """
    请求输入数据结构 - 封装发送给推理后端的请求参数

    这个数据类标准化了不同后端的请求格式，提供统一的接口。
    """
    prompt: str                                    # 输入提示文本
    api_url: str                                   # 推理服务的API端点URL
    prompt_len: int                                # 输入prompt的token数量
    output_len: int                                # 期望输出的token数量
    model: str                                     # 模型标识符或路径
    model_name: Optional[str] = None               # 服务中的模型名称 (可选)
    logprobs: Optional[int] = None                 # 返回的logprobs数量 (可选)
    extra_body: Optional[dict] = None              # 额外的请求参数 (如采样参数)
    multi_modal_content: Optional[dict] = None     # 多模态内容 (图像、音频等)
    ignore_eos: bool = False                       # 是否忽略结束符，强制生成指定长度


@dataclass
class RequestFuncOutput:
    """
    请求输出数据结构 - 封装推理后端返回的响应和性能指标

    这个数据类统一了不同后端的响应格式，并收集了详细的性能数据。
    """
    generated_text: str = ""                       # 生成的文本内容
    success: bool = False                          # 请求是否成功完成
    latency: float = 0.0                          # 总延迟时间 (秒)
    output_tokens: int = 0                         # 实际输出的token数量
    ttft: float = 0.0                             # 首次token时间 (Time to First Token, 秒)
    itl: list[float] = field(default_factory=list) # 迭代延迟列表 (Inter-Token Latency, 秒)
    tpot: float = 0.0                             # 平均每token时间 (Time Per Output Token, 秒)
    prompt_len: int = 0                           # 输入prompt长度 (用于验证)
    error: str = ""                               # 错误信息 (如果请求失败)


async def async_request_tgi(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        params = {
            "max_new_tokens": request_func_input.output_len,
            "do_sample": True,
            "temperature": 0.01,  # TGI does not accept 0.0 temperature.
            "top_p": 0.99,  # TGI does not accept 1.0 top_p.
            "truncate": request_func_input.prompt_len,
            "ignore_eos_token": request_func_input.ignore_eos,
        }
        payload = {
            "inputs": request_func_input.prompt,
            "parameters": params,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        if request_func_input.ignore_eos:
            output.output_tokens = request_func_input.output_len
        else:
            output.output_tokens = None

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
                        chunk_bytes = chunk_bytes.decode("utf-8")

                        # NOTE: Sometimes TGI returns a ping response without
                        # any data, we should skip it.
                        if chunk_bytes.startswith(":"):
                            continue
                        chunk = chunk_bytes.removeprefix("data:")

                        data = json.loads(chunk)
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp -
                                              most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True
                    output.generated_text = data["generated_text"]
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


async def async_request_trt_llm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "accumulate_tokens": True,
            "text_input": request_func_input.prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
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

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data:")

                        data = json.loads(chunk)
                        output.generated_text += data["text_output"]
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = timestamp - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp -
                                              most_recent_timestamp)

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
    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:

        payload = {
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
            async with session.post(url=request_func_input.api_url,
                                    json=payload) as response:
                if response.status == 200:
                    parsed_resp = await response.json()
                    output.latency = time.perf_counter() - st
                    if "choices" in parsed_resp:
                        output.generated_text = parsed_resp["choices"][0][
                            "text"]
                    elif "text" in parsed_resp:
                        output.generated_text = parsed_resp["text"][0]
                    else:
                        output.error = ("Unexpected response format: "
                                        "neither 'choices' nor 'text' found")
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
    """
    OpenAI Completions API异步请求函数 - 支持vLLM等OpenAI兼容的推理服务

    这是最重要的请求函数，用于与vLLM服务进行通信。支持流式响应和详细的性能指标收集。

    参数:
        request_func_input: RequestFuncInput - 请求输入参数
        pbar: Optional[tqdm] - 进度条对象 (可选)

    返回:
        RequestFuncOutput - 包含生成文本和性能指标的响应对象

    功能特性:
        1. 流式响应处理: 实时接收生成的token
        2. 性能指标收集: 精确测量TTFT和ITL
        3. 错误处理: 完整的异常捕获和错误报告
        4. 进度跟踪: 支持进度条更新

    适用后端:
        - vLLM (主要目标)
        - LMDeploy
        - SGLang
        - ScaleLLM
        - 其他OpenAI兼容服务
    """
    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("completions", "profile")
    ), "OpenAI Completions API URL must end with 'completions' or 'profile'."

    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model_name \
                if request_func_input.model_name else request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
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
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data: ")
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
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += text or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get(
                                    "completion_tokens")
                    if first_chunk_received:
                        output.success = True
                    else:
                        output.success = False
                        output.error = (
                            "Never received a valid chunk to calculate TTFT."
                            "This response will be marked as failed!")
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
    assert api_url.endswith(
        ("chat/completions", "profile")
    ), "OpenAI Chat Completions API URL must end with 'chat/completions'."

    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        content = [{"type": "text", "text": request_func_input.prompt}]
        if request_func_input.multi_modal_content:
            content.append(request_func_input.multi_modal_content)
        payload = {
            "model": request_func_input.model_name \
                if request_func_input.model_name else request_func_input.model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                },
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
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data: ")
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
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                generated_text += content or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get(
                                    "completion_tokens")

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
    if os.getenv('VLLM_USE_MODELSCOPE', 'False').lower() == 'true':
        from modelscope import snapshot_download

        from vllm.model_executor.model_loader.weight_utils import get_lock

        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(pretrained_model_name_or_path):
            model_path = snapshot_download(
                model_id=pretrained_model_name_or_path,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"])

            return model_path
    return pretrained_model_name_or_path


def get_tokenizer(
    pretrained_model_name_or_path: str,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    if pretrained_model_name_or_path is not None and not os.path.exists(
            pretrained_model_name_or_path):
        pretrained_model_name_or_path = get_model(
            pretrained_model_name_or_path)
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError(
                "Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False
    if tokenizer_mode == "mistral":
        try:
            from vllm.transformers_utils.tokenizer import MistralTokenizer
        except ImportError as e:
            raise ImportError("MistralTokenizer requires vllm package.\n"
                              "Please install it with `pip install vllm` "
                              "to use mistral tokenizer mode.") from e
        return MistralTokenizer.from_pretrained(
            str(pretrained_model_name_or_path))
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
    "tensorrt-llm": async_request_trt_llm,
    "scalellm": async_request_openai_completions,
    "sglang": async_request_openai_completions,
}

OPENAI_COMPATIBLE_BACKENDS = [
    k for k, v in ASYNC_REQUEST_FUNCS.items()
    if v in (async_request_openai_completions,
             async_request_openai_chat_completions)
]
