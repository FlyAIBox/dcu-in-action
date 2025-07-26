#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_serving.py - 在线推理服务基准测试主程序

本模块是vLLM基准测试框架的核心程序，用于评估大语言模型在线推理服务的性能指标。
主要功能包括：
1. 多后端支持：支持vLLM、TGI、TensorRT-LLM、OpenAI API等多种推理后端
2. 性能指标测量：精确测量延迟、吞吐量、TTFT(首个token时间)等关键指标
3. 数据集支持：支持ShareGPT、HuggingFace、MTBench等多种数据集
4. 并发控制：支持可配置的请求并发数和请求速率控制
5. 结果分析：生成详细的性能分析报告和统计数据

使用方法：
服务器端运行：
    vllm serve <your_model> --swap-space 16 --disable-log-requests

客户端运行：
    python benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path_to_dataset> \
        --request-rate <request_rate> \
        --num-prompts <num_prompts>

注意：使用TGI后端时，需要添加 --endpoint /generate_stream 参数

作者：vLLM团队
修改：添加详细中文注释
"""

# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import gc
import json
import os
import random
import time
import warnings
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from backend_request_func import (
    ASYNC_REQUEST_FUNCS,
    OPENAI_COMPATIBLE_BACKENDS,
    RequestFuncInput,
    RequestFuncOutput,
)

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

from benchmark_dataset import (
    AIMODataset,
    ASRDataset,
    BurstGPTDataset,
    ConversationDataset,
    HuggingFaceDataset,
    InstructCoderDataset,
    MTBenchDataset,
    NextEditPredictionDataset,
    RandomDataset,
    SampleRequest,
    ShareGPTDataset,
    SonnetDataset,
    VisionArenaDataset,
)
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json

# 毫秒到秒的转换常数
MILLISECONDS_TO_SECONDS_CONVERSION = 1000


@dataclass
class BenchmarkMetrics:
    """
    基准测试指标数据结构
    
    包含了基准测试过程中收集的所有性能指标和统计数据。
    这些指标用于全面评估推理服务的性能表现。
    """
    completed: int                                    # 成功完成的请求数量
    total_input: int                                  # 输入token总数
    total_output: int                                 # 输出token总数
    request_throughput: float                         # 请求吞吐量（请求/秒）
    request_goodput: float                            # 有效请求吞吐量（满足SLA的请求/秒）
    output_throughput: float                          # 输出token吞吐量（token/秒）
    total_token_throughput: float                     # 总token吞吐量（输入+输出token/秒）
    mean_ttft_ms: float                              # TTFT平均值（毫秒）
    median_ttft_ms: float                            # TTFT中位数（毫秒）
    std_ttft_ms: float                               # TTFT标准差（毫秒）
    percentiles_ttft_ms: list[tuple[float, float]]   # TTFT百分位数列表（毫秒）
    mean_tpot_ms: float                              # TPOT平均值（毫秒）
    median_tpot_ms: float                            # TPOT中位数（毫秒）
    std_tpot_ms: float                               # TPOT标准差（毫秒）
    percentiles_tpot_ms: list[tuple[float, float]]   # TPOT百分位数列表（毫秒）
    mean_itl_ms: float                               # ITL平均值（毫秒）
    median_itl_ms: float                             # ITL中位数（毫秒）
    std_itl_ms: float                                # ITL标准差（毫秒）
    percentiles_itl_ms: list[tuple[float, float]]    # ITL百分位数列表（毫秒）
    # E2EL代表端到端延迟（End-to-End Latency）
    # 从客户端发送请求到接收完整响应的总时间
    mean_e2el_ms: float                              # E2EL平均值（毫秒）
    median_e2el_ms: float                            # E2EL中位数（毫秒）
    std_e2el_ms: float                               # E2EL标准差（毫秒）
    percentiles_e2el_ms: list[tuple[float, float]]   # E2EL百分位数列表（毫秒）


async def get_request(
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[SampleRequest, None]:
    """
    异步请求生成器 - 按指定速率和突发性模式生成测试请求

    这个函数是压测的核心组件，控制请求的发送时机和模式，模拟真实的用户请求场景。
    它支持两种主要的请求发送模式：批量模式和流量控制模式。

    参数说明:
        input_requests: list[SampleRequest]
            待发送的请求列表，每个请求包含prompt、长度等信息
        request_rate: float
            请求发送速率 (请求/秒)
            - 如果为 inf，则立即发送所有请求 (批量模式)
            - 如果为有限值，则按指定速率发送 (流量控制模式)
        burstiness: float, 可选参数，默认1.0
            请求突发性因子，控制请求到达的时间分布模式
            - 仅在 request_rate 不为 inf 时生效
            - 默认值1.0: 遵循泊松过程 (Poisson process)，请求间隔呈指数分布
            - 其他值: 请求间隔遵循伽马分布 (Gamma distribution)
            - 0 < burstiness < 1: 更突发的请求模式 (请求更集中)
            - burstiness > 1: 更均匀的请求到达模式 (请求更分散)

    返回:
        AsyncGenerator[SampleRequest, None]: 异步生成器，按时序产生请求对象

    应用场景:
        - 批量测试: request_rate=inf, 测试系统最大处理能力
        - 流量模拟: request_rate=有限值, 模拟真实用户访问模式
        - 突发测试: 调整burstiness, 测试系统对流量波动的适应性

    实现原理:
        使用numpy的伽马分布生成请求间隔时间，当burstiness=1时退化为指数分布，
        这样可以更真实地模拟用户请求的随机性和突发性特征。

    input_requests: Iterable[SampleRequest] = iter(input_requests)

    # 计算尺度参数theta以维持期望的请求速率
    assert burstiness > 0, (
        f"期望正的突发性因子，但给定的值为 {burstiness}。"
    )
    theta = 1.0 / (request_rate * burstiness)

    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # 如果请求速率为无穷大，则无需等待
            continue

        # 从伽马分布中采样请求间隔
        # 如果burstiness为1，则遵循指数分布
        interval = np.random.gamma(shape=burstiness, scale=theta)
        # 下一个请求将在间隔后发送
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
) -> tuple[BenchmarkMetrics, list[int]]:
    """
    计算基准测试性能指标
    
    根据输入请求和输出结果计算详细的性能指标，包括延迟、吞吐量、
    百分位数等统计数据。这些指标用于全面评估推理服务的性能表现。
    
    Args:
        input_requests: 输入请求列表
        outputs: 输出结果列表，与输入请求一一对应
        dur_s: 基准测试总持续时间（秒）
        tokenizer: 用于token计算的分词器
        selected_percentile_metrics: 需要计算百分位数的指标列表
        selected_percentiles: 需要计算的百分位数列表
        goodput_config_dict: 有效吞吐量配置字典（SLA阈值）
        
    Returns:
        tuple[BenchmarkMetrics, list[int]]: 
            - BenchmarkMetrics: 包含所有性能指标的数据结构
            - list[int]: 每个请求的实际输出token长度列表
            
    Note:
        - 计算TTFT(首个token时间)、TPOT(每token时间)、ITL(token间延迟)等指标
        - 支持goodput计算，即满足SLA要求的有效吞吐量
        - 自动处理失败的请求，确保统计数据的准确性
    """
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens

            if not output_len:
                # We use the tokenizer to count the number of output tokens
                # for some serving backends instead of looking at
                # len(outputs[i].itl) since multiple output tokens may be
                # bundled together
                # Note : this may inflate the output token count slightly
                output_len = len(
                    tokenizer(
                        outputs[i].generated_text, add_special_tokens=False
                    ).input_ids
                )
            actual_output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            tpot = 0
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if goodput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(
                goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(
                goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(
                goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[
            (p, np.percentile(ttfts or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[
            (p, np.percentile(tpots or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[
            (p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles
        ],
    )

    return metrics, actual_output_lens


async def benchmark(
    backend: str,                                    # 推理后端类型 (vllm/tgi/openai等)
    api_url: str,                                    # API服务完整URL地址
    base_url: str,                                   # 服务基础URL (用于profile等功能)
    model_id: str,                                   # 模型标识符
    model_name: str,                                 # 模型显示名称
    tokenizer: PreTrainedTokenizerBase,              # 分词器实例
    input_requests: list[SampleRequest],             # 测试请求列表
    logprobs: Optional[int],                         # 返回的对数概率数量
    request_rate: float,                             # 请求发送速率 (req/s)
    burstiness: float,                               # 请求突发性因子
    disable_tqdm: bool,                              # 是否禁用进度条
    profile: bool,                                   # 是否启用性能分析
    selected_percentile_metrics: list[str],          # 需要计算百分位数的指标
    selected_percentiles: list[float],               # 需要计算的百分位数列表
    ignore_eos: bool,                                # 是否忽略EOS token
    goodput_config_dict: dict[str, float],           # 有效吞吐量SLA配置
    max_concurrency: Optional[int],                  # 最大并发请求数
    lora_modules: Optional[Iterable[str]],           # LoRA模块列表
    extra_body: Optional[dict],                      # 额外的请求参数
):
    """
    执行基准测试的核心异步函数

    这是整个基准测试框架的核心函数，负责协调所有测试组件，执行完整的性能测试流程。

    主要功能：
    1. 初始化测试环境和验证连接
    2. 配置请求生成器和并发控制
    3. 执行并发请求测试
    4. 收集和计算性能指标
    5. 生成详细的测试报告

    测试流程：
    1. 预热测试：发送单个测试请求验证连接
    2. 性能分析：可选启动profiler进行深度分析
    3. 并发测试：按配置的速率和并发数发送请求
    4. 指标计算：统计TTFT、TPOT、ITL、E2EL等关键指标
    5. 结果输出：生成格式化的测试报告

    Args:
        backend: 推理后端类型，支持vllm、tgi、openai等
        api_url: API服务的完整URL地址
        base_url: 服务的基础URL，用于profile等管理功能
        model_id: 模型的唯一标识符
        model_name: 模型的显示名称（可与model_id不同）
        tokenizer: 用于token计算的分词器实例
        input_requests: 包含所有测试请求的列表
        logprobs: 每个token返回的对数概率数量（可选）
        request_rate: 请求发送速率，单位为请求/秒，inf表示批量发送
        burstiness: 请求到达的突发性因子，1.0为泊松过程
        disable_tqdm: 是否禁用进度条显示
        profile: 是否启用Torch Profiler进行性能分析
        selected_percentile_metrics: 需要计算百分位数的指标列表
        selected_percentiles: 需要计算的百分位数列表（如[50, 90, 99]）
        ignore_eos: 是否在请求中设置ignore_eos标志
        goodput_config_dict: 有效吞吐量的SLA阈值配置
        max_concurrency: 最大并发请求数限制
        lora_modules: LoRA适配器模块列表（可选）
        extra_body: 额外的请求体参数（如采样参数）

    Returns:
        dict: 包含所有性能指标和测试结果的字典

    Raises:
        ValueError: 当初始测试失败或配置无效时

    Note:
        - 函数会自动进行预热测试以验证配置正确性
        - 支持可选的性能分析模式，需要服务端配置VLLM_TORCH_PROFILER_DIR
        - 并发控制通过信号量实现，避免过载服务器
        - 所有时间指标以秒为单位收集，最终转换为毫秒显示
    """
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_output_len, test_mm_content = (
        input_requests[0].prompt,
        input_requests[0].prompt_len,
        input_requests[0].expected_output_len,
        input_requests[0].multi_modal_data,
    )

    assert test_mm_content is None or isinstance(test_mm_content, dict)
    test_input = RequestFuncInput(
        model=model_id,
        model_name=model_name,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        logprobs=logprobs,
        multi_modal_content=test_mm_content,
        ignore_eos=ignore_eos,
        extra_body=extra_body,
    )

    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
    else:
        print("Initial test run completed. Starting main benchmark run...")

    if lora_modules:
        # For each input request, choose a LoRA module at random.
        lora_modules = iter(
            [random.choice(lora_modules) for _ in range(len(input_requests))]
        )

    if profile:
        print("Starting profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=test_prompt,
            api_url=base_url + "/start_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            multi_modal_content=test_mm_content,
            ignore_eos=ignore_eos,
            extra_body=extra_body,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler started")

    distribution = "Poisson process" if burstiness == 1.0 else "Gamma distribution"

    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    # This can be used once the minimum Python version is 3.10 or higher,
    # and it will simplify the code in limited_request_func.
    #    semaphore = (asyncio.Semaphore(max_concurrency)
    #                 if max_concurrency else contextlib.nullcontext())
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate, burstiness):
        prompt, prompt_len, output_len, mm_content = (
            request.prompt,
            request.prompt_len,
            request.expected_output_len,
            request.multi_modal_data,
        )
        req_model_id, req_model_name = model_id, model_name
        if lora_modules:
            req_lora_module = next(lora_modules)
            req_model_id, req_model_name = req_lora_module, req_lora_module

        request_func_input = RequestFuncInput(
            model=req_model_id,
            model_name=req_model_name,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            logprobs=logprobs,
            multi_modal_content=mm_content,
            ignore_eos=ignore_eos,
            extra_body=extra_body,
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)

    if profile:
        print("Stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
        goodput_config_dict=goodput_config_dict,
    )

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    if goodput_config_dict:
        print(
            "{:<40} {:<10.2f}".format(
                "Request goodput (req/s):", metrics.request_goodput
            )
        )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total Token throughput (tok/s):", metrics.total_token_throughput
        )
    )

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput:": metrics.request_goodput if goodput_config_dict else None,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):",
                getattr(metrics, f"mean_{metric_attribute_name}_ms"),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                getattr(metrics, f"median_{metric_attribute_name}_ms"),
            )
        )
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms"
        )
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms"
        )
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms"
        )
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result


def check_goodput_args(args):
    # Check and parse goodput arguments
    goodput_config_dict = {}
    VALID_NAMES = ["ttft", "tpot", "e2el"]
    if args.goodput:
        goodput_config_dict = parse_goodput(args.goodput)
        for slo_name, slo_val in goodput_config_dict.items():
            if slo_name not in VALID_NAMES:
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{str(VALID_NAMES)}. "
                )
            if slo_val < 0:
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be "
                    "non-negative."
                )
    return goodput_config_dict


def parse_goodput(slo_pairs):
    goodput_config_dict = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            goodput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format found for service level objectives. "
            'Specify service level objectives for goodput as "KEY:VALUE" '
            "pairs, where the key is a metric name, and the value is a "
            "number in milliseconds."
        ) from err
    return goodput_config_dict


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any], file_name: str
) -> None:
    metrics = [
        "median_ttft_ms",
        "mean_ttft_ms",
        "std_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "std_tpot_ms",
        "p99_tpot_ms",
        "median_itl_ms",
        "mean_itl_ms",
        "std_itl_ms",
        "p99_itl_ms",
    ]
    # These raw data might be useful, but they are rather big. They can be added
    # later if needed
    ignored_metrics = ["ttfts", "itls", "generated_texts", "errors"]
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={k: [results[k]] for k in metrics},
        extra_info={
            k: results[k]
            for k in results
            if k not in metrics and k not in ignored_metrics
        },
    )
    if pt_records:
        # Don't use json suffix here as we don't want CI to pick it up
        pt_file = f"{os.path.splitext(file_name)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    model_name = args.served_model_name
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer_mode = args.tokenizer_mode

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    tokenizer = get_tokenizer(
        tokenizer_id,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )

    if args.dataset_name is None:
        raise ValueError(
            "Please specify '--dataset-name' and the corresponding "
            "'--dataset-path' if required."
        )

    if args.dataset_name == "sonnet":
        dataset = SonnetDataset(dataset_path=args.dataset_path)
        # For the "sonnet" dataset, formatting depends on the backend.
        if args.backend == "openai-chat":
            input_requests = dataset.sample(
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
                return_prompt_formatted=False,
            )
        else:
            assert tokenizer.chat_template or tokenizer.default_chat_template, (
                "Tokenizer/model must have chat template for sonnet dataset."
            )
            input_requests = dataset.sample(
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
                return_prompt_formatted=True,
            )

    elif args.dataset_name == "hf":
        # all following datasets are implemented from the
        # HuggingFaceDataset base class
        if args.dataset_path in VisionArenaDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = VisionArenaDataset
            args.hf_split = "train"
            args.hf_subset = None
        elif args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = InstructCoderDataset
            args.hf_split = "train"
        elif args.dataset_path in MTBenchDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = MTBenchDataset
            args.hf_split = "train"
        elif args.dataset_path in ConversationDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = ConversationDataset
        elif args.dataset_path in AIMODataset.SUPPORTED_DATASET_PATHS:
            dataset_class = AIMODataset
            args.hf_split = "train"
        elif args.dataset_path in NextEditPredictionDataset.SUPPORTED_DATASET_PATHS:  # noqa: E501
            dataset_class = NextEditPredictionDataset
            args.hf_split = "train"
        elif args.dataset_path in ASRDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = ASRDataset
            args.hf_split = "train"
        else:
            supported_datasets = set(
                [
                    dataset_name
                    for cls in HuggingFaceDataset.__subclasses__()
                    for dataset_name in cls.SUPPORTED_DATASET_PATHS
                ]
            )
            raise ValueError(
                f"Unsupported dataset path: {args.dataset_path}. "
                "Huggingface dataset only supports dataset_path"
                f" from one of following: {supported_datasets}. "
                "Please consider contributing if you would "
                "like to add support for additional dataset formats."
            )

        if dataset_class.IS_MULTIMODAL and backend not in [
            "openai-chat",
            "openai-audio",
        ]:
            # multi-modal benchmark is only available on OpenAI Chat backend.
            raise ValueError(
                "Multi-modal content is only supported on 'openai-chat' and "
                "'openai-audio' backend."
            )
        input_requests = dataset_class(
            dataset_path=args.dataset_path,
            dataset_subset=args.hf_subset,
            dataset_split=args.hf_split,
            random_seed=args.seed,
        ).sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.hf_output_len,
        )

    else:
        # For datasets that follow a similar structure, use a mapping.
        dataset_mapping = {
            "sharegpt": lambda: ShareGPTDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                output_len=args.sharegpt_output_len,
            ),
            "burstgpt": lambda: BurstGPTDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(tokenizer=tokenizer, num_requests=args.num_prompts),
            "random": lambda: RandomDataset(dataset_path=args.dataset_path).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                prefix_len=args.random_prefix_len,
                input_len=args.random_input_len,
                output_len=args.random_output_len,
                range_ratio=args.random_range_ratio,
            ),
        }

        try:
            input_requests = dataset_mapping[args.dataset_name]()
        except KeyError as err:
            raise ValueError(f"Unknown dataset: {args.dataset_name}") from err
    goodput_config_dict = check_goodput_args(args)

    # Collect the sampling parameters.
    sampling_params = {
        k: v
        for k, v in {
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "temperature": args.temperature,
        }.items()
        if v is not None
    }

    # Sampling parameters are only supported by openai-compatible backend.
    if sampling_params and args.backend not in OPENAI_COMPATIBLE_BACKENDS:
        raise ValueError(
            "Sampling parameters are only supported by openai-compatible backends."
        )

    if "temperature" not in sampling_params:
        sampling_params["temperature"] = 0.0  # Default to greedy decoding.

    # Avoid GC processing "static" data - reduce pause times.
    gc.collect()
    gc.freeze()

    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            model_name=model_name,
            tokenizer=tokenizer,
            input_requests=input_requests,
            logprobs=args.logprobs,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm,
            profile=args.profile,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
            ignore_eos=args.ignore_eos,
            goodput_config_dict=goodput_config_dict,
            max_concurrency=args.max_concurrency,
            lora_modules=args.lora_modules,
            extra_body=sampling_params,
        )
    )

    # Save config and results to json
    if args.save_result or args.append_result:
        result_json: dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["num_prompts"] = args.num_prompts

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )
        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf"
        )
        result_json["burstiness"] = args.burstiness
        result_json["max_concurrency"] = args.max_concurrency

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        if not args.save_detailed:
            # Remove fields with too many data points
            for field in [
                "input_lens",
                "output_lens",
                "ttfts",
                "itls",
                "generated_texts",
                "errors",
            ]:
                if field in result_json:
                    del result_json[field]

        # Save to file
        base_model_id = model_id.split("/")[-1]
        max_concurrency_str = (
            f"-concurrency{args.max_concurrency}"
            if args.max_concurrency is not None
            else ""
        )
        file_name = f"{backend}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"  # noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(
            file_name, mode="a+" if args.append_result else "w", encoding="utf-8"
        ) as outfile:
            # Append a newline.
            if args.append_result and outfile.tell() != 0:
                outfile.write("\n")
            json.dump(result_json, outfile)
        save_to_pytorch_benchmark_format(args, result_json, file_name)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput.\n在线推理服务吞吐量基准测试。"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        help="推理后端类型，如 vllm、tgi、openai 等。"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="服务器或API基础URL（如果不使用host和port）。Server or API base url if not using http host and port."
    )
    # Use 127.0.0.1 here instead of localhost to force the use of ipv4
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器主机地址，默认127.0.0.1。Server host address, default 127.0.0.1.")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口号，默认8000。Server port, default 8000.")
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API接口路径，默认/v1/completions。API endpoint."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "burstgpt", "sonnet", "random", "hf"],
        help="基准测试所用数据集名称。Name of the dataset to benchmark on."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="sharegpt/sonnet数据集路径，或HF数据集ID。Path to the sharegpt/sonnet dataset or the huggingface dataset ID if using HF dataset."
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="最大并发请求数。Maximum number of concurrent requests. 用于模拟高层组件限制并发请求数量的环境。虽然--request-rate参数控制请求发起的速率，但此参数控制实际同时执行的请求数量。当两个参数同时使用时，如果服务器处理请求的速度跟不上，实际请求速率可能低于--request-rate指定的值。"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型名称。Name of the model."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="分词器名称或路径（如不使用默认分词器）。Name or path of the tokenizer, if not using the default tokenizer."
    )
    parser.add_argument("--use-beam-search", action="store_true", help="是否启用beam search。Enable beam search.")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="要处理的提示词数量。Number of prompts to process."
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=(
            "每个token返回的logprobs数量。Number of logprobs-per-token to compute & return as part of the request. "
            "未指定时：(1) 若未启用beam search，则不计算logprobs，每个token返回一个dummy logprob；"
            "(2) 若启用beam search，则每个token计算1个logprob。"
        ),
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="每秒请求数。Number of requests per second. 若为inf，则所有请求同时发出。否则使用泊松或伽马分布模拟请求到达。"
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="请求生成的突发性因子。Burstiness factor of the request generation. 仅在request_rate非inf时生效。1为泊松过程，<1更突发，>1更均匀。"
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子。Random seed.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="是否信任huggingface远程代码。Trust remote code from huggingface."
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="禁用进度条。Specify to disable tqdm progress bar."
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="使用Torch Profiler分析性能。Use Torch Profiler. 需服务端设置VLLM_TORCH_PROFILER_DIR。"
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="保存基准测试结果到json文件。Specify to save benchmark results to a json file."
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        help="保存详细结果（每个请求的响应、错误、ttfs、tpots等）。When saving the results, whether to include per request information such as response, error, ttfs, tpots, etc."
    )
    parser.add_argument(
        "--append-result",
        action="store_true",
        help="追加基准测试结果到已有json文件。Append the benchmark result to the existing json file."
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="元数据键值对（如 --metadata version=0.3.3 tp=1），用于结果记录。Key-value pairs for metadata of this run to be saved in the result JSON file for record keeping purposes."
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="保存基准测试json结果的目录。Specify directory to save benchmark json results. 未指定时保存在当前目录。"
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="保存基准测试json结果的文件名。Specify the filename to save benchmark json results. 未指定时按默认格式保存。"
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="发送请求时设置ignore_eos标志。Set ignore_eos flag when sending the benchmark request. 注意：deepspeed_mii和tgi不支持。"
    )
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl",
        help="要报告百分位数的指标列表（逗号分隔）。Comma-separated list of selected metrics to report percentils. 可选ttft、tpot、itl、e2el。"
    )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="要报告的百分位数列表（逗号分隔）。Comma-separated list of percentiles for selected metrics. 如25,50,75。"
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help="指定goodput的服务级目标（KEY:VALUE对，单位ms）。Specify service level objectives for goodput as 'KEY:VALUE' pairs, where the key is a metric name, and the value is in milliseconds. 可选ttft、tpot、e2el。"
    )

    # group for dataset specific arguments
    sonnet_group = parser.add_argument_group("sonnet dataset options")
    sonnet_group.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help="每个请求的输入token数，仅用于sonnet数据集。Number of input tokens per request, used only for sonnet dataset."
    )
    sonnet_group.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help="每个请求的输出token数，仅用于sonnet数据集。Number of output tokens per request, used only for sonnet dataset."
    )
    sonnet_group.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help="每个请求的前缀token数，仅用于sonnet数据集。Number of prefix tokens per request, used only for sonnet dataset."
    )

    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="每个请求的输出长度，覆盖ShareGPT数据集的默认输出长度。Output length for each request. Overrides the output length from the ShareGPT dataset."
    )

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="每个请求的输入token数，仅用于随机采样。Number of input tokens per request, used only for random sampling."
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="每个请求的输出token数，仅用于随机采样。Number of output tokens per request, used only for random sampling."
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="输入/输出长度采样的范围比例，仅用于随机采样。Range ratio for sampling input/output length, used only for random sampling. 取值[0,1)，定义对称采样区间。"
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help="每个请求的固定前缀token数。Number of fixed prefix tokens before the random context in a request. 总输入长度为random-prefix-len加上随机采样长度。"
    )

    hf_group = parser.add_argument_group("hf dataset options")
    hf_group.add_argument(
        "--hf-subset", type=str, default=None, help="HF数据集的子集。Subset of the HF dataset."
    )
    hf_group.add_argument(
        "--hf-split", type=str, default=None, help="HF数据集的分割。Split of the HF dataset."
    )
    hf_group.add_argument(
        "--hf-output-len",
        type=int,
        default=None,
        help="每个请求的输出长度，覆盖HF数据集的默认输出长度。Output length for each request. Overrides the output lengths from the sampled HF dataset."
    )

    sampling_group = parser.add_argument_group("sampling parameters")
    sampling_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="top-p采样参数，仅对openai兼容后端生效。Top-p sampling parameter. Only has effect on openai-compatible backends."
    )
    sampling_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="top-k采样参数，仅对openai兼容后端生效。Top-k sampling parameter. Only has effect on openai-compatible backends."
    )
    sampling_group.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="min-p采样参数，仅对openai兼容后端生效。Min-p sampling parameter. Only has effect on openai-compatible backends."
    )
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="温度采样参数，仅对openai兼容后端生效。Temperature sampling parameter. Only has effect on openai-compatible backends. 未指定时默认为贪婪解码（temperature=0.0）。"
    )

    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default="auto",
        choices=["auto", "slow", "mistral", "custom"],
        help='分词器模式。The tokenizer mode.\n* "auto" 自动选择最快分词器。* "slow" 始终使用慢速分词器。* "mistral" 使用mistral_common分词器。* "custom" 用--tokenizer选择自定义分词器。'
    )

    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="API中使用的模型名称。The model name used in the API. 未指定时与--model一致。"
    )

    parser.add_argument(
        "--lora-modules",
        nargs="+",
        default=None,
        help="服务器启动时传入的LoRA模块子集。A subset of LoRA module names passed in when launching the server. 每个请求随机选择一个LoRA模块。"
    )

    args = parser.parse_args()

    main(args)