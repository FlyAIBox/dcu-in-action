# 文件: benchmark_serving.py
# 功能: vLLM在线推理服务吞吐量压测工具
# 许可证: Apache-2.0

r"""
vLLM在线推理服务性能压测工具

这个脚本用于测试vLLM推理服务的性能指标，包括：
- 吞吐量 (Throughput)
- 延迟 (Latency) 
- 首token时间 (Time to First Token, TTFT)
- token间延迟 (Inter-token Latency, ITL)
- 每token输出时间 (Time per Output Token, TPOT)

使用方法:

1. 服务端启动vLLM OpenAI兼容API服务器:
    vllm serve <your_model> \
        --swap-space 16 \
        --disable-log-requests

2. 客户端运行压测脚本:
    python benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \    # 默认为inf（无限制）
        --num-prompts <num_prompts>        # 默认为1000

    使用TGI后端时，需要添加:
        --endpoint /generate_stream
"""

# 导入必要的Python标准库和第三方库
import argparse                                    # 命令行参数解析
import asyncio                                     # 异步编程支持
import gc                                          # 垃圾回收控制
import json                                        # JSON数据处理
import os                                          # 操作系统接口
import random                                      # 随机数生成
import time                                        # 时间相关功能
import warnings                                    # 警告处理
from collections.abc import AsyncGenerator, Iterable  # 异步生成器和可迭代对象类型
from dataclasses import dataclass                 # 数据类装饰器
from datetime import datetime                      # 日期时间处理
from typing import Any, Optional                   # 类型提示

import numpy as np                                 # 数值计算库
from tqdm.asyncio import tqdm                      # 异步进度条
from transformers import PreTrainedTokenizerBase   # HuggingFace分词器基类

# 导入后端请求处理相关模块
from ..backends.backend_request_func import (
    ASYNC_REQUEST_FUNCS,           # 异步请求函数映射
    OPENAI_COMPATIBLE_BACKENDS,    # OpenAI兼容后端列表
    RequestFuncInput,              # 请求输入数据结构
    RequestFuncOutput,             # 请求输出数据结构
)

# 尝试从vLLM导入分词器，如果失败则从后端请求模块导入
try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from ..backends.backend_request_func import get_tokenizer

# 尝试使用vLLM的灵活参数解析器，如果失败则使用标准解析器
try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

# 导入各种数据集处理类
from ..data_processing.benchmark_dataset import (
    AIMODataset,                   # AIMO数学竞赛数据集
    ASRDataset,                    # 自动语音识别数据集
    BurstGPTDataset,               # BurstGPT数据集
    ConversationDataset,           # 对话数据集
    HuggingFaceDataset,            # HuggingFace数据集基类
    InstructCoderDataset,          # 代码指令数据集
    MTBenchDataset,                # MT-Bench多轮对话数据集
    NextEditPredictionDataset,     # 下一步编辑预测数据集
    RandomDataset,                 # 随机生成数据集
    SampleRequest,                 # 样本请求数据结构
    ShareGPTDataset,               # ShareGPT对话数据集
    SonnetDataset,                 # 诗歌数据集
    VisionArenaDataset,            # 视觉竞技场数据集
)

# 导入工具函数
from ..utils.benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json

# 毫秒到秒的转换常数
MILLISECONDS_TO_SECONDS_CONVERSION = 1000


@dataclass
class BenchmarkMetrics:
    """
    压测性能指标数据类
    
    包含了vLLM推理服务的各项关键性能指标，用于全面评估服务性能
    """
    # 基础统计指标
    completed: int                                    # 成功完成的请求数量
    total_input: int                                  # 总输入token数量
    total_output: int                                 # 总输出token数量
    
    # 吞吐量指标
    request_throughput: float                         # 请求吞吐量 (请求/秒)
    request_goodput: float                            # 请求良好吞吐量 (满足SLA的请求/秒)
    output_throughput: float                          # 输出token吞吐量 (token/秒)
    total_token_throughput: float                     # 总token吞吐量 (输入+输出token/秒)
    
    # TTFT (Time to First Token) - 首token时间指标
    mean_ttft_ms: float                               # TTFT平均值 (毫秒)
    median_ttft_ms: float                             # TTFT中位数 (毫秒)
    std_ttft_ms: float                                # TTFT标准差 (毫秒)
    percentiles_ttft_ms: list[tuple[float, float]]    # TTFT百分位数列表 [(百分位, 值)]
    
    # TPOT (Time per Output Token) - 每输出token时间指标
    mean_tpot_ms: float                               # TPOT平均值 (毫秒)
    median_tpot_ms: float                             # TPOT中位数 (毫秒)
    std_tpot_ms: float                                # TPOT标准差 (毫秒)
    percentiles_tpot_ms: list[tuple[float, float]]    # TPOT百分位数列表
    
    # ITL (Inter-token Latency) - token间延迟指标
    mean_itl_ms: float                                # ITL平均值 (毫秒)
    median_itl_ms: float                              # ITL中位数 (毫秒)
    std_itl_ms: float                                 # ITL标准差 (毫秒)
    percentiles_itl_ms: list[tuple[float, float]]     # ITL百分位数列表
    
    # E2EL (End-to-End Latency) - 端到端延迟指标
    # 表示从客户端发送请求到接收完整响应的总时间
    mean_e2el_ms: float                               # E2EL平均值 (毫秒)
    median_e2el_ms: float                             # E2EL中位数 (毫秒)
    std_e2el_ms: float                                # E2EL标准差 (毫秒)
    percentiles_e2el_ms: list[tuple[float, float]]    # E2EL百分位数列表


async def get_request(
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[SampleRequest, None]:
    """
    异步生成指定速率的请求流，支持可选的突发性控制
    
    这个函数是压测的核心调度器，控制请求的发送时机和频率。
    通过调整请求速率和突发性参数，可以模拟不同的负载模式。
    
    参数:
        input_requests: 输入请求列表，每个元素都是SampleRequest对象
        request_rate: 请求生成速率 (请求/秒)
                     - 设为float("inf")表示无限制，尽快发送所有请求
                     - 设为具体数值表示按该速率发送请求
        burstiness: 请求突发性因子 (可选，默认1.0)
                   - 仅在request_rate不为无穷大时生效
                   - 默认值1.0遵循泊松过程（指数分布间隔）
                   - 其他值遵循伽马分布间隔
                   - 较低值(0 < burstiness < 1)产生更突发的请求模式
                   - 较高值(burstiness > 1)产生更均匀的请求到达模式
    
    返回:
        AsyncGenerator[SampleRequest, None]: 异步生成器，按指定速率产生请求
    """
    # 将请求列表转换为迭代器
    input_requests: Iterable[SampleRequest] = iter(input_requests)

    # 计算伽马分布的尺度参数theta，以维持期望的请求速率
    assert burstiness > 0, (
        f"突发性因子必须为正数，但得到了 {burstiness}。"
    )
    theta = 1.0 / (request_rate * burstiness)

    # 遍历所有输入请求
    for request in input_requests:
        # 立即产生当前请求
        yield request

        # 如果请求速率为无穷大，则不需要等待，立即处理下一个请求
        if request_rate == float("inf"):
            continue

        # 从伽马分布中采样请求间隔时间
        # 当burstiness=1时，这等价于指数分布（泊松过程）
        interval = np.random.gamma(shape=burstiness, scale=theta)
        
        # 等待计算出的间隔时间后再发送下一个请求
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
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: list[SampleRequest],
    logprobs: Optional[int],
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    ignore_eos: bool,
    goodput_config_dict: dict[str, float],
    max_concurrency: Optional[int],
    lora_modules: Optional[Iterable[str]],
    extra_body: Optional[dict],
):
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
    '''
    1. 主要描述
    description: "Benchmark the online serving throughput. / 在线推理服务吞吐量压测工具"
    2. 基础参数
    --backend: 后端选择
    --base-url: 服务器或API基础URL
    --host: 服务器主机地址
    --port: 服务器端口号
    --endpoint: API端点
    --model: 模型名称
    --tokenizer: 分词器名称或路径
    --use-beam-search: 使用束搜索进行生成
    --num-prompts: 要处理的提示数量
    --seed: 用于可重现性的随机种子
    3. 性能相关参数
    --logprobs: 对数概率计算
    --request-rate: 每秒请求数
    --burstiness: 请求生成的突发性因子
    --max-concurrency: 最大并发请求数
    --trust-remote-code: 信任来自huggingface的远程代码
    --disable-tqdm: 禁用tqdm进度条
    --profile: 使用Torch Profiler
    4. 结果保存参数
    --save-result: 保存压测结果到json文件
    --save-detailed: 保存详细结果信息
    --append-result: 追加结果到现有文件
    --metadata: 元数据键值对
    --result-dir: 结果保存目录
    --result-filename: 结果文件名
    5. 性能指标参数
    --ignore-eos: 忽略EOS标志
    --percentile-metrics: 报告百分位数的指标
    --metric-percentiles: 指标的百分位数
    --goodput: 良好吞吐量的服务级别目标
    6. 数据集相关参数
    --dataset-name: 数据集名称
    --dataset-path: 数据集路径
    Sonnet数据集选项
    --sonnet-input-len: 输入token数量
    --sonnet-output-len: 输出token数量
    --sonnet-prefix-len: 前缀token数量
    ShareGPT数据集选项
    --sharegpt-output-len: 输出长度
    随机数据集选项
    --random-input-len: 输入token数量
    --random-output-len: 输出token数量
    --random-range-ratio: 范围比率
    --random-prefix-len: 前缀token数量
    HF数据集选项
    --hf-subset: HF数据集子集
    --hf-split: HF数据集分割
    --hf-output-len: 输出长度
    7. 采样参数
    --top-p: Top-p采样参数
    --top-k: Top-k采样参数
    --min-p: Min-p采样参数
    --temperature: 温度采样参数
    8. 其他参数
    --tokenizer-mode: 分词器模式
    --served-model-name: API中的模型名称
    --lora-modules: LoRA模块名称子集
    所有参数现在都提供了清晰的中英文双语说明，方便中文用户理解和使用这个vLLM在线推理服务压测工具。
    '''
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput. / 在线推理服务吞吐量压测工具"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port. / 服务器或API基础URL（当不使用http主机和端口时）",
    )
    # Use 127.0.0.1 here instead of localhost to force the use of ipv4
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1",
        help="Server host address. / 服务器主机地址"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Server port number. / 服务器端口号"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint. / API端点",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "burstgpt", "sonnet", "random", "hf"],
        help="Name of the dataset to benchmark on. / 用于压测的数据集名称",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the sharegpt/sonnet dataset. "
        "Or the huggingface dataset ID if using HF dataset. / "
        "sharegpt/sonnet数据集的路径，或使用HF数据集时的huggingface数据集ID",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up. / "
        "最大并发请求数。可用于模拟高级组件强制执行最大并发请求数的环境。"
        "虽然--request-rate参数控制请求的发起速率，但此参数将控制实际允许同时执行的请求数。"
        "这意味着当结合使用时，如果服务器处理请求的速度不够快，实际请求速率可能低于--request-rate指定的速率。",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model. / 模型名称",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer, if not using the default tokenizer. / 分词器的名称或路径（如果不使用默认分词器）",  # noqa: E501
    )
    parser.add_argument(
        "--use-beam-search", 
        action="store_true",
        help="Use beam search for generation. / 使用束搜索进行生成"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process. / 要处理的提示数量",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=(
            "Number of logprobs-per-token to compute & return as part of "
            "the request. If unspecified, then either (1) if beam search "
            "is disabled, no logprobs are computed & a single dummy "
            "logprob is returned for each token; or (2) if beam search "
            "is enabled 1 logprob per token is computed / "
            "作为请求一部分计算和返回的每个token的对数概率数量。如果未指定，则："
            "(1) 如果禁用束搜索，不计算对数概率，每个token返回单个虚拟对数概率；"
            "或 (2) 如果启用束搜索，每个token计算1个对数概率"
        ),
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process or gamma distribution "
        "to synthesize the request arrival times. / "
        "每秒请求数。如果为inf，则所有请求在时间0发送。"
        "否则，我们使用泊松过程或伽马分布来合成请求到达时间。",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation. "
        "Only take effect when request_rate is not inf. "
        "Default value is 1, which follows Poisson process. "
        "Otherwise, the request intervals follow a gamma distribution. "
        "A lower burstiness value (0 < burstiness < 1) results in more "
        "bursty requests. A higher burstiness value (burstiness > 1) "
        "results in a more uniform arrival of requests. / "
        "请求生成的突发性因子。仅在request_rate不为inf时生效。"
        "默认值为1，遵循泊松过程。否则，请求间隔遵循伽马分布。"
        "较低的突发性值(0 < burstiness < 1)产生更突发的请求。"
        "较高的突发性值(burstiness > 1)产生更均匀的请求到达。",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0,
        help="Random seed for reproducibility. / 用于可重现性的随机种子"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface / 信任来自huggingface的远程代码",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar. / 指定禁用tqdm进度条",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler. / "
        "使用Torch Profiler。端点必须使用VLLM_TORCH_PROFILER_DIR启动才能启用分析器。",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file / 指定将压测结果保存到json文件",
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        help="When saving the results, whether to include per request "
        "information such as response, error, ttfs, tpots, etc. / "
        "保存结果时，是否包含每个请求的信息，如响应、错误、ttfs、tpots等",
    )
    parser.add_argument(
        "--append-result",
        action="store_true",
        help="Append the benchmark result to the existing json file. / 将压测结果追加到现有的json文件",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes. / "
        "键值对（例如，--metadata version=0.3.3 tp=1），用于保存此次运行的元数据到结果JSON文件中以供记录保存。",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory. / "
        "指定保存压测json结果的目录。如果未指定，结果保存在当前目录。",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format. / "
        "指定保存压测json结果的文件名。如果未指定，结果将以"
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json格式保存。",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi. / "
        "发送压测请求时设置ignore_eos标志。警告：deepspeed_mii和tgi不支持ignore_eos。",
    )
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl",
        help="Comma-separated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        'Allowed metric names are "ttft", "tpot", "itl", "e2el". '
        'Default value is "ttft,tpot,itl". / '
        '报告百分位数的选定指标的逗号分隔列表。此参数指定要报告百分位数的指标。'
        '允许的指标名称有"ttft"、"tpot"、"itl"、"e2el"。默认值为"ttft,tpot,itl"。',
    )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-separated list of percentiles for selected metrics. "
        'To report 25-th, 50-th, and 75-th percentiles, use "25,50,75". '
        'Default value is "99". '
        'Use "--percentile-metrics" to select metrics. / '
        '选定指标的百分位数逗号分隔列表。要报告第25、50和75百分位数，使用"25,50,75"。'
        '默认值为"99"。使用"--percentile-metrics"选择指标。',
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help='Specify service level objectives for goodput as "KEY:VALUE" '
        "pairs, where the key is a metric name, and the value is in "
        'milliseconds. Multiple "KEY:VALUE" pairs can be provided, '
        "separated by spaces. Allowed request level metric names are "
        '"ttft", "tpot", "e2el". For more context on the definition of '
        "goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 "
        "and the blog: https://hao-ai-lab.github.io/blogs/distserve / "
        '将良好吞吐量的服务级别目标指定为"KEY:VALUE"对，其中键是指标名称，值以毫秒为单位。'
        '可以提供多个"KEY:VALUE"对，用空格分隔。允许的请求级别指标名称有"ttft"、"tpot"、"e2el"。'
        '有关良好吞吐量定义的更多上下文，请参阅DistServe论文：https://arxiv.org/pdf/2401.09670 '
        '和博客：https://hao-ai-lab.github.io/blogs/distserve',
    )

    # group for dataset specific arguments
    sonnet_group = parser.add_argument_group("sonnet dataset options / sonnet数据集选项")
    sonnet_group.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help="Number of input tokens per request, used only for sonnet dataset. / 每个请求的输入token数量，仅用于sonnet数据集",
    )
    sonnet_group.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help="Number of output tokens per request, used only for sonnet dataset. / 每个请求的输出token数量，仅用于sonnet数据集",
    )
    sonnet_group.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help="Number of prefix tokens per request, used only for sonnet dataset. / 每个请求的前缀token数量，仅用于sonnet数据集",
    )

    sharegpt_group = parser.add_argument_group("sharegpt dataset options / sharegpt数据集选项")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset. / 每个请求的输出长度。覆盖ShareGPT数据集中的输出长度",
    )

    random_group = parser.add_argument_group("random dataset options / 随机数据集选项")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random sampling. / 每个请求的输入token数量，仅用于随机采样",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for random sampling. / 每个请求的输出token数量，仅用于随机采样",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range ratio for sampling input/output length, "
        "used only for random sampling. Must be in the range [0, 1) to define "
        "a symmetric sampling range"
        "[length * (1 - range_ratio), length * (1 + range_ratio)]. / "
        "采样输入/输出长度的范围比率，仅用于随机采样。必须在[0, 1)范围内以定义对称采样范围"
        "[length * (1 - range_ratio), length * (1 + range_ratio)]。",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help=(
            "Number of fixed prefix tokens before the random context "
            "in a request. "
            "The total input length is the sum of `random-prefix-len` and "
            "a random "
            "context length sampled from [input_len * (1 - range_ratio), "
            "input_len * (1 + range_ratio)]. / "
            "请求中随机上下文之前的固定前缀token数量。"
            "总输入长度是`random-prefix-len`和从[input_len * (1 - range_ratio), "
            "input_len * (1 + range_ratio)]采样的随机上下文长度的总和。"
        ),
    )

    hf_group = parser.add_argument_group("hf dataset options / hf数据集选项")
    hf_group.add_argument(
        "--hf-subset", type=str, default=None, help="Subset of the HF dataset. / HF数据集的子集"
    )
    hf_group.add_argument(
        "--hf-split", type=str, default=None, help="Split of the HF dataset. / HF数据集的分割"
    )
    hf_group.add_argument(
        "--hf-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output lengths "
        "from the sampled HF dataset. / 每个请求的输出长度。覆盖采样的HF数据集中的输出长度",
    )

    sampling_group = parser.add_argument_group("sampling parameters / 采样参数")
    sampling_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p sampling parameter. Only has effect on openai-compatible backends. / Top-p采样参数。仅在openai兼容后端上有效",
    )
    sampling_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling parameter. Only has effect on openai-compatible backends. / Top-k采样参数。仅在openai兼容后端上有效",
    )
    sampling_group.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Min-p sampling parameter. Only has effect on openai-compatible backends. / Min-p采样参数。仅在openai兼容后端上有效",
    )
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature sampling parameter. Only has effect on "
        "openai-compatible backends. If not specified, default to greedy "
        "decoding (i.e. temperature==0.0). / "
        "温度采样参数。仅在openai兼容后端上有效。如果未指定，默认为贪婪解码（即temperature==0.0）。",
    )

    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default="auto",
        choices=["auto", "slow", "mistral", "custom"],
        help='The tokenizer mode.\n\n* "auto" will use the '
        'fast tokenizer if available.\n* "slow" will '
        "always use the slow tokenizer. \n* "
        '"mistral" will always use the `mistral_common` tokenizer. \n*'
        '"custom" will use --tokenizer to select the preregistered tokenizer. / '
        '分词器模式。\n\n* "auto"将使用快速分词器（如果可用）\n* "slow"将始终使用慢速分词器\n* '
        '"mistral"将始终使用`mistral_common`分词器\n* "custom"将使用--tokenizer选择预注册的分词器',
    )

    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. "
        "If not specified, the model name will be the "
        "same as the ``--model`` argument. / "
        "API中使用的模型名称。如果未指定，模型名称将与``--model``参数相同。",
    )

    parser.add_argument(
        "--lora-modules",
        nargs="+",
        default=None,
        help="A subset of LoRA module names passed in when "
        "launching the server. For each request, the "
        "script chooses a LoRA module at random. / "
        "启动服务器时传入的LoRA模块名称子集。对于每个请求，脚本随机选择一个LoRA模块。",
    )

    args = parser.parse_args()

    main(args)