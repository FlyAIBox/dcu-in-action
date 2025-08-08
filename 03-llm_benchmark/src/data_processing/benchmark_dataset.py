# 文件: benchmark_dataset.py
# 功能: vLLM压测数据集处理框架
# 许可证: Apache-2.0

"""
vLLM压测数据集处理模块

本模块定义了从各种数据集中采样压测请求的框架。每个BenchmarkDataset的子类
都必须实现样本生成功能。支持的数据集类型包括：

数据集类型:
  - ShareGPT: 对话数据集，来自ShareGPT项目
  - Random: 合成随机数据集，用于基础性能测试
  - Sonnet: 诗歌数据集，用于文学文本生成测试
  - BurstGPT: BurstGPT数据集，用于突发负载测试
  - HuggingFace: HuggingFace Hub上的各种数据集
  - VisionArena: 视觉竞技场数据集，支持多模态输入
  - ConversationDataset: 通用对话数据集
  - InstructCoderDataset: 代码指令数据集
  - MTBenchDataset: MT-Bench多轮对话评测数据集
  - AIMODataset: AIMO数学竞赛数据集
  - ASRDataset: 自动语音识别数据集
  - NextEditPredictionDataset: 下一步编辑预测数据集

TODO: 实现CustomDataset来解析JSON文件并将其内容转换为SampleRequest实例，
      类似于ShareGPT中使用的方法。
"""

import base64
import io
import json
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from io import BytesIO
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image
from transformers import PreTrainedTokenizerBase

from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.multimodal import MultiModalDataDict
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_lora_tokenizer

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class SampleRequest:
    """
    表示单个推理请求的数据结构
    
    这个类封装了压测中单个请求的所有必要信息，包括输入提示、
    长度信息、多模态数据和LoRA配置等。
    """

    prompt: Union[str, Any]                                      # 输入提示文本或其他格式
    prompt_len: int                                              # 提示文本的token长度
    expected_output_len: int                                     # 期望的输出token长度
    multi_modal_data: Optional[Union[MultiModalDataDict, dict]] = None  # 多模态数据（图像、音频等）
    lora_request: Optional[LoRARequest] = None                   # LoRA适配器请求配置


# -----------------------------------------------------------------------------
# Benchmark Dataset Base Class
# -----------------------------------------------------------------------------


class BenchmarkDataset(ABC):
    """
    压测数据集基类
    
    所有压测数据集的抽象基类，定义了数据集处理的通用接口和方法。
    子类必须实现load_data()和sample()方法。
    """
    DEFAULT_SEED = 0          # 默认随机种子
    IS_MULTIMODAL = False     # 是否为多模态数据集

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        random_seed: int = DEFAULT_SEED,
    ) -> None:
        """
        初始化压测数据集
        
        参数:
            dataset_path: 数据集路径（可选）
                         如果为None，表示可能使用默认或随机数据集
            random_seed: 随机种子，用于可重现的洗牌或采样
                        默认为DEFAULT_SEED
        """
        self.dataset_path = dataset_path
        # 设置随机种子，确保None值被替换为默认种子
        self.random_seed = random_seed if random_seed is not None else self.DEFAULT_SEED
        self.data = None          # 数据集内容，由子类的load_data()方法填充

    def apply_multimodal_chat_transformation(
        self, prompt: str, mm_content: Optional[MultiModalDataDict] = None
    ) -> list[dict]:
        """
        Transform a prompt and optional multimodal content into a chat format.
        This method is used for chat models that expect a specific conversation
        format.
        """
        content = [{"text": prompt, "type": "text"}]
        if mm_content is not None:
            content.append(mm_content)
        return [{"role": "user", "content": content}]

    def load_data(self) -> None:
        """
        Load data from the dataset path into self.data.

        This method must be overridden by subclasses since the method to load
        data will vary depending on the dataset format and source.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        # TODO (jenniferzhao): add support for downloading data
        raise NotImplementedError("load_data must be implemented in subclasses.")

    def get_random_lora_request(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_loras: Optional[int] = None,
        lora_path: Optional[str] = None,
    ) -> tuple[Optional[LoRARequest], AnyTokenizer]:
        """
        Optionally select a random LoRA request and return its associated
        tokenizer.

        This method is used when LoRA parameters are provided.  It randomly
        selects a LoRA based on max_loras and retrieves a cached tokenizer for
        that LoRA if available. Otherwise, it returns the base tokenizer.

        Args:
            tokenizer (PreTrainedTokenizerBase): The base tokenizer to use if no
            LoRA is selected.  max_loras (Optional[int]): The maximum number of
            LoRAs available. If None, LoRA is not used.  lora_path
            (Optional[str]): Path to the LoRA parameters on disk. If None, LoRA
            is not used.

        Returns:
            tuple[Optional[LoRARequest], AnyTokenizer]: A tuple where the first
            element is a LoRARequest (or None if not applicable) and the second
            element is the tokenizer associated with the LoRA request (or the
            base tokenizer).
        """
        if max_loras is None or lora_path is None:
            return None, tokenizer

        # Generate a random LoRA ID in the range [1, max_loras].
        lora_id = random.randint(1, max_loras)
        lora_request = LoRARequest(
            lora_name=str(lora_id),
            lora_int_id=lora_id,
            lora_path=lora_path_on_disk(lora_path),
        )
        if lora_id not in lora_tokenizer_cache:
            lora_tokenizer_cache[lora_id] = get_lora_tokenizer(lora_request)
        # Return lora_request and the cached tokenizer if available; otherwise,
        # return the base tokenizer
        return lora_request, lora_tokenizer_cache[lora_id] or tokenizer

    @abstractmethod
    def sample(
        self, tokenizer: PreTrainedTokenizerBase, num_requests: int
    ) -> list[SampleRequest]:
        """
        Abstract method to generate sample requests from the dataset.

        Subclasses must override this method to implement dataset-specific logic
        for generating a list of SampleRequest objects.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to be used
             for processing the dataset's text.
            num_requests (int): The number of sample requests to generate.

        Returns:
            list[SampleRequest]: A list of sample requests generated from the
            dataset.
        """
        raise NotImplementedError("sample must be implemented in subclasses.")

    def maybe_oversample_requests(
        self, requests: list[SampleRequest], num_requests: int
    ) -> None:
        """
        Oversamples the list of requests if its size is less than the desired
        number.

        Args:
            requests (List[SampleRequest]): The current list of sampled
            requests.  num_requests (int): The target number of requests.
        """
        if len(requests) < num_requests:
            random.seed(self.random_seed)
            additional = random.choices(requests, k=num_requests - len(requests))
            requests.extend(additional)
            logger.info("Oversampled requests to reach %d total samples.", num_requests)


# -----------------------------------------------------------------------------
# Utility Functions and Global Caches
# -----------------------------------------------------------------------------


def is_valid_sequence(
    prompt_len: int,
    output_len: int,
    min_len: int = 4,
    max_prompt_len: int = 1024,
    max_total_len: int = 2048,
    skip_min_output_len_check: bool = False,
) -> bool:
    """
    Validate a sequence based on prompt and output lengths.

    Default pruning criteria are copied from the original `sample_hf_requests`
    and `sample_sharegpt_requests` functions in benchmark_serving.py, as well as
    from `sample_requests` in benchmark_throughput.py.
    """
    # Check for invalid conditions
    prompt_too_short = prompt_len < min_len
    output_too_short = (not skip_min_output_len_check) and (output_len < min_len)
    prompt_too_long = prompt_len > max_prompt_len
    combined_too_long = (prompt_len + output_len) > max_total_len

    # Return True if none of the invalid conditions are met
    return not (
        prompt_too_short or output_too_short or prompt_too_long or combined_too_long
    )


@cache
def lora_path_on_disk(lora_path: str) -> str:
    return get_adapter_absolute_path(lora_path)


# Global cache for LoRA tokenizers.
lora_tokenizer_cache: dict[int, AnyTokenizer] = {}


def process_image(image: Any) -> Mapping[str, Any]:
    """
    Process a single image input and return a multimedia content dictionary.

    Supports three input types:

    1. Dictionary with raw image bytes: - Expects a dict with a 'bytes' key
       containing raw image data.  - Loads the bytes as a PIL.Image.Image.

    2. PIL.Image.Image input: - Converts the image to RGB.  - Saves the image as
       a JPEG in memory.  - Encodes the JPEG data as a base64 string.  - Returns
       a dictionary with the image as a base64 data URL.

    3. String input: - Treats the string as a URL or local file path.  -
       Prepends "file://" if the string doesn't start with "http://" or
       "file://".  - Returns a dictionary with the image URL.

    Raises:
        ValueError: If the input is not a supported type.
    """
    if isinstance(image, dict) and "bytes" in image:
        image = Image.open(BytesIO(image["bytes"]))
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        with io.BytesIO() as image_data:
            image.save(image_data, format="JPEG")
            image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
        }

    if isinstance(image, str):
        image_url = (
            image if image.startswith(("http://", "file://")) else f"file://{image}"
        )
        return {"type": "image_url", "image_url": {"url": image_url}}

    raise ValueError(
        f"Invalid image input {image}. Must be a PIL.Image.Image"
        " or str or dictionary with raw image bytes."
    )


# -----------------------------------------------------------------------------
# Random Dataset Implementation (Synthetic Data)
# -----------------------------------------------------------------------------


class RandomDataset(BenchmarkDataset):
    """
    随机数据集实现类
    
    生成合成的随机token序列作为压测数据。这种数据集主要用于：
    1. 基础性能测试，不受具体文本内容影响
    2. 控制输入输出长度的精确测试
    3. 大规模压测时避免真实数据集的版权问题
    """
    
    # 默认参数值（从benchmark_serving.py复制）
    DEFAULT_PREFIX_LEN = 0      # 默认前缀长度
    DEFAULT_RANGE_RATIO = 0.0   # 默认长度变化范围比例
    DEFAULT_INPUT_LEN = 1024    # 默认输入长度
    DEFAULT_OUTPUT_LEN = 128    # 默认输出长度

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        range_ratio: float = DEFAULT_RANGE_RATIO,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        **kwargs,
    ) -> list[SampleRequest]:
        """
        生成随机样本请求
        
        参数:
            tokenizer: 分词器，用于生成和解码token序列
            num_requests: 要生成的请求数量
            prefix_len: 前缀token长度
            range_ratio: 长度变化范围比例，用于生成不同长度的请求
            input_len: 基础输入长度
            output_len: 基础输出长度
            
        返回:
            list[SampleRequest]: 生成的样本请求列表
        """
        # 确保range_ratio小于1，以保证有效的采样范围
        assert range_ratio < 1.0, (
            "random_range_ratio必须小于1.0以确保有效的采样范围"
        )

        vocab_size = tokenizer.vocab_size                    # 词汇表大小
        num_special_tokens = tokenizer.num_special_tokens_to_add()  # 特殊token数量
        real_input_len = input_len - num_special_tokens      # 实际输入长度（减去特殊token）

        # 生成前缀token序列（如果需要）
        prefix_token_ids = (
            np.random.randint(0, vocab_size, size=prefix_len).tolist()
            if prefix_len > 0
            else []
        )

        # 新的采样逻辑：在[X * (1 - b), X * (1 + b)]范围内采样
        input_low = int(real_input_len * (1 - range_ratio))   # 输入长度下限
        input_high = int(real_input_len * (1 + range_ratio))  # 输入长度上限
        output_low = int(output_len * (1 - range_ratio))      # 输出长度下限
        output_high = int(output_len * (1 + range_ratio))     # 输出长度上限

        # 记录采样范围用于调试
        logger.info("从[%s, %s]范围采样input_len", input_low, input_high)
        logger.info("从[%s, %s]范围采样output_len", output_low, output_high)

        # 为每个请求随机生成长度
        input_lens = np.random.randint(input_low, input_high + 1, size=num_requests)
        output_lens = np.random.randint(output_low, output_high + 1, size=num_requests)
        offsets = np.random.randint(0, vocab_size, size=num_requests)  # 随机偏移量

        requests = []
        for i in range(num_requests):
            # 生成内部token序列：使用偏移量和索引确保每个请求的唯一性
            inner_seq = (
                (offsets[i] + i + np.arange(input_lens[i])) % vocab_size
            ).tolist()
            
            # 组合前缀和内部序列
            token_sequence = prefix_token_ids + inner_seq
            prompt = tokenizer.decode(token_sequence)
            
            # 重要：解码后需要重新编码再解码
            # 这是因为在某些情况下，N个连续token解码后的字符串
            # 重新分词可能不等于N个token
            # 例如GPT2Tokenizer：
            # [6880, 6881] -> ['Ġcalls', 'here'] ->
            # [1650, 939, 486] -> ['Ġcall', 'sh', 'ere']
            # 为避免提示长度的不受控变化，在重新解码前截断编码序列
            re_encoded_sequence = tokenizer.encode(prompt, add_special_tokens=False)[
                : input_lens[i]
            ]
            prompt = tokenizer.decode(re_encoded_sequence)
            
            # 计算总输入长度
            total_input_len = prefix_len + int(input_lens[i])
            
            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=total_input_len,
                    expected_output_len=int(output_lens[i]),
                )
            )
        return requests


# -----------------------------------------------------------------------------
# ShareGPT Dataset Implementation
# -----------------------------------------------------------------------------


class ShareGPTDataset(BenchmarkDataset):
    """
    Implements the ShareGPT dataset.  Loads data from a JSON file and generates
    sample requests based on conversation turns.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = json.load(f)
        # Filter entries with at least two conversation turns.
        self.data = [
            entry
            for entry in self.data
            if "conversations" in entry and len(entry["conversations"]) >= 2
        ]
        random.seed(self.random_seed)
        random.shuffle(self.data)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        lora_path: Optional[str] = None,
        max_loras: Optional[int] = None,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list:
        samples: list = []
        for entry in self.data:
            if len(samples) >= num_requests:
                break
            prompt, completion = (
                entry["conversations"][0]["value"],
                entry["conversations"][1]["value"],
            )

            lora_request, tokenizer = self.get_random_lora_request(
                tokenizer=tokenizer, max_loras=max_loras, lora_path=lora_path
            )
            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            new_output_len = len(completion_ids) if output_len is None else output_len
            if not is_valid_sequence(
                prompt_len,
                new_output_len,
                skip_min_output_len_check=output_len is not None,
            ):
                continue
            if enable_multimodal_chat:
                prompt = self.apply_multimodal_chat_transformation(prompt, None)
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=new_output_len,
                    lora_request=lora_request,
                )
            )
        self.maybe_oversample_requests(samples, num_requests)
        return samples


# -----------------------------------------------------------------------------
# Sonnet Dataset Implementation
# -----------------------------------------------------------------------------


class SonnetDataset(BenchmarkDataset):
    """
    Simplified implementation of the Sonnet dataset.  Loads poem lines from a
    text file and generates sample requests.  Default values here copied from
    `benchmark_serving.py` for the sonnet dataset.
    """

    DEFAULT_PREFIX_LEN = 200
    DEFAULT_INPUT_LEN = 550
    DEFAULT_OUTPUT_LEN = 150

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        if not self.dataset_path:
            raise ValueError("dataset_path must be provided.")
        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = f.readlines()

    def sample(
        self,
        tokenizer,
        num_requests: int,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        return_prompt_formatted: bool = False,
        **kwargs,
    ) -> list:
        # Calculate average token length for a poem line.
        tokenized_lines = [tokenizer(line).input_ids for line in self.data]
        avg_len = sum(len(tokens) for tokens in tokenized_lines) / len(tokenized_lines)

        # Build the base prompt.
        base_prompt = "Pick as many lines as you can from these poem lines:\n"
        base_msg = [{"role": "user", "content": base_prompt}]
        base_fmt = tokenizer.apply_chat_template(
            base_msg, add_generation_prompt=True, tokenize=False
        )
        base_offset = len(tokenizer(base_fmt).input_ids)
        if input_len <= base_offset:
            raise ValueError(
                f"'input_len' must be higher than the base prompt length "
                f"({base_offset})."
            )

        # Determine how many poem lines to use.
        num_input_lines = round((input_len - base_offset) / avg_len)
        num_prefix_lines = max(round((prefix_len - base_offset) / avg_len), 0)
        prefix_lines = self.data[:num_prefix_lines]

        samples = []
        while len(samples) < num_requests:
            extra_lines = random.choices(
                self.data, k=num_input_lines - num_prefix_lines
            )
            prompt = f"{base_prompt}{''.join(prefix_lines + extra_lines)}"
            msg = [{"role": "user", "content": prompt}]
            prompt_formatted = tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False
            )
            prompt_len = len(tokenizer(prompt_formatted).input_ids)
            if prompt_len <= input_len:
                samples.append(
                    SampleRequest(
                        prompt=prompt_formatted if return_prompt_formatted else prompt,
                        prompt_len=prompt_len,
                        expected_output_len=output_len,
                    )
                )
        return samples


# -----------------------------------------------------------------------------
# BurstGPT Dataset Implementation
# -----------------------------------------------------------------------------


class BurstGPTDataset(BenchmarkDataset):
    """
    Implements the BurstGPT dataset.  Loads data from a CSV file and generates
    sample requests based on synthetic prompt generation. Only rows with Model
    "GPT-4" and positive response tokens are used.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(
        self,
    ):
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        df = pd.read_csv(self.dataset_path)
        # Filter to keep only GPT-4 rows.
        gpt4_df = df[df["Model"] == "GPT-4"]
        # Remove failed requests (where Response tokens is 0 or less).
        gpt4_df = gpt4_df[gpt4_df["Response tokens"] > 0]
        # Sample the desired number of rows.
        self.data = gpt4_df

    def _sample_loaded_data(self, num_requests: int) -> list:
        if num_requests <= len(self.data):
            data = self.data.sample(n=num_requests, random_state=self.random_seed)
        else:
            data = self.data.sample(
                n=num_requests,
                random_state=self.random_seed,
                replace=True,
            )
        # Convert the dataframe to a list of lists.
        return data.values.tolist()

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        max_loras: Optional[int] = None,
        lora_path: Optional[str] = None,
        **kwargs,
    ) -> list[SampleRequest]:
        samples = []
        data = self._sample_loaded_data(num_requests=num_requests)
        for i in range(num_requests):
            input_len = int(data[i][2])
            output_len = int(data[i][3])
            lora_req, tokenizer = self.get_random_lora_request(
                tokenizer=tokenizer, max_loras=max_loras, lora_path=lora_path
            )
            vocab_size = tokenizer.vocab_size
            # Generate a synthetic prompt: a list of token IDs computed as (i +
            # j) modulo vocab_size.
            token_ids = [(i + j) % vocab_size for j in range(input_len)]
            prompt = tokenizer.decode(token_ids)
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=input_len,
                    expected_output_len=output_len,
                    lora_request=lora_req,
                )
            )
        return samples


# -----------------------------------------------------------------------------
# HuggingFace Dataset Base Implementation
# -----------------------------------------------------------------------------
class HuggingFaceDataset(BenchmarkDataset):
    """Base class for datasets hosted on HuggingFace."""

    SUPPORTED_DATASET_PATHS: Union[set[str], dict[str, Callable]] = set()

    def __init__(
        self,
        dataset_path: str,
        dataset_split: str,
        dataset_subset: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(dataset_path=dataset_path, **kwargs)

        self.dataset_split = dataset_split
        self.dataset_subset = dataset_subset
        self.load_data()

    def load_data(self) -> None:
        """Load data from HuggingFace datasets."""
        self.data = load_dataset(
            self.dataset_path,
            name=self.dataset_subset,
            split=self.dataset_split,
            streaming=True,
        )
        self.data = self.data.shuffle(seed=self.random_seed)


# -----------------------------------------------------------------------------
# Conversation Dataset Implementation
# -----------------------------------------------------------------------------


class ConversationDataset(HuggingFaceDataset):
    """Dataset for conversation data with multimodal support."""

    SUPPORTED_DATASET_PATHS = {
        "lmms-lab/LLaVA-OneVision-Data",
        "Aeala/ShareGPT_Vicuna_unfiltered",
    }
    IS_MULTIMODAL = True

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list:
        # Filter examples with at least 2 conversations
        filtered_data = self.data.filter(lambda x: len(x["conversations"]) >= 2)
        sampled_requests = []
        dynamic_output = output_len is None

        for item in filtered_data:
            if len(sampled_requests) >= num_requests:
                break
            conv = item["conversations"]
            prompt, completion = conv[0]["value"], conv[1]["value"]

            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            completion_len = len(completion_ids)
            output_len = completion_len if dynamic_output else output_len
            assert isinstance(output_len, int) and output_len > 0
            if dynamic_output and not is_valid_sequence(prompt_len, completion_len):
                continue
            mm_content = process_image(item["image"]) if "image" in item else None
            if enable_multimodal_chat:
                # Note: when chat is enabled the request prompt_len is no longer
                # accurate and we will be using request output to count the
                # actual prompt len and output len
                prompt = self.apply_multimodal_chat_transformation(prompt, mm_content)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=mm_content,
                )
            )
        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests


# -----------------------------------------------------------------------------
# Vision Arena Dataset Implementation
# -----------------------------------------------------------------------------


class VisionArenaDataset(HuggingFaceDataset):
    """
    Vision Arena Dataset.
    """

    DEFAULT_OUTPUT_LEN = 128
    SUPPORTED_DATASET_PATHS = {
        "lmarena-ai/VisionArena-Chat": lambda x: x["conversation"][0][0]["content"],
        "lmarena-ai/vision-arena-bench-v0.1": lambda x: x["turns"][0][0]["content"],
    }
    IS_MULTIMODAL = True

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list:
        output_len = output_len if output_len is not None else self.DEFAULT_OUTPUT_LEN
        sampled_requests = []
        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            parser_fn = self.SUPPORTED_DATASET_PATHS.get(self.dataset_path)
            if parser_fn is None:
                raise ValueError(f"Unsupported dataset path: {self.dataset_path}")
            prompt = parser_fn(item)
            mm_content = process_image(item["images"][0])
            prompt_len = len(tokenizer(prompt).input_ids)
            if enable_multimodal_chat:
                # Note: when chat is enabled the request prompt_len is no longer
                # accurate and we will be using request output to count the
                # actual prompt len
                prompt = self.apply_multimodal_chat_transformation(prompt, mm_content)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=mm_content,
                )
            )
        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests


# -----------------------------------------------------------------------------
# Instruct Coder Dataset Implementation
# -----------------------------------------------------------------------------


class InstructCoderDataset(HuggingFaceDataset):
    """
    InstructCoder Dataset.
    https://huggingface.co/datasets/likaixin/InstructCoder

    InstructCoder is the dataset designed for general code editing.  It consists
    of 114,239 instruction-input-output triplets, and covers multiple distinct
    code editing scenario.
    """

    DEFAULT_OUTPUT_LEN = 200  # this is the average default output length
    SUPPORTED_DATASET_PATHS = {
        "likaixin/InstructCoder",
    }

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list:
        output_len = output_len if output_len is not None else self.DEFAULT_OUTPUT_LEN
        sampled_requests = []
        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            prompt = f"{item['instruction']}:\n{item['input']}"
            prompt_len = len(tokenizer(prompt).input_ids)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                )
            )
        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests


# -----------------------------------------------------------------------------
# MT-Bench Dataset Implementation
# -----------------------------------------------------------------------------


class MTBenchDataset(HuggingFaceDataset):
    """
    MT-Bench Dataset.
    https://huggingface.co/datasets/philschmid/mt-bench

    We create a single turn dataset for MT-Bench.
    This is similar to Spec decoding benchmark setup in vLLM
    https://github.com/vllm-project/vllm/blob/9d98ab5ec/examples/offline_inference/eagle.py#L14-L18
    """  # noqa: E501

    DEFAULT_OUTPUT_LEN = 256  # avg len used in SD bench in vLLM
    SUPPORTED_DATASET_PATHS = {
        "philschmid/mt-bench",
    }

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list:
        output_len = output_len if output_len is not None else self.DEFAULT_OUTPUT_LEN
        sampled_requests = []

        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            prompt = item["turns"][0]

            # apply template
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )

            prompt_len = len(tokenizer(prompt).input_ids)
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                )
            )
        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests


# -----------------------------------------------------------------------------
# AIMO Dataset Implementation
# -----------------------------------------------------------------------------


class AIMODataset(HuggingFaceDataset):
    """
    Dataset class for processing a AIMO dataset with reasoning questions.
    """

    SUPPORTED_DATASET_PATHS = {
        "AI-MO/aimo-validation-aime",
        "AI-MO/NuminaMath-1.5",
        "AI-MO/NuminaMath-CoT",
    }

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        **kwargs,
    ) -> list:
        sampled_requests = []
        dynamic_output = output_len is None

        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            prompt, completion = item["problem"], item["solution"]

            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            completion_len = len(completion_ids)
            output_len = completion_len if dynamic_output else output_len
            assert isinstance(output_len, int) and output_len > 0
            if dynamic_output and not is_valid_sequence(
                prompt_len, completion_len, max_prompt_len=2048, max_total_len=32000
            ):
                continue
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=None,
                )
            )
        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests


# -----------------------------------------------------------------------------
# Next Edit Prediction Dataset Implementation
# -----------------------------------------------------------------------------


zeta_prompt = """### Instruction:
You are a code completion assistant and your task is to analyze user edits and then rewrite an excerpt that the user provides, suggesting the appropriate edits within the excerpt, taking into account the cursor location.

### User Edits:

{}

### User Excerpt:

{}

### Response:

"""  # noqa: E501


def _format_zeta_prompt(
    sample: dict, original_start_marker: str = "<|editable_region_start|>"
) -> dict:
    """Format the zeta prompt for the Next Edit Prediction (NEP) dataset.

    This function formats examples from the NEP dataset
    into prompts and expected outputs. It could be
    further extended to support more NEP datasets.

    Args:
        sample: The dataset sample containing events,
            inputs, and outputs.
        original_start_marker: The marker indicating the
            start of the editable region. Defaults to
            "<|editable_region_start|>".

    Returns:
        A dictionary with the formatted prompts and expected outputs.
    """
    events = sample["events"]
    input = sample["input"]
    output = sample["output"]
    prompt = zeta_prompt.format(events, input)

    # following the original implementation, extract the focused region
    # from the raw output
    output_start_index = output.find(original_start_marker)
    output_focused_region = output[output_start_index:]
    expected_output = output_focused_region

    return {"prompt": prompt, "expected_output": expected_output}


class NextEditPredictionDataset(HuggingFaceDataset):
    """
    Dataset class for processing a Next Edit Prediction dataset.
    """

    SUPPORTED_DATASET_PATHS = {
        "zed-industries/zeta",
    }
    MAPPING_PROMPT_FUNCS = {
        "zed-industries/zeta": _format_zeta_prompt,
    }

    def sample(self, tokenizer: PreTrainedTokenizerBase, num_requests: int, **kwargs):
        formatting_prompt_func = self.MAPPING_PROMPT_FUNCS.get(self.dataset_path)
        if formatting_prompt_func is None:
            raise ValueError(f"Unsupported dataset path: {self.dataset_path}")
        samples = []
        for sample in self.data:
            sample = formatting_prompt_func(sample)
            samples.append(
                SampleRequest(
                    prompt=sample["prompt"],
                    prompt_len=len(tokenizer(sample["prompt"]).input_ids),
                    expected_output_len=len(
                        tokenizer(sample["expected_output"]).input_ids
                    ),
                )
            )
            if len(samples) >= num_requests:
                break
        self.maybe_oversample_requests(samples, num_requests)
        return samples


# -----------------------------------------------------------------------------
# ASR Dataset Implementation
# -----------------------------------------------------------------------------


class ASRDataset(HuggingFaceDataset):
    """
    Dataset class for processing a ASR dataset for transcription.
    Tested on the following set:

    +----------------+----------------------------------------+--------------------------+-----------------------------+
    | Dataset        | Domain                                 | Speaking Style           | hf-subset                   |
    +----------------+----------------------------------------+--------------------------+-----------------------------+
    | TED-LIUM       | TED talks                              | Oratory                  | release1, release2, release3|
    |                |                                        |                          | release3-speaker-adaptation |
    | VoxPopuli      | European Parliament                    | Oratory                  | en, de, it, fr,  ...        |
    | LibriSpeech    | Audiobook                              | Narrated                 | "LIUM/tedlium"              |
    | GigaSpeech     | Audiobook, podcast, YouTube            | Narrated, spontaneous    | xs, s, m, l, xl, dev, test  |
    | SPGISpeech     | Financial meetings                     | Oratory, spontaneous     | S, M, L, dev, test          |
    | AMI            | Meetings                               | Spontaneous              | ihm, sdm                    |
    +----------------+----------------------------------------+--------------------------+-----------------------------+

    """  # noqa: E501

    SUPPORTED_DATASET_PATHS = {
        "openslr/librispeech_asr",
        "facebook/voxpopuli",
        "LIUM/tedlium",
        "edinburghcstr/ami",
        "speechcolab/gigaspeech",
        "kensho/spgispeech",
    }

    DEFAULT_OUTPUT_LEN = 128
    IS_MULTIMODAL = True

    # TODO Whisper-specific. Abstract interface when more models are supported.
    TRANSCRIPTION_PREAMBLE = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
    skip_long_audios: bool = True

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        **kwargs,
    ) -> list:
        import librosa

        output_len = output_len if output_len is not None else self.DEFAULT_OUTPUT_LEN
        prompt = ASRDataset.TRANSCRIPTION_PREAMBLE
        prompt_len = len(tokenizer(prompt).input_ids)
        sampled_requests = []
        skipped = 0
        for item in self.data:
            if len(sampled_requests) >= num_requests:
                break
            audio = item["audio"]
            y, sr = audio["array"], audio["sampling_rate"]
            duration_s = librosa.get_duration(y=y, sr=sr)
            # Whisper max supported duration
            if self.skip_long_audios and duration_s > 30:
                skipped += 1
                continue

            mm_content = {"audio": (y, sr)}
            sampled_requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len,
                    multi_modal_data=mm_content,
                )
            )
        if skipped:
            logger.warning(
                "%d samples discarded from dataset due to"
                " their length being greater than"
                " what Whisper supports.",
                skipped,
            )
        self.maybe_oversample_requests(sampled_requests, num_requests)
        return sampled_requests