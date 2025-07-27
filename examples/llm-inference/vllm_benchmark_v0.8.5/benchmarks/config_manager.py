# SPDX-License-Identifier: Apache-2.0
"""
配置管理模块 - 支持YAML配置文件和命令行参数合并

这个模块提供了完整的配置管理功能，包括：
1. YAML配置文件的读取和解析
2. 命令行参数与配置文件的合并
3. 配置参数的验证和默认值设置
4. 配置文件的保存和导出功能

主要用于简化vLLM基准测试的参数管理，让用户可以通过配置文件
而不是冗长的命令行参数来运行测试。
"""

import argparse
import os
import yaml
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    """
    基准测试配置类

    这个数据类包含了vLLM基准测试所需的所有配置参数。
    使用dataclass装饰器可以自动生成__init__、__repr__等方法，
    并提供类型提示和默认值。
    """

    # ==================== 模型相关配置 ====================
    model: str = "meta-llama/Llama-2-7b-hf"  # 模型名称或路径，支持HuggingFace模型ID
    backend: str = "vllm"                     # 推理后端：vllm, openai, openai-chat, tgi
    endpoint: str = "/v1/completions"         # API端点路径
    host: str = "localhost"                   # 服务器主机地址
    port: int = 8000                          # 服务器端口号

    # ==================== 数据集相关配置 ====================
    dataset_name: str = "random"             # 数据集类型：random, sharegpt, sonnet, hf等
    dataset_path: Optional[str] = None       # 数据集文件路径（某些数据集需要）
    num_prompts: int = 1000                  # 测试请求数量
    input_len: int = 1024                    # 输入序列长度（用于random数据集）
    output_len: int = 128                    # 期望输出序列长度

    # ==================== 测试行为参数 ====================
    request_rate: float = float("inf")       # 请求发送速率（req/s），inf表示尽可能快
    burstiness: float = 1.0                  # 请求突发性参数，控制请求时间分布
    seed: int = 0                            # 随机种子，确保测试结果可重现

    # ==================== 模型采样参数 ====================
    temperature: float = 1.0                 # 温度参数，控制输出随机性
    top_p: float = 1.0                       # nucleus采样参数
    top_k: int = -1                          # top-k采样参数，-1表示不使用
    max_tokens: Optional[int] = None         # 最大生成token数，None使用output_len

    # ==================== 结果输出配置 ====================
    save_result: bool = True                 # 是否保存测试结果到文件
    result_dir: str = "./benchmark_results"  # 结果保存目录
    result_filename: Optional[str] = None    # 结果文件名，None则自动生成
    enable_visualization: bool = True        # 是否生成可视化图表

    # ==================== 高级性能配置 ====================
    tensor_parallel_size: int = 1            # 张量并行度，用于多GPU推理
    pipeline_parallel_size: int = 1         # 流水线并行度
    max_model_len: Optional[int] = None      # 模型最大序列长度限制
    trust_remote_code: bool = False          # 是否信任远程代码（某些模型需要）

    # ==================== HuggingFace数据集配置 ====================
    hf_split: str = "train"                  # HF数据集的分割：train, test, validation
    hf_subset: Optional[str] = None          # HF数据集的子集名称

    # ==================== LoRA适配器配置 ====================
    enable_lora: bool = False                # 是否启用LoRA适配器
    max_loras: Optional[int] = None          # 最大LoRA数量
    max_lora_rank: Optional[int] = None      # LoRA的最大rank值
    lora_path: Optional[str] = None          # LoRA权重文件路径


class ConfigManager:
    """
    配置管理器

    这个类负责管理基准测试的所有配置参数，提供以下功能：
    1. 从YAML文件加载配置
    2. 与命令行参数合并
    3. 配置验证和错误处理
    4. 配置的保存和导出
    5. 配置信息的格式化显示

    使用示例:
        config_manager = ConfigManager()
        config_manager.load_from_yaml("config.yaml")
        config_manager.merge_with_args(args)
        config = config_manager.get_config()
    """

    def __init__(self):
        """
        初始化配置管理器
        创建一个默认的BenchmarkConfig实例
        """
        self.config = BenchmarkConfig()

    def load_from_yaml(self, config_path: str) -> None:
        """
        从YAML文件加载配置

        Args:
            config_path: YAML配置文件的路径

        Raises:
            FileNotFoundError: 当配置文件不存在时抛出
            yaml.YAMLError: 当YAML文件格式错误时抛出
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"YAML文件格式错误: {e}")

        if yaml_config is None:
            print("警告: 配置文件为空，使用默认配置")
            return

        # 更新配置，只接受已定义的配置项
        for key, value in yaml_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"警告: 未知配置项 '{key}' 将被忽略")

    def merge_with_args(self, args: argparse.Namespace) -> None:
        """
        合并命令行参数到配置中

        命令行参数的优先级高于配置文件，即如果同时指定了
        配置文件和命令行参数，将使用命令行参数的值。

        Args:
            args: 解析后的命令行参数对象
        """
        for key, value in vars(args).items():
            # 只有当命令行参数不为None且配置类有该属性时才更新
            if value is not None and hasattr(self.config, key):
                setattr(self.config, key, value)

    def save_to_yaml(self, config_path: str) -> None:
        """
        保存当前配置到YAML文件

        Args:
            config_path: 要保存的YAML文件路径
        """
        config_dict = {}
        # 遍历配置对象的所有属性，排除私有属性
        for key, value in self.config.__dict__.items():
            if not key.startswith('_'):
                config_dict[key] = value

        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        # 保存为YAML格式，使用UTF-8编码
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    def get_config(self) -> BenchmarkConfig:
        """
        获取当前配置对象

        Returns:
            BenchmarkConfig: 当前的配置对象
        """
        return self.config
    
    def print_config(self) -> None:
        """
        打印当前配置信息

        以分类的方式显示所有配置参数，便于用户查看和确认
        测试配置是否正确。配置按功能分组显示，包括：
        - 模型配置：模型相关的基本设置
        - 数据集配置：测试数据相关设置
        - 测试参数：控制测试行为的参数
        - 采样参数：模型生成时的采样设置
        - 输出配置：结果保存和可视化设置
        - 高级配置：性能优化相关设置
        """
        print("=" * 50)
        print("当前基准测试配置:")
        print("=" * 50)

        # 按功能分组显示配置参数
        sections = {
            "模型配置": ["model", "backend", "endpoint", "host", "port"],
            "数据集配置": ["dataset_name", "dataset_path", "num_prompts", "input_len", "output_len"],
            "测试参数": ["request_rate", "burstiness", "seed"],
            "采样参数": ["temperature", "top_p", "top_k", "max_tokens"],
            "输出配置": ["save_result", "result_dir", "enable_visualization"],
            "高级配置": ["tensor_parallel_size", "pipeline_parallel_size", "max_model_len"]
        }

        # 遍历每个配置分组
        for section_name, keys in sections.items():
            print(f"\n{section_name}:")
            for key in keys:
                if hasattr(self.config, key):
                    value = getattr(self.config, key)
                    # 格式化显示配置值
                    if isinstance(value, str) and len(value) > 50:
                        # 长字符串截断显示
                        print(f"  {key}: {value[:47]}...")
                    else:
                        print(f"  {key}: {value}")

        print("=" * 50)


def create_default_configs():
    """
    创建默认配置文件模板

    这个函数会创建三个预设的配置文件模板，涵盖了最常用的测试场景：
    1. serving_test.yaml - 在线服务基准测试配置
    2. throughput_test.yaml - 离线吞吐量测试配置
    3. vision_test.yaml - 视觉语言模型测试配置

    每个模板都包含了该测试场景的推荐参数设置，用户可以直接使用
    或在此基础上进行修改。
    """
    # 定义三种不同场景的配置模板
    configs = {
        # 在线服务测试配置 - 用于测试vLLM服务的在线推理性能
        "serving_test.yaml": {
            "model": "meta-llama/Llama-2-7b-hf",                    # 使用Llama-2 7B模型
            "backend": "vllm",                                       # 使用vLLM后端
            "endpoint": "/v1/completions",                           # OpenAI兼容的API端点
            "dataset_name": "sharegpt",                              # 使用ShareGPT对话数据集
            "dataset_path": "./ShareGPT_V3_unfiltered_cleaned_split.json",  # 数据集文件路径
            "num_prompts": 100,                                      # 测试100个请求
            "request_rate": 2.0,                                     # 每秒2个请求的速率
            "enable_visualization": True,                            # 启用可视化
            "save_result": True                                      # 保存测试结果
        },

        # 离线吞吐量测试配置 - 用于测试最大吞吐量性能
        "throughput_test.yaml": {
            "model": "meta-llama/Llama-2-7b-hf",                    # 使用相同的模型便于对比
            "backend": "vllm",                                       # vLLM后端
            "dataset_name": "random",                                # 使用随机生成的数据
            "num_prompts": 1000,                                     # 更多请求数以获得稳定的吞吐量数据
            "input_len": 1024,                                       # 输入序列长度
            "output_len": 128,                                       # 输出序列长度
            "tensor_parallel_size": 1,                               # 张量并行度
            "enable_visualization": True                             # 启用可视化
        },

        # 视觉语言模型测试配置 - 用于测试多模态模型性能
        "vision_test.yaml": {
            "model": "Qwen/Qwen2-VL-7B-Instruct",                   # 使用Qwen2-VL视觉语言模型
            "backend": "openai-chat",                                # 使用chat格式的API
            "endpoint": "/v1/chat/completions",                      # chat completions端点
            "dataset_name": "hf",                                    # 使用HuggingFace数据集
            "dataset_path": "lmarena-ai/VisionArena-Chat",           # VisionArena数据集
            "hf_split": "train",                                     # 使用训练集分割
            "num_prompts": 50,                                       # 较少的请求数（视觉模型推理较慢）
            "enable_visualization": True                             # 启用可视化
        }
    }

    # 创建配置文件目录
    config_dir = "./benchmark_configs"
    os.makedirs(config_dir, exist_ok=True)

    # 生成每个配置文件
    for filename, config in configs.items():
        config_path = os.path.join(config_dir, filename)
        with open(config_path, 'w', encoding='utf-8') as f:
            # 使用YAML格式保存，不使用流式格式以提高可读性
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"创建配置模板: {config_path}")


if __name__ == "__main__":
    """
    主函数 - 当直接运行此脚本时执行

    主要用于创建默认的配置文件模板，方便用户快速开始使用。
    运行此脚本会在当前目录下创建benchmark_configs文件夹，
    并生成三个预设的配置文件模板。
    """
    # 创建默认配置文件模板
    create_default_configs()
    print("配置文件模板创建完成!")
    print("\n使用方法:")
    print("1. 查看生成的配置文件: ls benchmark_configs/")
    print("2. 编辑配置文件以适应您的需求")
    print("3. 使用配置文件运行测试: python enhanced_benchmark_serving.py --config benchmark_configs/serving_test.yaml")
