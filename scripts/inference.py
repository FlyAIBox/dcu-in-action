#!/usr/bin/env python3
"""
DCU-in-Action 推理服务脚本
提供模型推理、批量推理、流式推理等功能
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
except ImportError as e:
    print(f"请安装PyTorch: {e}")
    sys.exit(1)

from common.utils.logger import setup_global_logging, get_logger
from common.llm.inference_utils import (
    InferenceConfig,
    InferenceEngine,
    vLLMInferenceEngine,
    InferenceServer,
    create_inference_engine,
    get_preset_config
)
from common.dcu.device_manager import DCUDeviceManager

logger = get_logger(__name__)


def load_test_prompts(file_path: str) -> List[str]:
    """加载测试提示词"""
    if not os.path.exists(file_path):
        # 返回默认测试提示词
        return [
            "你好，请介绍一下自己。",
            "什么是人工智能？",
            "请写一首关于春天的诗。",
            "解释一下机器学习的基本概念。",
            "如何学习Python编程？"
        ]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "prompts" in data:
                return data["prompts"]
            else:
                logger.warning(f"无法解析提示词文件格式: {file_path}")
                return []
    except Exception as e:
        logger.error(f"加载提示词文件失败: {e}")
        return []


def create_inference_config_from_args(args: argparse.Namespace) -> InferenceConfig:
    """从命令行参数创建推理配置"""
    config = InferenceConfig(
        model_name_or_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        batch_size=args.batch_size,
        use_vllm=args.use_vllm,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        enable_streaming=args.enable_streaming,
        tensor_parallel_size=args.tensor_parallel_size,
        max_workers=args.max_workers
    )
    
    return config


def single_inference(args: argparse.Namespace):
    """单条推理"""
    logger.info("开始单条推理...")
    
    # 创建配置
    if args.preset:
        config = get_preset_config(args.preset, args.model_path)
        # 应用命令行覆盖
        if args.temperature is not None:
            config.temperature = args.temperature
        if args.max_new_tokens is not None:
            config.max_new_tokens = args.max_new_tokens
    else:
        config = create_inference_config_from_args(args)
    
    logger.info(f"推理配置: {config}")
    
    # 创建推理引擎
    engine = create_inference_engine(config)
    
    # 执行推理
    prompt = args.prompt or "你好，请介绍一下自己。"
    
    try:
        if config.enable_streaming:
            logger.info(f"开始流式推理，提示词: {prompt}")
            print(f"\n提示词: {prompt}")
            print("回答: ", end="", flush=True)
            
            for chunk in engine.stream_generate(prompt):
                print(chunk, end="", flush=True)
            print("\n")
        else:
            logger.info(f"开始推理，提示词: {prompt}")
            start_time = time.time()
            
            response = engine.generate(prompt)
            
            inference_time = time.time() - start_time
            
            print(f"\n提示词: {prompt}")
            print(f"回答: {response}")
            print(f"推理时间: {inference_time:.2f}秒")
            
            # 计算tokens/秒
            if hasattr(engine, 'tokenizer') and engine.tokenizer:
                response_tokens = len(engine.tokenizer.encode(response))
                throughput = response_tokens / inference_time
                print(f"吞吐量: {throughput:.1f} tokens/秒")
        
        # 清理资源
        if hasattr(engine, 'cleanup'):
            engine.cleanup()
            
    except Exception as e:
        logger.error(f"推理失败: {e}", exc_info=True)
        sys.exit(1)


def batch_inference(args: argparse.Namespace):
    """批量推理"""
    logger.info("开始批量推理...")
    
    # 加载提示词
    prompts = load_test_prompts(args.prompts_file) if args.prompts_file else []
    
    if not prompts:
        prompts = [
            "你好，请介绍一下自己。",
            "什么是人工智能？",
            "请写一首关于春天的诗。",
            "解释一下机器学习的基本概念。",
            "如何学习Python编程？"
        ]
    
    logger.info(f"加载了 {len(prompts)} 个提示词")
    
    # 创建配置
    if args.preset:
        config = get_preset_config(args.preset, args.model_path)
    else:
        config = create_inference_config_from_args(args)
    
    # 创建推理引擎
    engine = create_inference_engine(config)
    
    try:
        logger.info("开始批量推理...")
        start_time = time.time()
        
        responses = engine.batch_generate(prompts)
        
        total_time = time.time() - start_time
        
        # 输出结果
        results = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            result = {
                "id": i + 1,
                "prompt": prompt,
                "response": response,
                "timestamp": time.time()
            }
            results.append(result)
            
            print(f"\n=== 样本 {i+1} ===")
            print(f"提示词: {prompt}")
            print(f"回答: {response}")
        
        # 统计信息
        avg_time_per_sample = total_time / len(prompts)
        throughput = len(prompts) / total_time
        
        print(f"\n=== 批量推理统计 ===")
        print(f"总样本数: {len(prompts)}")
        print(f"总时间: {total_time:.2f}秒")
        print(f"平均每样本时间: {avg_time_per_sample:.2f}秒")
        print(f"吞吐量: {throughput:.2f} 样本/秒")
        
        # 保存结果
        if args.output_file:
            output_data = {
                "config": config.__dict__,
                "statistics": {
                    "total_samples": len(prompts),
                    "total_time": total_time,
                    "avg_time_per_sample": avg_time_per_sample,
                    "throughput": throughput
                },
                "results": results
            }
            
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"结果已保存到: {args.output_file}")
        
        # 清理资源
        if hasattr(engine, 'cleanup'):
            engine.cleanup()
            
    except Exception as e:
        logger.error(f"批量推理失败: {e}", exc_info=True)
        sys.exit(1)


def start_inference_server(args: argparse.Namespace):
    """启动推理服务器"""
    logger.info("启动推理服务器...")
    
    # 创建配置
    if args.preset:
        config = get_preset_config(args.preset, args.model_path)
    else:
        config = create_inference_config_from_args(args)
    
    # 创建推理服务器
    server = InferenceServer(config)
    
    try:
        # 启动服务器
        server.start_server()
        
        logger.info("推理服务器已启动")
        logger.info("开始交互式对话，输入 'quit' 退出")
        
        while True:
            try:
                # 获取用户输入
                prompt = input("\n用户: ").strip()
                
                if prompt.lower() in ['quit', 'exit', '退出']:
                    break
                
                if not prompt:
                    continue
                
                # 提交推理请求
                request_id = server.submit_request(prompt)
                
                # 获取结果
                result = server.get_result(request_id, timeout=30)
                
                if result["status"] == "success":
                    print(f"助手: {result['result']}")
                else:
                    print(f"错误: {result.get('error', '未知错误')}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"处理请求失败: {e}")
        
        # 停止服务器
        server.stop_server()
        logger.info("推理服务器已停止")
        
    except Exception as e:
        logger.error(f"启动推理服务器失败: {e}", exc_info=True)
        sys.exit(1)


def benchmark_inference(args: argparse.Namespace):
    """推理性能基准测试"""
    logger.info("开始推理性能基准测试...")
    
    # 创建配置
    config = create_inference_config_from_args(args)
    
    # 测试不同的配置
    test_configs = [
        {"name": "小批量", "batch_size": 1, "max_new_tokens": 128},
        {"name": "中批量", "batch_size": 4, "max_new_tokens": 128},
        {"name": "大批量", "batch_size": 8, "max_new_tokens": 128},
        {"name": "长文本", "batch_size": 1, "max_new_tokens": 512},
    ]
    
    # 测试提示词
    test_prompts = [
        "请简单介绍一下机器学习。",
        "什么是深度学习？",
        "解释一下神经网络的原理。",
        "人工智能的发展历程是怎样的？",
    ]
    
    benchmark_results = []
    
    for test_config in test_configs:
        logger.info(f"测试配置: {test_config['name']}")
        
        # 更新配置
        config.batch_size = test_config["batch_size"]
        config.max_new_tokens = test_config["max_new_tokens"]
        
        try:
            # 创建推理引擎
            engine = create_inference_engine(config)
            
            # 预热
            engine.generate(test_prompts[0][:50])
            
            # 基准测试
            start_time = time.time()
            
            if test_config["batch_size"] == 1:
                for prompt in test_prompts:
                    engine.generate(prompt)
            else:
                engine.batch_generate(test_prompts[:test_config["batch_size"]])
            
            total_time = time.time() - start_time
            
            # 计算统计
            num_samples = len(test_prompts) if test_config["batch_size"] == 1 else test_config["batch_size"]
            throughput = num_samples / total_time
            
            result = {
                "config_name": test_config["name"],
                "batch_size": test_config["batch_size"],
                "max_new_tokens": test_config["max_new_tokens"],
                "total_time": total_time,
                "throughput": throughput,
                "avg_time_per_sample": total_time / num_samples
            }
            
            benchmark_results.append(result)
            
            logger.info(f"{test_config['name']}: {throughput:.2f} 样本/秒")
            
            # 清理
            if hasattr(engine, 'cleanup'):
                engine.cleanup()
                
        except Exception as e:
            logger.error(f"基准测试失败 {test_config['name']}: {e}")
    
    # 输出基准测试结果
    print("\n=== 推理性能基准测试结果 ===")
    print(f"{'配置':<10} {'批量大小':<8} {'最大tokens':<12} {'吞吐量(样本/秒)':<15} {'平均时间(秒)':<12}")
    print("-" * 70)
    
    for result in benchmark_results:
        print(f"{result['config_name']:<10} {result['batch_size']:<8} "
              f"{result['max_new_tokens']:<12} {result['throughput']:<15.2f} "
              f"{result['avg_time_per_sample']:<12.2f}")
    
    # 保存基准测试结果
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
        logger.info(f"基准测试结果已保存到: {args.output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DCU-in-Action 推理服务脚本")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="推理命令")
    
    # 单条推理
    single_parser = subparsers.add_parser("single", help="单条推理")
    single_parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    single_parser.add_argument("--prompt", type=str, help="输入提示词")
    single_parser.add_argument("--preset", type=str, help="预设配置")
    
    # 批量推理
    batch_parser = subparsers.add_parser("batch", help="批量推理")
    batch_parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    batch_parser.add_argument("--prompts_file", type=str, help="提示词文件路径")
    batch_parser.add_argument("--output_file", type=str, help="输出文件路径")
    batch_parser.add_argument("--preset", type=str, help="预设配置")
    
    # 推理服务器
    server_parser = subparsers.add_parser("server", help="推理服务器")
    server_parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    server_parser.add_argument("--preset", type=str, help="预设配置")
    
    # 基准测试
    benchmark_parser = subparsers.add_parser("benchmark", help="性能基准测试")
    benchmark_parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    benchmark_parser.add_argument("--output_file", type=str, help="输出文件路径")
    
    # 公共参数
    for sub_parser in [single_parser, batch_parser, server_parser, benchmark_parser]:
        sub_parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成tokens数")
        sub_parser.add_argument("--temperature", type=float, default=0.7, help="温度参数")
        sub_parser.add_argument("--top_p", type=float, default=0.9, help="Top-p参数")
        sub_parser.add_argument("--top_k", type=int, default=50, help="Top-k参数")
        sub_parser.add_argument("--batch_size", type=int, default=1, help="批量大小")
        sub_parser.add_argument("--use_vllm", action="store_true", help="使用vLLM引擎")
        sub_parser.add_argument("--load_in_4bit", action="store_true", help="使用4bit量化")
        sub_parser.add_argument("--load_in_8bit", action="store_true", help="使用8bit量化")
        sub_parser.add_argument("--enable_streaming", action="store_true", help="启用流式生成")
        sub_parser.add_argument("--tensor_parallel_size", type=int, default=1, help="张量并行大小")
        sub_parser.add_argument("--max_workers", type=int, default=4, help="最大工作线程数")
        sub_parser.add_argument("--debug", action="store_true", help="调试模式")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 设置日志
    log_level = "DEBUG" if args.debug else "INFO"
    setup_global_logging(level=log_level)
    
    try:
        # DCU设备检查
        device_manager = DCUDeviceManager()
        device_info = device_manager.get_device_info()
        logger.info(f"DCU设备信息: {device_info}")
        
        # 执行相应命令
        if args.command == "single":
            single_inference(args)
        elif args.command == "batch":
            batch_inference(args)
        elif args.command == "server":
            start_inference_server(args)
        elif args.command == "benchmark":
            benchmark_inference(args)
        
        logger.info("推理任务完成！")
        
    except Exception as e:
        logger.error(f"推理任务失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 