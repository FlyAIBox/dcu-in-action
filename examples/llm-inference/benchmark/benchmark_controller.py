#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
海光DCU K100-AI大模型推理基准测试控制器
版本: v2.0
作者: DCU-in-Action Team

主要功能:
1. 统一管理vLLM、SGLang、Xinference三个推理框架的测试
2. 自动化测试流程，包括环境准备、模型加载、性能测试、结果收集
3. 支持单卡和多卡测试场景
4. 实时监控系统资源和性能指标
5. 生成详细的测试报告
"""

import os
import sys
import time
import json
import yaml
import asyncio
import logging
import argparse
import subprocess
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import psutil
import requests
import aiohttp
import pandas as pd
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    """测试配置类"""
    model_name: str
    model_path: str
    framework: str  # vllm, sglang, xinference
    gpu_count: int
    max_tokens: int
    temperature: float
    top_p: float
    input_lengths: List[int]
    output_lengths: List[int]
    concurrency_levels: List[int]
    test_duration: int
    warmup_requests: int
    port: int

@dataclass
class TestResult:
    """测试结果类"""
    timestamp: str
    framework: str
    model_name: str
    gpu_count: int
    input_length: int
    output_length: int
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    throughput_tokens_per_sec: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    first_token_latency_ms: float
    inter_token_latency_ms: float
    gpu_utilization_avg: float
    gpu_memory_used_gb: float
    cpu_utilization_avg: float
    system_memory_used_gb: float
    power_consumption_w: float
    error_rate: float

class SystemMonitor:
    """系统监控类"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.metrics = []
        
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        
    def collect_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        try:
            # CPU指标
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存指标
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            
            # GPU指标 (通过rocm-smi)
            gpu_metrics = self._get_gpu_metrics()
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_utilization': cpu_percent,
                'memory_used_gb': memory_used_gb,
                'memory_total_gb': memory.total / (1024**3),
                'gpu_metrics': gpu_metrics
            }
            
            if self.monitoring:
                self.metrics.append(metrics)
                
            return metrics
            
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
            return {}
    
    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """获取GPU指标"""
        try:
            # 获取GPU使用率
            result = subprocess.run(['rocm-smi', '--showuse', '--csv'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return {}
                
            lines = result.stdout.strip().split('\n')
            gpu_metrics = {'utilization': [], 'memory_used': [], 'temperature': [], 'power': []}
            
            # 解析CSV输出
            for line in lines[1:]:  # 跳过标题行
                if 'GPU' in line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            utilization = float(parts[1].strip().replace('%', ''))
                            gpu_metrics['utilization'].append(utilization)
                        except (ValueError, IndexError):
                            pass
            
            # 获取温度和功耗
            temp_result = subprocess.run(['rocm-smi', '--showtemp', '--csv'], 
                                       capture_output=True, text=True, timeout=10)
            power_result = subprocess.run(['rocm-smi', '--showpower', '--csv'], 
                                        capture_output=True, text=True, timeout=10)
            
            return gpu_metrics
            
        except Exception as e:
            logger.error(f"获取GPU指标失败: {e}")
            return {}
    
    def get_average_metrics(self) -> Dict[str, float]:
        """获取平均指标"""
        if not self.metrics:
            return {}
            
        avg_metrics = {}
        
        # CPU和内存平均值
        avg_metrics['cpu_utilization_avg'] = sum(m['cpu_utilization'] for m in self.metrics) / len(self.metrics)
        avg_metrics['memory_used_avg'] = sum(m['memory_used_gb'] for m in self.metrics) / len(self.metrics)
        
        # GPU平均值
        gpu_utils = []
        for m in self.metrics:
            if 'gpu_metrics' in m and 'utilization' in m['gpu_metrics']:
                gpu_utils.extend(m['gpu_metrics']['utilization'])
        
        if gpu_utils:
            avg_metrics['gpu_utilization_avg'] = sum(gpu_utils) / len(gpu_utils)
        
        return avg_metrics

class FrameworkManager:
    """推理框架管理器"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.process = None
        
    async def start_server(self) -> bool:
        """启动推理服务器"""
        try:
            if self.config.framework == 'vllm':
                return await self._start_vllm_server()
            elif self.config.framework == 'sglang':
                return await self._start_sglang_server()
            elif self.config.framework == 'xinference':
                return await self._start_xinference_server()
            else:
                logger.error(f"不支持的框架: {self.config.framework}")
                return False
        except Exception as e:
            logger.error(f"启动{self.config.framework}服务器失败: {e}")
            return False
    
    async def _start_vllm_server(self) -> bool:
        """启动vLLM服务器"""
        cmd = [
            'python', '-m', 'vllm.entrypoints.openai.api_server',
            '--model', self.config.model_path,
            '--dtype', 'float16',
            '--tensor-parallel-size', str(self.config.gpu_count),
            '--gpu-memory-utilization', '0.9',
            '--max-model-len', str(self.config.max_tokens),
            '--swap-space', '16',
            '--port', str(self.config.port),
            '--host', '0.0.0.0'
        ]
        
        logger.info(f"启动vLLM服务器: {' '.join(cmd)}")
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 等待服务器启动
        return await self._wait_for_server_ready()
    
    async def _start_sglang_server(self) -> bool:
        """启动SGLang服务器"""
        cmd = [
            'python', '-m', 'sglang.launch_server',
            '--model-path', self.config.model_path,
            '--tp-size', str(self.config.gpu_count),
            '--mem-fraction-static', '0.8',
            '--port', str(self.config.port),
            '--host', '0.0.0.0'
        ]
        
        logger.info(f"启动SGLang服务器: {' '.join(cmd)}")
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        return await self._wait_for_server_ready()
    
    async def _start_xinference_server(self) -> bool:
        """启动Xinference服务器"""
        # Xinference需要先启动服务，然后注册模型
        cmd = [
            'xinference-local',
            '--host', '0.0.0.0',
            '--port', str(self.config.port)
        ]
        
        logger.info(f"启动Xinference服务器: {' '.join(cmd)}")
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if await self._wait_for_server_ready():
            # 注册模型
            return await self._register_xinference_model()
        
        return False
    
    async def _register_xinference_model(self) -> bool:
        """注册Xinference模型"""
        try:
            url = f"http://localhost:{self.config.port}/v1/models"
            data = {
                "model_name": self.config.model_name,
                "model_path": self.config.model_path,
                "model_type": "LLM"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        logger.info("模型注册成功")
                        return True
                    else:
                        logger.error(f"模型注册失败: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"注册模型失败: {e}")
            return False
    
    async def _wait_for_server_ready(self, timeout: int = 300) -> bool:
        """等待服务器就绪"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"http://localhost:{self.config.port}/v1/models"
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            logger.info(f"{self.config.framework}服务器已就绪")
                            return True
            except:
                pass
                
            await asyncio.sleep(5)
        
        logger.error(f"{self.config.framework}服务器启动超时")
        return False
    
    def stop_server(self):
        """停止服务器"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=30)
                logger.info(f"{self.config.framework}服务器已停止")
            except subprocess.TimeoutExpired:
                self.process.kill()
                logger.warning(f"强制终止{self.config.framework}服务器")
            except Exception as e:
                logger.error(f"停止服务器失败: {e}")

class LoadGenerator:
    """负载生成器"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.results = []
        
    async def run_test(self, input_length: int, output_length: int, concurrency: int) -> TestResult:
        """运行测试"""
        logger.info(f"开始测试: 输入长度={input_length}, 输出长度={output_length}, 并发数={concurrency}")
        
        # 生成测试prompts
        prompts = self._generate_prompts(input_length, concurrency * 10)  # 生成足够的prompts
        
        # 运行预热
        await self._warmup(prompts[:self.config.warmup_requests])
        
        # 开始监控
        monitor = SystemMonitor()
        monitor.start_monitoring()
        
        # 运行实际测试
        start_time = time.time()
        request_results = await self._run_concurrent_requests(prompts, concurrency, output_length)
        end_time = time.time()
        
        # 停止监控
        monitor.stop_monitoring()
        avg_metrics = monitor.get_average_metrics()
        
        # 计算结果
        result = self._calculate_metrics(
            request_results, start_time, end_time, 
            input_length, output_length, concurrency, avg_metrics
        )
        
        return result
    
    def _generate_prompts(self, target_length: int, count: int) -> List[str]:
        """生成指定长度的prompts"""
        # 基础prompt模板
        base_prompts = [
            "请详细解释人工智能在现代社会中的应用和发展趋势，包括机器学习、深度学习等技术的具体实现。",
            "分析大数据技术对企业决策的影响，讨论数据挖掘和数据分析在商业智能中的重要作用。",
            "探讨云计算技术的发展历程，比较不同云服务模式的优缺点，以及未来的发展方向。",
            "描述区块链技术的工作原理，分析其在金融、供应链管理等领域的应用前景。",
            "讨论物联网技术对智慧城市建设的推动作用，包括传感器网络和边缘计算的应用。"
        ]
        
        prompts = []
        for i in range(count):
            base = base_prompts[i % len(base_prompts)]
            
            # 调整到目标长度（粗略估算，1个中文字符约等于1.5个token）
            if target_length > len(base):
                # 扩展prompt
                expansion = "请进一步详细说明，提供更多的技术细节和实际案例。" * ((target_length - len(base)) // 20 + 1)
                prompt = base + expansion
            else:
                # 截断prompt
                prompt = base[:target_length]
            
            prompts.append(prompt)
        
        return prompts
    
    async def _warmup(self, prompts: List[str]):
        """预热"""
        logger.info("开始预热...")
        for prompt in prompts[:min(len(prompts), self.config.warmup_requests)]:
            try:
                await self._send_request(prompt, 32)  # 短输出用于预热
            except:
                pass
        logger.info("预热完成")
    
    async def _run_concurrent_requests(self, prompts: List[str], concurrency: int, output_length: int) -> List[Dict]:
        """运行并发请求"""
        results = []
        semaphore = asyncio.Semaphore(concurrency)
        
        async def worker(prompt: str) -> Dict:
            async with semaphore:
                return await self._send_request(prompt, output_length)
        
        # 创建任务
        tasks = []
        request_count = 0
        start_time = time.time()
        
        while time.time() - start_time < self.config.test_duration:
            if request_count < len(prompts):
                task = asyncio.create_task(worker(prompts[request_count]))
                tasks.append(task)
                request_count += 1
                await asyncio.sleep(0.1)  # 控制请求速率
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤异常结果
        valid_results = [r for r in results if isinstance(r, dict) and 'error' not in r]
        
        return valid_results
    
    async def _send_request(self, prompt: str, max_tokens: int) -> Dict:
        """发送单个请求"""
        try:
            url = f"http://localhost:{self.config.port}/v1/completions"
            data = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "stream": False
            }
            
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, timeout=120) as response:
                    if response.status == 200:
                        result = await response.json()
                        end_time = time.time()
                        
                        # 计算token数量（粗略估算）
                        input_tokens = len(prompt) // 2  # 粗略估算
                        output_tokens = len(result.get('choices', [{}])[0].get('text', '')) // 2
                        
                        return {
                            'success': True,
                            'latency': (end_time - start_time) * 1000,  # ms
                            'input_tokens': input_tokens,
                            'output_tokens': output_tokens,
                            'total_tokens': input_tokens + output_tokens,
                            'first_token_latency': 0,  # 需要流式响应才能准确测量
                            'response': result
                        }
                    else:
                        return {'error': f"HTTP {response.status}"}
                        
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_metrics(self, request_results: List[Dict], start_time: float, 
                          end_time: float, input_length: int, output_length: int, 
                          concurrency: int, system_metrics: Dict) -> TestResult:
        """计算测试指标"""
        
        total_requests = len(request_results)
        successful_requests = len([r for r in request_results if r.get('success')])
        failed_requests = total_requests - successful_requests
        
        if successful_requests == 0:
            logger.warning("没有成功的请求")
            return TestResult(
                timestamp=datetime.now().isoformat(),
                framework=self.config.framework,
                model_name=self.config.model_name,
                gpu_count=self.config.gpu_count,
                input_length=input_length,
                output_length=output_length,
                concurrency=concurrency,
                total_requests=total_requests,
                successful_requests=0,
                failed_requests=failed_requests,
                throughput_tokens_per_sec=0,
                latency_p50_ms=0,
                latency_p90_ms=0,
                latency_p99_ms=0,
                first_token_latency_ms=0,
                inter_token_latency_ms=0,
                gpu_utilization_avg=0,
                gpu_memory_used_gb=0,
                cpu_utilization_avg=0,
                system_memory_used_gb=0,
                power_consumption_w=0,
                error_rate=100.0
            )
        
        # 计算延迟统计
        latencies = [r['latency'] for r in request_results if r.get('success')]
        latencies.sort()
        
        p50_idx = int(len(latencies) * 0.5)
        p90_idx = int(len(latencies) * 0.9)
        p99_idx = int(len(latencies) * 0.99)
        
        # 计算吞吐量
        total_tokens = sum(r['total_tokens'] for r in request_results if r.get('success'))
        total_time = end_time - start_time
        throughput = total_tokens / total_time if total_time > 0 else 0
        
        return TestResult(
            timestamp=datetime.now().isoformat(),
            framework=self.config.framework,
            model_name=self.config.model_name,
            gpu_count=self.config.gpu_count,
            input_length=input_length,
            output_length=output_length,
            concurrency=concurrency,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            throughput_tokens_per_sec=throughput,
            latency_p50_ms=latencies[p50_idx] if latencies else 0,
            latency_p90_ms=latencies[p90_idx] if latencies else 0,
            latency_p99_ms=latencies[p99_idx] if latencies else 0,
            first_token_latency_ms=0,  # 需要流式支持
            inter_token_latency_ms=0,  # 需要流式支持
            gpu_utilization_avg=system_metrics.get('gpu_utilization_avg', 0),
            gpu_memory_used_gb=0,  # 需要实现
            cpu_utilization_avg=system_metrics.get('cpu_utilization_avg', 0),
            system_memory_used_gb=system_metrics.get('memory_used_avg', 0),
            power_consumption_w=0,  # 需要实现
            error_rate=(failed_requests / total_requests * 100) if total_requests > 0 else 0
        )

class BenchmarkController:
    """基准测试控制器"""
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.configs = self._load_configs()
        self.results = []
        
    def _load_configs(self) -> List[TestConfig]:
        """加载测试配置"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            configs = []
            for test_case in data['test_cases']:
                config = TestConfig(**test_case)
                configs.append(config)
            
            return configs
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            sys.exit(1)
    
    async def run_all_tests(self):
        """运行所有测试"""
        logger.info(f"开始运行 {len(self.configs)} 个测试配置")
        
        for i, config in enumerate(self.configs):
            logger.info(f"运行测试配置 {i+1}/{len(self.configs)}: {config.framework} - {config.model_name}")
            
            try:
                await self._run_single_framework_test(config)
            except Exception as e:
                logger.error(f"测试失败: {e}")
                continue
        
        # 保存结果
        self._save_results()
        logger.info("所有测试完成")
    
    async def _run_single_framework_test(self, config: TestConfig):
        """运行单个框架的测试"""
        framework_manager = FrameworkManager(config)
        load_generator = LoadGenerator(config)
        
        try:
            # 启动服务器
            if not await framework_manager.start_server():
                logger.error(f"启动{config.framework}服务器失败")
                return
            
            logger.info(f"{config.framework}服务器启动成功，开始测试")
            
            # 运行测试矩阵
            for input_length in config.input_lengths:
                for output_length in config.output_lengths:
                    for concurrency in config.concurrency_levels:
                        try:
                            result = await load_generator.run_test(
                                input_length, output_length, concurrency
                            )
                            self.results.append(result)
                            
                            logger.info(f"测试完成 - 吞吐量: {result.throughput_tokens_per_sec:.2f} tokens/s, "
                                      f"延迟P50: {result.latency_p50_ms:.2f}ms")
                            
                            # 短暂休息
                            await asyncio.sleep(10)
                            
                        except Exception as e:
                            logger.error(f"测试失败 (input={input_length}, output={output_length}, "
                                       f"concurrency={concurrency}): {e}")
                            continue
            
        finally:
            # 清理
            framework_manager.stop_server()
            await asyncio.sleep(30)  # 等待服务器完全停止
    
    def _save_results(self):
        """保存测试结果"""
        try:
            # 创建结果目录
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存为JSON
            json_file = results_dir / f"benchmark_results_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(r) for r in self.results], f, ensure_ascii=False, indent=2)
            
            # 保存为CSV
            csv_file = results_dir / f"benchmark_results_{timestamp}.csv"
            df = pd.DataFrame([asdict(r) for r in self.results])
            df.to_csv(csv_file, index=False)
            
            logger.info(f"结果已保存: {json_file}, {csv_file}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")

def create_sample_config():
    """创建示例配置文件"""
    config = {
        'test_cases': [
            {
                'model_name': 'deepseek-7b',
                'model_path': '/path/to/deepseek-llm-7b-base',
                'framework': 'vllm',
                'gpu_count': 1,
                'max_tokens': 4096,
                'temperature': 0.7,
                'top_p': 0.9,
                'input_lengths': [64, 128, 256],
                'output_lengths': [64, 128, 256],
                'concurrency_levels': [1, 4, 8, 16, 32],
                'test_duration': 120,
                'warmup_requests': 10,
                'port': 8000
            },
            {
                'model_name': 'deepseek-7b',
                'model_path': '/path/to/deepseek-llm-7b-base',
                'framework': 'sglang',
                'gpu_count': 1,
                'max_tokens': 4096,
                'temperature': 0.7,
                'top_p': 0.9,
                'input_lengths': [64, 128, 256],
                'output_lengths': [64, 128, 256],
                'concurrency_levels': [1, 4, 8, 16, 32],
                'test_duration': 120,
                'warmup_requests': 10,
                'port': 8001
            }
        ]
    }
    
    with open('benchmark_config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    print("示例配置文件已创建: benchmark_config.yaml")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='海光DCU K100-AI大模型推理基准测试')
    parser.add_argument('--config', type=str, default='benchmark_config.yaml',
                       help='测试配置文件')
    parser.add_argument('--create-config', action='store_true',
                       help='创建示例配置文件')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config()
        return
    
    if not os.path.exists(args.config):
        logger.error(f"配置文件不存在: {args.config}")
        logger.info("使用 --create-config 创建示例配置文件")
        return
    
    # 运行测试
    controller = BenchmarkController(args.config)
    
    try:
        asyncio.run(controller.run_all_tests())
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试执行失败: {e}")

if __name__ == '__main__':
    main() 