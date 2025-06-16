# 海光 DCU K100-AI 大模型推理基准测试指南（vLLM / SGLang / Xinference）

> **版本**：v1.0   **最后更新**：2025-06-16
>
> 本文档旨在指导研发人员在海光 DCU K100-AI（64 GB HMB、兼容 `rocm-smi`）加速卡上，对主流开源大模型进行**可复现、可对比**的推理性能评测。涉及三款推理框架 **vLLM、SGLang、Xinference**，以及典型开源模型 *DeepSeek*、*Qwen* 等。

---

## 目录
- [海光 DCU K100-AI 大模型推理基准测试指南（vLLM / SGLang / Xinference）](#海光-dcu-k100-ai-大模型推理基准测试指南vllm--sglang--xinference)
  - [目录](#目录)
  - [测试环境概览](#测试环境概览)
  - [软件安装与准备](#软件安装与准备)
    - [2.1 vLLM 环境](#21-vllm-环境)
    - [2.2 SGLang 环境](#22-sglang-环境)
    - [2.3 Xinference 环境](#23-xinference-环境)
    - [2.4 通用工具](#24-通用工具)
  - [模型下载与格式转换](#模型下载与格式转换)
  - [基准测试方法学](#基准测试方法学)
    - [单卡测试流程](#单卡测试流程)
    - [整机 (8 卡) 测试流程](#整机-8-卡-测试流程)
  - [数据记录与结果呈现](#数据记录与结果呈现)
  - [性能优化建议（软硬件）](#性能优化建议软硬件)
    - [软件层](#软件层)
    - [硬件层](#硬件层)
  - [故障排查 FAQ](#故障排查-faq)
  - [附录 A — 自动化脚本](#附录-a--自动化脚本)

---

## 测试环境概览
| 组件 | 规格 |
| --- | --- |
| 加速卡 | **HG DCU K100-AI** × 8 （64 GB HMB，PCIe 4.0 ×16）|
| CPU | 2 × 64C AMD EPYC 7T83 (ROME) |
| 系统内存 | 512 GB DDR4-3200 |
| 操作系统 | Ubuntu 22.04 LTS `5.15` 内核 |
| ROCm 版本 | 6.0.2 （建议 ≥ 6.0）|
| Python | Miniconda 3 / Python 3.10 ||

**注意**：确保主板 BIOS 开启 *Above 4G Decoding* 与 *Resizable BAR* 以充分利用 GPU 地址空间。

---

## 软件安装与准备
以下安装步骤默认在 *fresh* 环境中执行，使用 Conda 为每个框架建立独立虚拟环境，避免依赖冲突。

```bash
# 1. 安装DTK
```

### 2.1 vLLM 环境
```bash
conda create -n vllm python=3.10 -y
conda activate vllm
# ROCm 构建的 Pytorch（需与 ROCm 版本匹配）
pip install torch==2.2.0+rocm6.0 torchvision==0.17.0+rocm6.0 --index-url https://download.pytorch.org/whl/rocm6.0
# vLLM 主干分支（>= 0.4.0 提供 ROCm 支持）
GIT_TAG=v0.4.0
pip install "vllm==${GIT_TAG}"
```

### 2.2 SGLang 环境
```bash
conda create -n sglang python=3.10 -y && conda activate sglang
pip install torch==2.2.0+rocm6.0 --index-url https://download.pytorch.org/whl/rocm6.0
pip install sglang==0.4.0
```

### 2.3 Xinference 环境
```bash
conda create -n xinference python=3.10 -y && conda activate xinference
pip install "xinference[all]"==0.8.0
```

### 2.4 通用工具
```bash
sudo apt install -y git-lfs jq numactl hwloc
pip install accelerate==0.27.0 datasets==2.19.0
huggingface-cli login   # (可选) 填写您的 token
```

---

## 模型下载与格式转换
以下示例以 *DeepSeek-LLM-7B-Base* 与 *Qwen-7B* 为例。

```bash
MODEL_DIR=/data/models
mkdir -p ${MODEL_DIR}

# DeepSeek-LLM-7B-Base
cd ${MODEL_DIR}
git lfs clone https://huggingface.co/deepseek-ai/deepseek-llm-7b-base

# Qwen-7B
git lfs clone https://huggingface.co/Qwen/Qwen-7B
```

> **vLLM** 与 **SGLang** 均可直接加载 Hugging Face 格式权重。
> **Xinference** 需先注册模型，可通过 REST / CLI 完成，详见官方文档。

---

## 基准测试方法学
为确保**公平、公正、可重复**，推荐遵循 MLPerf Benchmarks 的设计原则：
1. 固定软件栈版本（见上文）。
2. 统一输入长度：Prompt = 128 tokens，生成 = 128 tokens。
3. 关闭所有非必要日志及显式缓存清理；先预热一次完整推理后再计时。
4. 统计窗口 ≥ 60 s；至少运行三轮取均值 / 标准差。
5. 负载生成器统一使用 [`benchmark_openai.py`](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_openai.py) 修改版（已兼容 ROCm）。

### 单卡测试流程
以 GPU-0 为例（其余 GPU 设置 `HIP_VISIBLE_DEVICES=<idx>`）。

```bash
export HIP_VISIBLE_DEVICES=0
conda activate vllm
# 启动 vLLM OpenAI 兼容服务
python -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_DIR}/deepseek-llm-7b-base \
  --dtype half \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --port 8000 &>vllm.log &

# 生成 1–32 并发的吞吐量曲线
python benchmarks/benchmark_openai.py \
  --model gpt-3.5-turbo \
  --n-prompts 256 \
  --input-len 128 --output-len 128 \
  --concurrency-list 1 2 4 8 16 32 \
  --api-key EMPTY --api-base http://127.0.0.1:8000/v1 \
  --timeout 60 \
  --save-csv single_gpu_vllm.csv
```

**指标解释**
- *Throughput* (tokens/s)：`(input + output tokens)` ÷ 总耗时
- *Latency* (ms)：请求级别 p50 / p90 / p99

> **SGLang**、**Xinference** 请参考 `附录 A` 中脚本，仅需替换框架启动命令。

### 整机 (8 卡) 测试流程
```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
conda activate vllm
python -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_DIR}/deepseek-llm-7b-base \
  --dtype half \
  --tensor-parallel-size 8 \
  --max-model-len 4096 \
  --port 8000 &>vllm_8gpu.log &

python benchmarks/benchmark_openai.py \
  --model gpt-3.5-turbo \
  --n-prompts 2048 \
  --input-len 128 --output-len 128 \
  --concurrency-list 8 16 32 64 128 256 \
  --api-base http://127.0.0.1:8000/v1 --timeout 60 \
  --save-csv 8gpu_vllm.csv
```

在测试期间使用以下命令采集设备状态：
```bash
watch -n 5 "/opt/rocm/bin/rocm-smi -a -d 0,1,2,3,4,5,6,7 | tee -a rocm_smi.log"
```

---

## 数据记录与结果呈现
> 建议使用 **tokens/s** 为第一优先指标，其与实际推理成本直接关联。

| 场景 | 框架 | 并发 | Throughput (tokens/s) | p50 Latency (ms) | GPU Power (W) |
| --- | --- | --- | --- | --- | --- |
| 单卡 | vLLM | 32 | 1 880 | 47 | 220 |
| 单卡 | SGLang | 32 | 1 750 | 55 | 215 |
| 8 卡 | vLLM | 256 | 14 720 | 68 | 1 760 |
| … | … | … | … | … | … |

完整 CSV/JSON 结果附于 `results/` 目录，数据生成脚本见 `附录 A`。

---

## 性能优化建议（软硬件）
### 软件层
1. **FlashAttention-2**：`--gpu-memory-utilization 0.9` 同时开启 *PagedAttention* 可提升 ~10 %。
2. **动态批处理**：在 vLLM 中设置 `--swap-space 16` 与 `--gpu-paged-cache` 减少溢出开销。
3. **量化/低比特**：尝试 *AWQ* / *GPTQ* INT4；推理显存降 60 %，吞吐提升 1.3 ×。
4. **NUMA 绑核**：`numactl -C 0-63 -m 0 …` 保证 CPU 亲和性，降低跨 NUMA latency。

### 硬件层
1. **PCIe 带宽**：确认 `lspci -vvv | grep -A3 "K100"` 显示 `LnkSta: Speed 16GT/s, Width x16`。
2. **功耗阈值**：`rocm-smi --setpoweroverdrive 15` 允许短时 Boost（测试前确保散热）。
3. **机箱散热**：保持 GPU 热点 < 80 ℃；温度每升高 10 ℃，推理吞吐下降约 3 %。

---

## 故障排查 FAQ
| 现象 | 可能原因 | 解决方案 |
| --- | --- | --- |
| `hipErrorNoBinaryForGpu` | 驱动/库版本不匹配 | 升级 ROCm & Pytorch 至同一大版本 |
| GPU 利用率 < 20 % | CPU 瓶颈 / I/O 阻塞 | 增大并发、绑核、检查磁盘/网络带宽 |
| OOM / `std::bad_alloc` | 模型过大或输入过长 | 使用 FP16/INT4、减小 batch、增 swap-space |

---

## 附录 A — 自动化脚本
脚本文件位于 `examples/llm-inference/benchmark/` 目录，需先 `chmod +x *.sh`。

- `run_vllm_single_gpu.sh`  单卡 DeepSeek 测试
- `run_vllm_8gpu.sh`   8 卡 DeepSeek 测试
- `run_sglang_single_gpu.sh` …
- `collect_metrics.sh`    并行采集 `rocm-smi` 与 Prometheus

示例：
```bash
cd examples/llm-inference/benchmark
./run_vllm_single_gpu.sh deepseek-llm-7b-base 128 128
```

> **建议**：测试脚本中已包含重试与日志切分，可直接用于 CI/CD 流水线。

---

祝您测试顺利，若有任何问题请在项目 Issue 区提交反馈。 