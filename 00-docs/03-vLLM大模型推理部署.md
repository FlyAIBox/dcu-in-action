## 大模型推理

### 什么是“推理”？为什么需要专门的推理系统

- 定义（简单话）：**推理 = 用已经训练好的大模型去生成答案或文本**。训练在一个阶段，推理是把模型放进生产环境实时使用。
- 为什么不直接用“训练时”的代码？训练注重梯度与更新，推理注重**速度、并发、成本和稳定性**。
- 核心工程冲突：**想要低延迟（快速单次回应）**与**想要高吞吐（便宜地同时处理很多请求）**往往冲突，需要中间的工程折中与优化。
   （适合口头类比：训练像做菜的配方，推理像餐馆在高峰期连续出菜，得有高效流水线）

### 大模型推理工程必须解决的 5 个“真问题”（初级版）

1. **上下文（KV）爆炸**：模型每生成 token 都会产生 key/value，长上下文会占满显存。
2. **延迟 vs 吞吐**：把很多请求放一堆（批处理）能提高 GPU 利用率，但会拉高单个请求的等待时间。
3. **显存碎片与管理**：多会话、多长度请求导致显存不可预测。
4. **多模型和多硬件兼容**：不同模型（如 Qwen、Mistral、LLaMA）和不同 GPU/CUDA 版本之间有兼容性差异。
5. **生成输出的格式控制**（比如一定要返回 JSON）：直接输出文本常有“格式错误”。

> - **高效 KV（key/value）缓存管理**：避免内存爆炸、实现长上下文。
> - **连续批处理（continuous batching）与调度**：在保持高吞吐的同时尽量不增加单请求延迟。
> - **快速执行路径**：利用优化 CUDA kernel、FlashAttention、CUDA/HIP Graph。
> - **解码策略多样化**：并行采样、束搜索、speculative decoding（推测性解码）等以兼顾速度与质量。
> - **可运维性**：OpenAI 兼容接口、流式输出、结构化输出支持、监控/日志。

### vLLM 的核心架构与技术关键

vLLM 是一个快速且易于使用的库，专为大型语言模型 (LLM) 的推理和部署而设计。

**vLLM 的核心特性包括：**

- 最先进的服务吞吐量
- 使用 **PagedAttention** 高效管理注意力键和值的内存
- 连续批处理传入请求
- 使用 CUDA/HIP 图实现快速执行模型
- 量化： [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), INT4, INT8, 和 FP8
- 优化的 CUDA 内核，包括与 FlashAttention 和 FlashInfer 的集成
- 推测性解码
- 分块预填充

**vLLM 的灵活性和易用性体现在以下方面：**

- 无缝集成流行的 HuggingFace 模型
- 具有高吞吐量服务以及各种解码算法，包括*并行采样*、*束搜索*等
- 支持张量并行和流水线并行的分布式推理
- 流式输出
- 提供与 OpenAI 兼容的 API 服务器
- 支持 NVIDIA GPU、AMD CPU 和 GPU、Intel CPU 和 GPU、PowerPC CPU、TPU 以及 AWS Neuron
- 前缀缓存支持
- 支持多 LoRA

#### PagedAttention（关键点）

- 思路：**将 KV 缓存视为“分页”管理（类似操作系统分页）**，只在需要时把“页”载入 GPU 内存／快速访问区域，从而减少连续长上下文对显存的压力并降低碎片化。
- 效果：在长上下文或多会话场景下显著降低内存占用并提高吞吐。官方 强调 PagedAttention 为其高效管理 KV 的核心技术。[GitHub](https://github.com/vllm-project/vllm/tree/v0.8.5)

#### 连续批处理（Continuous batching）与 Unified Scheduler（v0.8.x 行为）

- vLLM 通过把进来的短请求与长 prefills 合理打包，构建连续批处理流水线来提升 GPU 利用率，同时结合调度器细粒度控制以平衡延迟/吞吐。chunked prefill 是将长 prefills 拆 chunk 与 decode 请求共同批处理以提升效率。[GitHub](https://github.com/vllm-project/vllm/tree/v0.8.5)[docs.vllm.ai](https://docs.vllm.ai/en/latest/configuration/optimization.html?utm_source=chatgpt.com)

#### Speculative Decoding（推测性解码）

- 用轻量/低延迟“猜测模型”先预测若干 token，然后用主模型验证，减少主模型实际执行 token 数。适合高吞吐场景（需要注意一致性与回退逻辑）。vLLM 支持多种 speculative 方法，且社区有基于 Arctic 等方案的加速实践。[GitHub](https://github.com/vllm-project/vllm/tree/v0.8.5)[snowflake.com](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/?utm_source=chatgpt.com)

#### Chunked Prefill 与 Prefix/自动缓存

- chunked prefill 把一次性很大的上下文分块处理，避免一次性巨量计算/内存占用，并能与 decode 请求并行化调度；同时支持 prefix/prefill 缓存以重复利用 prompt 开头的计算。官方文档与 issue 讨论说明这一点在 v0.8.5 中默认/常用（V1 引擎延续并改进）。[docs.vllm.ai](https://docs.vllm.ai/en/latest/configuration/optimization.html?utm_source=chatgpt.com)[GitHub](https://github.com/vllm-project/vllm/issues/20531?utm_source=chatgpt.com)

#### 3.5 量化与内核优化

- 支持 GPTQ、AWQ、INT4/INT8/FP8 等多种量化方案，并集成 FlashAttention/FlashInfer 的优化 kernel 来减少内存占用与提升速度（尤其是大型 transformer 层）

## 海光DCU大模型推理

------

### 1. 准备工作：环境配置

在进行大模型性能测试之前，首先需要搭建好必要的运行环境。

- ### 使用指定镜像：

  我们将使用 

  `image.sourcefind.cn:5000/dcu/admin/base/vllm:0.8.5-ubuntu22.04-dtk25.04.1-rc5-das1.6-py3.10-20250711` 这个 Docker 镜像来创建运行容器。这个镜像包含了 vLLM 库以及其他必要的依赖，简化了环境配置过程 。

- ### 创建 Docker 容器：

  通过运行以下 `docker run` 命令来创建并启动一个名为 `llm-benchmark` 的容器 2。这个命令配置了容器的名称、数据卷映射、IPC 模式、网络模式、设备访问以及内存共享等参数，以确保 vLLM 服务能够正常运行并高效利用硬件资源 。

  ```Bash
  # 前台启动/交互方式
  docker run -it \
  --name=llm-benchmrk \
  -v /data:/data \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /usr/local/hyhal:/usr/local/hyhal:ro \
  --ipc=host \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/mkfd \
  --device=/dev/dri \
  --shm-size=64G \
  --security-opt seccomp=unconfined \
  --group-add video \
  --privileged \
  image.sourcefind.cn:5000/dcu/admin/base/vllm:0.8.5-ubuntu22.04-dtk25.04.1-rc5-das1.6-py3.10-20250711 \
  /bin/bash
  
  # 后台启动/
  docker run -d \
  --name=llm-benchmrk \
  -v /data:/data \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /usr/local/hyhal:/usr/local/hyhal:ro \
  --ipc=host \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/mkfd \
  --device=/dev/dri \
  --shm-size=64G \
  --security-opt seccomp=unconfined \
  --group-add video \
  --privileged \
  image.sourcefind.cn:5000/dcu/admin/base/vllm:0.8.5-ubuntu22.04-dtk25.04.1-rc5-das1.6-py3.10-20250711 \
  sleep infinity
  ```
  
  **解释各个参数：**
  
  - `-it`: 保持标准输入打开并分配一个伪 TTY，通常用于交互式会话 。
  - `--name=llm-benchmark`: 为容器指定一个名称，方便管理和识别 。
  - `-v /data:/data`: 将宿主机的 `/data` 目录挂载到容器的 `/data` 目录，用于模型存储和数据交换 。
  - `-v /opt/hyhal:/opt/hyhal:ro`: 以只读方式挂载宿主机的 `/opt/hyhal` 目录到容器 。
  - `-v /usr/local/hyhal:/usr/local/hyhal:ro`: 以只读方式挂载宿主机的 `/usr/local/hyhal` 目录到容器 。
  - `--ipc=host`: 允许容器与宿主机共享 IPC（进程间通信）命名空间，这对于某些高性能应用（如深度学习框架）很有用 
  - `--network=host`: 容器使用宿主机的网络命名空间，可以直接访问宿主机的网络接口 10。
  - `--device=/dev/kfd`, `--device=/dev/mkfd`, `--device=/dev/dri`: 挂载 ROCm（AMD GPU）相关的设备文件到容器，以便容器可以访问 GPU。
  - `--shm-size=64G`: 设置共享内存的大小为 64GB，这对于大模型推理非常重要，可以避免内存不足的问题 
  - `--security-opt seccomp=unconfined`: 关闭 Seccomp 安全限制，可能在某些情况下需要，但会降低安全性 。
  - `--group-add video`: 将容器内的用户添加到 `video` 组，以便访问 GPU 设备 。
  - `--privileged`: 授予容器扩展的权限，使其可以访问宿主机的所有设备。慎用此选项，因为它会降低安全性 。
  - `image.sourcefind.cn:5000/dcu/admin/base/vllm:0.8.5-ubuntu22.04-dtk25.04.1-rc5-das1.6-py3.10-20250711`: 指定用于创建容器的 Docker 镜像 。
  - `/bin/bash`: 在容器启动后执行的命令，这里是启动一个 bash shell 。

------

### 2. 启动 vLLM 服务

在容器内部，需要启动 vLLM 服务来加载并运行大模型 。这里以微调后的 `deepseek32b-medical` 模型为例 。

- 进入容器内部

  ```bash
  docker exec -it llm-benchmrk /bin/bash
  ```
  
- 配置环境变量并启动 vLLM 服务：

  以下脚本设置了一系列环境变量，然后启动了 vLLM 服务 。这些环境变量对于优化 vLLM 在多 GPU 环境下的性能至关重要 。

  ```Bash
  # 环境变量设置
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # 指定使用8块DCU
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # 指定使用8块DCU
  
  # 部署微调后台的模型
  # vllm serve /data/model/deepseek32b-medical --trust-remote-code --dtype float16 --max-model-len 32768 -tp 8 --gpu-memory-utilization 0.80 --block-size 32 --port 8010
  
  --------------------------------------------------------------------------------------
  --------------------------------------------------------------------------------------
  
  # 部署deepseek-671b-awq 模型
  
  # 下载模型
  pip install modelscope
  modelscope download --model cognitivecomputations/DeepSeek-R1-awq  --local_dir /data
  
  # 配置环境变量并启动 vLLM 服务
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export ALLREDUCE_STREAM_WITH_COMPUTE=1
  export NCCL_MIN_NCHANNELS=16
  export NCCL_MAX_NCHANNELS=16
  export VLLM_NUMA_BIND=1
  export VLLM_RANK0_NUMA=0
  export VLLM_RANK1_NUMA=0
  export VLLM_RANK2_NUMA=0
  export VLLM_RANK3_NUMA=0
  export VLLM_RANK4_NUMA=0
  export VLLM_RANK5_NUMA=0
  export VLLM_RANK6_NUMA=0
  export VLLM_RANK7_NUMA=0
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export VLLM_PCIE_USE_CUSTOM_ALLREDUCE=1
  export VLLM_FUSED_MOE_CHUNK_SIZE=16384
  export LMSLIM_USE_LIGHTOP=0
  export W4A16_MOE_CUDA=1
  
  vllm serve /data/model/cognitivecomputations/DeepSeek-R1-awq --trust-remote-code --dtype float16 --max-model-len 32768 -tp 8 -q moe_wna16 --gpu-memory-utilization 0.90 --block-size 64 --port 8010
  ```

#### 日志解析

1. **参数接受与处理**: vLLM成功识别并处理了你提供的所有参数，包括：
   - `--model /data/model/deepseek32b-medical`
   - `--trust-remote-code`
   - `--dtype float16`
   - `--max-model-len 32768`
   - `-tp 8`
   - `--gpu-memory-utilization 0.80`
   - `--block-size 32`
   - `--port 8010`
2. **平台和后端识别**:
   - `Automatically detected platform rocm.`: 自动检测到了你的硬件平台是 **ROCm**。
   - `Using ROCmFlashAttention backend.`: 确认vLLM正在使用专门为ROCm优化的 **FlashAttention** 后端。
3. **内存和资源分配**:
   - `the current vLLM instance can use total_gpu_memory (63.98GiB) x gpu_memory_utilization (0.80) = 51.19GiB`: 成功按照你设定的`0.80`比例计算了可用显存，即 **51.19 GiB**。
   - `model weights take 7.71GiB; non_torch_memory takes 1.78GiB; PyTorch activation peak memory takes 2.20GiB; the rest of the memory reserved for KV Cache is 39.51GiB.`: 详细列出了内存分配情况，其中为 **KV Cache** 预留了 **39.51 GiB**。
   - `# rocm blocks: 40178, # CPU blocks: 4096`: 根据`block-size`和可用显存，计算并分配了GPU和CPU的块（blocks）数量。
4. **模型预热和图捕获**:
   - `Capturing cudagraphs for decoding.`: vLLM正在为解码过程捕获 **CUDA图**，这是一种性能优化技术，旨在减少运行时开销。
   - `Graph capturing finished in 28 secs, took 0.35 GiB`: **图捕获成功完成**，耗时28秒，额外占用了0.35 GiB的显存。
5. **服务启动**:
   - `init engine (profile, create kv cache, warmup model) took 48.45 seconds`: 整个引擎初始化过程耗时 **48.45秒**。
   - `Starting vLLM API server on http://0.0.0.0:8010`: **服务已成功启动**，现在你可以通过 `http://0.0.0.0:8010` 这个地址访问API服务了。
6. **路由信息**:
   - 日志最后列出了所有可用的API路由，这证明了服务已经完全准备就绪。

------

#### 3. 参数解释

##### GPU设备配置
- `HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`: 指定使用所有8个DCU设备进行推理

##### 通信优化
- `ALLREDUCE_STREAM_WITH_COMPUTE=1`: 启用计算与通信重叠，提高多GPU并行效率
- `NCCL_MIN_NCHANNELS=16` / `NCCL_MAX_NCHANNELS=16`: 设置NCCL通信通道数为16，优化GPU间数据传输带宽

##### NUMA绑定优化
- `VLLM_NUMA_BIND=1`: 启用NUMA绑定，优化内存访问局部性
- `VLLM_RANK0_NUMA=0` 到 `VLLM_RANK7_NUMA=0`: 将所有GPU工作进程绑定到NUMA节点0，确保内存访问一致性

##### vLLM特定优化
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`: 使用spawn方式创建工作进程，避免CUDA上下文冲突
- `VLLM_PCIE_USE_CUSTOM_ALLREDUCE=1`: 启用自定义PCIe AllReduce优化，提高GPU间通信效率
- `VLLM_FUSED_MOE_CHUNK_SIZE=16384`: 设置MoE模型的融合块大小为16KB，优化专家网络计算

##### 模型特定配置
- `LMSLIM_USE_LIGHTOP=0`: 禁用LightOp优化（可能与当前模型不兼容）

##### vLLM服务参数

该命令用于使用 **vLLM** 框架启动一个大型语言模型服务。下面是每个参数的详细解释：

###### 模型和基础配置

- `vllm serve /data/model/deepseek32b-medical`
  - **`vllm serve`**: 这是启动vLLM服务器的命令。
  - `/data/model/deepseek32b-medical`: 指定要加载的**模型路径**。这个路径必须包含模型的配置文件（`config.json`）、权重文件（`model-*.safetensors`）和分词器文件等。
- `--trust-remote-code`
  - **`--trust-remote-code`**: 允许vLLM执行模型仓库中自定义的Python代码（例如在`modeling_qwen2.py`等文件中）。这对于加载某些需要特殊逻辑的模型（如Qwen2）是必要的。
- `--dtype float16`
  - **`--dtype float16`**: 指定模型权重的**数据类型**。`float16`（半精度浮点数）是一种常用的选择，因为它能显著减少显存占用，同时在大多数情况下对模型性能影响不大。这对于内存有限的GPU尤其重要。

###### 性能和内存优化

- `--max-model-len 32768`
  - **`--max-model-len`**: 设置模型可以处理的**最大序列长度**。这个值越大，模型能处理的上下文越长，但同时也会占用更多的显存，因为KV Cache（键值缓存）的大小与此成正比。将其设置为`32768`意味着你可以处理非常长的输入和输出序列。
- `-tp 8`
  - **`-tp`**: 这是`--tensor-parallel-size`的简写，表示**张量并行**的等级。`tp 8`意味着模型权重将分成8份，分别加载到8个GPU上。这对于在多GPU设备上加载像**Deepseek-32B**这样的大模型是必需的。
- `--gpu-memory-utilization 0.80`
  - **`--gpu-memory-utilization`**: 设置vLLM可以使用的**显存比例**，范围从0到1。`0.80`表示vLLM将限制自己只使用GPU总显存的80%。剩下的20%显存会留给系统、PyTorch的开销和其他可能运行的进程。这个参数对于避免因内存不足而导致的崩溃至关重要。
- `--block-size 32`
  - **`--block-size`**: vLLM的**KV Cache**被划分为固定大小的块（Block），这个参数定义了每个块中可以存储的token数量。`32`意味着每个块可以存储32个token的键和值。较小的块大小可以减少内存碎片，而较大的块大小可以提高效率，但必须与你使用的硬件和vLLM版本兼容。
- `--port 8010`
  - **`--port`**: 指定API服务器的**监听端口**。客户端（如`curl`命令）将通过这个端口与vLLM服务进行通信。`8010`是这个服务的端口号。



## 模型测试

```Bash
curl -k -X POST "http://172.16.1.3:8010/v1/chat/completions" \
 -H "Content-Type: application/json" \
 -d '{
  "model": "/data/model/deepseek32b-medical",
  "messages": [
   {"role": "system", "content": "你是一位专业的医疗AI助手，请根据提供的临床症状和检查结果，进行严谨的医学诊断推理。"},
   {"role": "user", "content": "一名40岁男性突发上腹剧痛伴恶心半小时，查体显示上腹部深压痛，但Murphy征阴性，血常规显示白细胞计数略增，尿便常规正常，考虑其最可能的诊断是什么？ 请逐步推理并给出答案。"}
  ],
  "max_tokens": 3000,
  "temperature": 0.5,
  "top_p": 0.95
 }'
 
 
 curl -k -X POST "http://172.16.1.3:8010/v1/chat/completions" \
 -H "Content-Type: application/json" \
 -d '{
  "model": "/data/model/cognitivecomputations/",
  "messages": [
   {"role": "system", "content": "你是一位专业的医疗AI助手，请根据提供的临床症状和检查结果，进行严谨的医学诊断推理。"},
   {"role": "user", "content": "一名40岁男性突发上腹剧痛伴恶心半小时，查体显示上腹部深压痛，但Murphy征阴性，血常规显示白细胞计数略增，尿便常规正常，考虑其最可能的诊断是什么？ 请逐步推理并给出答案。"}
  ],
  "max_tokens": 3000,
  "temperature": 0.5,
  "top_p": 0.95
 }'
```

------

### 命令参数解释

我根据你的需求和通常的医疗问答场景对命令进行了调整，主要变化如下：

- **`URL`**：已更新为你的目标地址 `http://172.16.1.3:8010/v1/chat/completions`。
- **`model`**：模型名称已更改为你在日志中看到的名字 `deepseek32b-medical`。
- **`messages`**：
  - **`system`** 角色：我为你添加了一个更适合医疗问答的系统提示，要求模型扮演一位专业的医疗AI助手，进行严谨的诊断推理。这有助于引导模型给出更专业的回答。
  - **`user`** 角色：已更新为你的具体问题，即一名40岁男性患者的临床病例。
- **`max_tokens`**：我将最大生成 token 数调整为 `500`。对于复杂的推理和诊断，`100` 个 token 可能不够用，`500` 能确保模型有足够的空间进行详细的推理和结论。
- **`temperature`** 和 **`top_p`**：我将 `temperature` 调整为 `0.5`，`top_p` 调整为 `0.95`。
  - **`temperature`** 较低（0.5）会让模型生成的结果更稳定、更具确定性，这对于医学诊断这种需要严谨性的任务非常重要。
  - **`top_p`** 较高（0.95）则允许模型在更广泛的词汇范围内选择，确保推理过程不会因为过度保守而遗漏关键信息。
- **`-k`** 参数：保留了 `-k` 参数，这在处理自签名证书等情况时很有用。如果你的服务没有使用HTTPS，这个参数可以移除，但保留它通常不会影响命令的执行。
