# 海光DCU大模型性能测试指南

本指南旨在为大模型技术初学者提供一份详细的性能测试操作步骤，确保每一步都清晰明了，易于理解和执行。

------

## 1. 准备工作：环境配置

在进行大模型性能测试之前，首先需要搭建好必要的运行环境。

- ### 使用指定镜像：

  我们将使用 

  `image.sourcefind.cn:5000/dcu/admin/base/vllm:0.8.5-ubuntu22.04-dtk25.04.1-rc5-das1.6-py3.10-20250711` 这个 Docker 镜像来创建运行容器。这个镜像包含了 vLLM 库以及其他必要的依赖，简化了环境配置过程 。

- ### 创建 Docker 容器：

  通过运行以下 `docker run` 命令来创建并启动一个名为 `llm-benchmark` 的容器 2。这个命令配置了容器的名称、数据卷映射、IPC 模式、网络模式、设备访问以及内存共享等参数，以确保 vLLM 服务能够正常运行并高效利用硬件资源 。

  ```Bash
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

## 2. 启动 vLLM 服务

在容器内部，需要启动 vLLM 服务来加载并运行大模型 。这里以 `ds-671b-awq` 模型为例 。

- 下载模型

  ```bash
  pip install modelscope
  modelscope download --model cognitivecomputations/DeepSeek-R1-awq  --local_dir /data
  ```
  
- 配置环境变量并启动 vLLM 服务：

  以下脚本设置了一系列环境变量，然后启动了 vLLM 服务 。这些环境变量对于优化 vLLM 在多 GPU 环境下的性能至关重要 。

  ```Bash
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
  vllm serve /data/model/cognitivecomputations/DeepSeek-R1-awq --trust-remote-code  --dtype float16 -q moe_wna16 --gpu-memory-utilization 0.90 --tensor-parallel-size 8 --max-model-len 32768 --block-size 64 --max-num-seqs 128  --speculative_config '{""num_speculative_tokens"": 3}'
  
  
  # 模型优化前的明立
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
  vllm serve /data/model/cognitivecomputations/DeepSeek-R1-awq --trust-remote-code --dtype float16 --max-model-len 32768 -tp 8 -q moe_wna16 --gpu-memory-utilization 0.90 --block-size 64 
  
  vllm serve /data/model/cognitivecomputations/DeepSeek-R1-awq --trust-remote-code --dtype float16 --max-model-len 32768 -tp 8 -q moe_wna16 --gpu-memory-utilization 0.90 --block-size 64 --port 8010
  ```


这个vLLM服务启动脚本包含了完整的DCU环境优化配置，以下是详细的参数解释：

### GPU设备配置
- `HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`: 指定使用所有8个DCU设备进行推理

### 通信优化
- `ALLREDUCE_STREAM_WITH_COMPUTE=1`: 启用计算与通信重叠，提高多GPU并行效率
- `NCCL_MIN_NCHANNELS=16` / `NCCL_MAX_NCHANNELS=16`: 设置NCCL通信通道数为16，优化GPU间数据传输带宽

### NUMA绑定优化
- `VLLM_NUMA_BIND=1`: 启用NUMA绑定，优化内存访问局部性
- `VLLM_RANK0_NUMA=0` 到 `VLLM_RANK7_NUMA=0`: 将所有GPU工作进程绑定到NUMA节点0，确保内存访问一致性

### vLLM特定优化
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`: 使用spawn方式创建工作进程，避免CUDA上下文冲突
- `VLLM_PCIE_USE_CUSTOM_ALLREDUCE=1`: 启用自定义PCIe AllReduce优化，提高GPU间通信效率
- `VLLM_FUSED_MOE_CHUNK_SIZE=16384`: 设置MoE模型的融合块大小为16KB，优化专家网络计算

### 模型特定配置
- `LMSLIM_USE_LIGHTOP=0`: 禁用LightOp优化（可能与当前模型不兼容）
- `W4A16_MOE_CUDA=1`: 启用4位权重16位激活的MoE CUDA优化

## vLLM服务参数

### 基础配置
- `/data/model/cognitivecomputations/DeepSeek-R1-awq`: 模型路径
- `--trust-remote-code`: 允许执行模型仓库中的自定义代码
- `--dtype float16`: 使用FP16精度，节省显存并提升速度

### 量化配置
- `-q moe_wna16`: 使用MoE权重量化方案，4位权重16位激活

### 性能配置
- `--gpu-memory-utilization 0.90`: 使用90%的GPU显存
- `--tensor-parallel-size 8`: 8卡张量并行
- `--max-model-len 32768`: 支持最大32K上下文长度
- `--block-size 64`: KV缓存块大小64
- `--max-num-seqs 128`: 最大并发序列数128

### 推测解码
- `--speculative_config '{"num_speculative_tokens": 3}'`: 启用推测解码，预测3个token提升生成速度

这个配置针对DeepSeek-R1-AWQ模型在8卡DCU环境下进行了全面优化，可以实现高吞吐量的推理服务。

- 其他模型类型：

  文档中还提到了 `ds-671b-int8` 模型 ，以及参考文档《Deepseek671b/qwen3 VLLM使用部署方法说明（对外版本）》 41，这些可以作为扩展阅读，了解不同量化类型模型的部署方法。

------



```bash
curl -k -X POST "http://127.0.0.1:8010/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-VL-72B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "分析图中场景并生成详细描述"},
          {
            "type": "image_url",
            "image_url": {
              "url": "http://8.154.30.196:5001/uploads/202506271354/1751003672427-80c84e67-d399-4c92-ae11-67fa9cc7b970.JPG",
              "detail": "high"
            }
          }
        ]
      }
    ],
    "max_tokens": 1024
  }'

这个也是，去掉前面的api前缀
```





### **3. 性能测试**

vLLM 服务启动后，就可以进行性能测试了 。

- 下载并解压测试工具：

  首先，下载 

  `online.zip` 文件并解压，其中包含了性能测试所需的脚本和工具 。

- 修改 test.sh 脚本：

  打开 

  `test.sh` 脚本文件 ，按照启动的 vLLM serve 以及测试需求，修改为实际使用的参数 

  下面的图片展示了 

  `test.sh` 脚本的关键修改部分 。

  ```bash
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  data_type=batch,prompt_tokens,completion_tokens,TOTAL_THROUGHPUT(toks/s),generate_throughput(toks/s),TTFT(ms),TPOT(ms),ITL(MS) > r1-awq-0705.csv
  pairs=("512 512" "1024 1024")
  model_path="/data/model/cognitivecomputations/DeepSeek-R1-awq"
  tp=8
  
  data_type="float16"
  mkdir -p ./log/
  for batch in 2 4 6 8 10 16 20 24 32 64 do
      for pair in ${pairs[@]}; do
          prompt_tokens=$(echo $pair%* | cut -d' ' -f1)
          completion_tokens=$(echo $pair#* | cut -d' ' -f2)
          log_path="log/vllm_${model}_${batch}_${prompt_tokens}_${completion_tokens}_${tp}.log"
          touch $log_path
          # benchmark_throughput.py
          python benchmark_serving.py \
          --backend openai \
          --port 8000 \
          --model $model_path \
          --trust-remote-code \
          --dataset-name random \
          --ignore-eos \
          --random-input-len $prompt_tokens \
          --random-output-len $completion_tokens \
          --num-prompts $batch \
          2>&1 | tee $log_path
  
          #metric
          E2E_TIME=`grep "benchmark duration" $log_path | awk -F' ' '{print $4}'`
          REQ_THROUGHPUT=`grep "Request throughput" $log_path | awk -F' ' '{print $4}'`
          GEN_THROUGHPUT=`grep "Output throughput" $log_path | awk -F' ' '{print $5}'`
          TOTAL_THROUGHPUT=`grep "Total Token" $log_path | awk -F' ' '{print $5}'`
          TTFT=`grep "Mean TTFT" $log_path | awk -F' ' '{print $4}'`
          TPOT=`grep "Mean TPOT" $log_path | awk -F' ' '{print $4}'`
          ITL=`grep "Mean ITL" $log_path | awk -F' ' '{print $4}'`
          P99_ITL=`grep "P99 ITL" $log_path | awk -F' ' '{print $4}'`
          P99_TTFT=`grep "P99 TTFT" $log_path | awk -F' ' '{print $4}'`
          P99_TPOT=`grep "P99 TPOT" $log_path | awk -F' ' '{print $4}'`
          echo "$tp,$data_type,$batch,$prompt_tokens,$completion_tokens,$TOTAL_THROUGHPUT,$GEN_THROUGHPUT,$TTFT,$TPOT,$ITL" >> r1-awq-0705.csv
      done
  done
  ```

  **请注意修改以下关键参数：**

  - `data_type`: 输出 CSV 文件头，记录了性能指标。
  - `pairs`: 定义了不同的 `prompt_tokens` 和 `completion_tokens` 组合，例如 `("512 512" "1024 1024")` 表示测试输入 512 输出 512 和输入 1024 输出 1024 两种场景。
  - `model_path`: 确保与你启动 vLLM 服务时使用的模型路径一致。
  - `tp`: 与你启动 vLLM 服务时设置的 `--tensor-parallel-size` 参数一致，这里是 `8`。
  - `--port`: 如果 vLLM 服务监听的端口不是 8000，请修改为实际端口。
  - `--random-input-len` 和 `--random-output-len`: 这些参数控制了测试时生成的随机输入和输出 token 长度，确保与 `pairs` 变量中的设置一致。
  - `r1-awq-0705.csv`: 这是输出结果的 CSV 文件名，可以根据需要进行修改。

通过上述步骤，你就可以成功地进行大模型性能测试，并得到详细的性能指标数据。

------

## 模型推理压测指标汇总

| 指标名称                    | 英文缩写/原始名称                       | 解释                                                         | 实际意义                                                     |
| --------------------------- | --------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **端到端时间**              | `E2E_TIME` / `benchmark duration`       | 整个基准测试运行的总时间，从开始到结束。                     | 衡量完成所有请求和生成所有输出所需的总时长，时间越短表示整体执行效率越高。 |
| **请求吞吐量**              | `REQ_THROUGHPUT` / `Request throughput` | 单位时间内系统处理的请求数量（通常是每秒请求数）。           | 反映模型服务处理并发请求的能力，数值越高表示服务器能处理更多用户请求。 |
| **生成吞吐量**              | `GEN_THROUGHPUT` / `Output throughput`  | 单位时间内模型生成的 token 数量（通常是每秒 token 数），也称**解码吞吐量**或**生成速度**。 | 衡量模型实际生成文本的速度，数值越高意味着用户可以更快地获得完整的响应。 |
| **总吞吐量**                | `TOTAL_THROUGHPUT` / `Total Token`      | 每秒处理的总 token 数量，包括输入 prompt tokens 和生成的 completion tokens。 | 衡量模型整体处理能力的重要指标，综合反映了输入处理和输出生成的效率。 |
| **平均首次 token 时间**     | `TTFT` / `Mean TTFT`                    | 从发送请求到接收到第一个 token 所需的平均时间（毫秒）。      | 对用户体验影响很大，决定用户看到第一个响应所需的时间，时间越短用户感觉响应越快。 |
| **平均每 token 时间**       | `TPOT` / `Mean TPOT`                    | 生成后续每个 token 所需的平均时间（毫秒），也称**解码步长**或**迭代时间**。 | 决定整个响应生成的速度，时间越短，生成整个长文本所需的时间就越少。 |
| **平均迭代延迟**            | `ITL` / `Mean ITL`                      | 模型生成每个 token (或每个解码步骤) 的内部处理延迟。         | 反映模型内部计算效率，更低的 `ITL` 意味着模型在 GPU 上的计算过程更有效率。 |
| **99百分位迭代延迟**        | `P99_ITL` / `P99 ITL`                   | 99% 的迭代延迟都小于或等于这个值。                           | 用于评估系统在最坏情况下的表现，识别和优化少数特别慢的迭代。 |
| **99百分位首次 token 时间** | `P99_TTFT` / `P99 TTFT`                 | 99% 的请求的首次 token 时间都小于或等于这个值。              | 确保绝大多数用户都能获得快速的首次响应，即使在高负载或有少量异常情况时。 |
| **99百分位每 token 时间**   | `P99_TPOT` / `P99 TPOT`                 | 99% 的生成 token 的每 token 时间都小于或等于这个值。         | 衡量模型在绝大多数情况下持续生成 token 的稳定性，有助于发现生成过程中的偶尔卡顿或延迟。 |

------

其中 P99 ITL、P99 TTFT 和 P99 TPOT 这三个指标：

| 指标         | 全称                      | 核心含义                             | 衡量什么？                                 | 对用户体验的影响                         | 关注点                                   |
| ------------ | ------------------------- | ------------------------------------ | ------------------------------------------ | ---------------------------------------- | ---------------------------------------- |
| **P99 ITL**  | P99 Inter-Token Latency   | 第 99 百分位数的**令牌间延迟**       | 模型生成**连续两个输出令牌之间**的时间间隔 | 影响输出的**流畅性和连贯性**，避免卡顿。 | **流式传输**性能，避免单次卡顿。         |
| **P99 TTFT** | P99 Time to First Token   | 第 99 百分位数的**首令牌生成时间**   | 从请求到生成**第一个输出令牌**所需的时间   | 影响用户的**第一印象和响应速度感知**。   | **首次响应速度**，即“模型多快能开始说？” |
| **P99 TPOT** | P99 Time per Output Token | 第 99 百分位数的**每个输出令牌时间** | 生成**每个输出令牌的平均时间**             | 影响**完成整个输出的等待时间**。         | **持续生成效率**，即“模型说得有多快？”   |

------

希望这个表格能让您更清晰地理解这些指标！

## 大模型推理压测结果数据解读

| 字段名              | 代码中的实际变量名  | 英文全称              | 解释                                                         | 实际意义                                                     |
| ------------------- | ------------------- | --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **张量并行度**      | `tp`                | Tensor Parallelism    | 在模型推理过程中使用的**张量并行 (Tensor Parallelism)** 的 GPU 数量。 | 反映了本次测试中模型利用的计算资源量。更高的值通常意味着能处理更大模型或达到更高吞吐量。 |
| **数据类型**        | `data_type`         | Data Type             | 模型推理时使用的数据精度，如 `float16`、`bfloat16` 或 `int8`。 | 影响模型推理的**性能**和**显存占用**。低精度数据类型（如 `float16`）通常能提升速度并减少资源消耗。 |
| **批处理大小**      | `batch`             | Batch Size            | 一次性送入模型进行处理的请求数量。                           | 影响**吞吐量**和**延迟**的关键参数。增大批处理大小通常能提高 GPU 利用率和总吞吐量，但会增加平均延迟。 |
| **输入 token 数量** | `prompt_tokens`     | Prompt Tokens         | 每个请求中，用户输入的提示 (prompt) 所包含的 token 数量。    | 影响**模型预填充 (prefill) 阶段的计算量**。长的 prompt 会增加生成第一个 token 之前的计算负担。 |
| **输出 token 数量** | `completion_tokens` | Completion Tokens     | 模型为每个请求生成的响应 (completion) 所包含的 token 数量。  | **直接决定模型生成阶段的计算量和持续时间**。长的输出 token 对模型的生成吞吐量和每 token 时间影响更大。 |
| **总吞吐量**        | `TOTAL_THROUGHPUT`  | Total Throughput      | 单位时间内模型处理的总 token 数量（包括输入和输出），通常以“token/秒”为单位。 | **衡量模型整体处理能力的核心指标**。它综合反映了模型在给定时间内的输入处理和输出生成效率。 |
| **生成吞吐量**      | `GEN_THROUGHPUT`    | Generation Throughput | 单位时间内模型生成的 token 数量，也称**解码吞吐量**或**生成速度**，通常以“token/秒”为单位。 | 衡量模型**实际生成文本的速度**。数值越高，意味着用户可以更快地获得完整的响应。 |
| **首次 token 时间** | `TTFT`              | Time To First Token   | 从发送请求到接收到第一个 token 所需的**平均时间**（毫秒）。  | **对用户体验影响很大**，因为它决定了用户感知到的模型响应速度。在交互式应用中，TTFT 越低，用户会感觉响应越“实时”。 |
| **每 token 时间**   | `TPOT`              | Time Per Output Token | 生成后续每个 token 所需的**平均时间**（毫秒），也称**解码步长**或**迭代时间**。 | **决定整个响应生成的速度**。TPOT 越低，生成整个长文本所需的时间就越少。它主要反映了模型在连续生成阶段的效率。 |
| **迭代延迟**        | `ITL`               | Iteration Latency     | 模型生成每个 token（或每个解码步骤）的**内部处理延迟**（毫秒），不包括网络传输等外部开销。 | 反映模型**内部计算的效率**。更低的 ITL 意味着模型在 GPU 上的计算过程更有效率，有助于提高整体生成速度。 |

**测试效果如下**

![测试效果](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202507251545542.png)