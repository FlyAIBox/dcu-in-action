# 03-DAS：DCU 人工智能基础软件系统

## DAS：国产加速卡人工智能基础软件系统 概览 🚀

**DAS (DCU AI Software Stack)** 是一套专为国产加速卡打造的完整人工智能基础软件系统。它旨在全面支持从模型训练到推理落地的人工智能全领域应用，包括各类大模型（自然语言、多模态、MoE、视觉）及通用模型等，助力实现 AI 应用在国产加速卡上的快速迁移、开发与迭代。DAS 与国产加速卡硬件、DTK 开发工具栈、模型库 (Model Zoo) 及镜像仓库等共同构成了完善的国产 AI 软硬件生态。

![DAS架构.png](/docs/img/DCU计算平台软硬件架构.png)

------

DAS 包含以下核心层面：

### 🧱 一、基础算子层：性能优化的基石

此层专注于提供高性能的定制化及融合算子，最大化硬件效能。

- 核心算子库 `lightop`:
  - 深度优化并与 PyTorch 集成，支持可融合的稀疏计算类算子。
  - 整合 **BLAS、DNN、RCCL** 等基础库优化计算与通信。
  - 针对访存密集型算子，提供大量手写融合算子以提升性能。
- 高级开发支持 `CK (Composable Kernel)`:
  - 提供支持 Tensor Core 指令的算子模板库。
  - 通过细粒度的 tile 级别运算模板化，提升算子性能，并为高阶开发者提供灵活的开发选项。
- AI 编译器集成:
  - 整合 **Triton、XLA、TVM** 等 AI 编译组件进行算子生成。
  - 为 `FlashAttention`、`Xformer`、`Bitsandbytes` 等组件提供 AI 编译支持，拓展开发方式。
- **获取途径**: 已发布的算子可通过“光合开发者社区-DAS”获取。

------

### 🛠️ 二、框架工具层：高效开发的支撑

该层提供适配国产加速卡的主流深度学习框架及一系列实用开发工具。

- 适配的深度学习框架:
  - 训练框架: **PyTorch、TensorFlow、JAX、PaddlePaddle (Paddle)、OneFlow**。
  - 通用推理框架: **OnnxRuntime、MIGraphX、AITemplate**。
- 关键开发工具:
  - **`FastPT`**: 快速适配基于 PyTorch 的第三方组件。
  - **`LayerCheck`**: 单层精度检测工具，快速定位精度问题。
  - **`GraphRay`**: 图优化组件，通过图算匹配优化模型性能。
- **开源社区 `OpenDAS`**: 提供大量经过适配优化的 AI 第三方组件。
- **获取途径**: 已发布的框架工具可通过“光合开发者社区-DAS”获取。

------

### 🧩 三、扩展组件层：赋能复杂应用场景

此层针对通用模型及大模型的训练与推理需求，提供优化的组件和系统级方案。

- **目标**: 针对不同计算访存特点，组合优化手段，构建系统工程优化能力。
- 大模型训练支持:
  - 适配 **DeepSpeed、Megatron-LM** 等主流分布式训练框架。
  - 采用拓扑优化、多维组合并行、激活重计算等策略全面覆盖大模型场景。
  - 关键组件: **DeepSpeed、FastMoE、Bitsandbytes、FlashAttention**。
- 大模型推理优化:
  - 针对 **`FlashAttention`、`FlashDecoding`、KV Cache、`PagedAttention`** 等进行适配优化。
  - 目标是降低首字延迟，提高吞吐能力。
- 通用训练组件:
  - 支持国产加速卡的 **Apex、Vision (torchvision)、Audio (torchaudio)、MMCV、DGL** 等。

通过这三个层面的协同工作，DAS 为国产加速卡构建了强大的人工智能软件基础，有效支撑各类 AI 应用的开发和部署。

## DAS中框架组件版本号

### 定义

当DAS产品中的框架和组件经过一段时间的迭代研发和测试，对外正式发布时，版本号的定义要符合外部版本号编码规则。 ![DAS版本号规范示意图.png](https://das.sourcefind.cn:55011/portal/api-admin/auth/MyFile?FileKey=/markdown/DAS版本号规范示意图.png_1742890387283.png)

- **基础版本号**：对于主流框架和适配组件，该三位版本号一般继承自官方正式发布的版本号，当官方没有正式发布版本时，我们将组件的初始版本定义为0.0.1。

- 扩展版本号

  ：+号后面是我们在das软件栈中自定义的框架和组件的扩展版本号，具体定义如下：

  - **das前缀**：表示这是 DAS 软件栈的一部分。
  - **优化次数**：每次软件规划版本发布时，发布的主流框架和组件都会在某个版本的dtk上完成一次优化适配。在基础版本号和dtk版本号不变的情况下，完成一次优化适配，则进行opt优化次数升级。
  - **dtk版本号**：表示软件适配的dtk版本，由两部分组成，第一部分为前缀：dtk，第二部分为dtk的版本数字，dtk25041表示dtk的25.04.1版本。

### 应用

下面介绍DAS版本号规范如何应用到DAS产品中的框架/组件的软件包标识。

DAS产品中框架/组件的软件包标识包含以下几个部分：

- wheel包： **框架/组件名称-框架/组件版本-适用python版本号-适用操作系统_适用系统架构.whl**

- run包 **框架/组件名称-框架/组件版本-适用python版本号-适用操作系统_适用系统架构.run**

  说明：黑色加粗部分为命名中必须包含的部分，各部分之间通过“-”连接。

**软件包标识示例**： `vllm-0.6.2+das.opt3.dtk2504-cp310-cp310-manylinux_2_28_x86_64.whl `表示das软件栈中的该软件包是基于vllm-0.6.2版本进行第三次优化的版本，该版本已适配dtk 25.04版本，适用于Python 3.10，兼容不同的64位linux发行版且glibc版本为2.28。

## DAS1.5主要组件

| **组件 (Component)**                                         | **说明 (Description)**                                       | **应用场景 (Application Scenarios)**                         | **版本号 (Version Number)**                                  | **功能改进 (Feature Improvements)**                          | **问题修复 (Problem Fixes)**                                 | **已知问题 (Known Issues)**                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **PyTorch**                                                  | 开源机器学习框架，广泛用于计算机视觉和自然语言处理等领域。   | 深度学习模型研究与开发、神经网络训练与推理、图像识别、自然语言处理、推荐系统等。 | 2.4.1+das.opt2.dtk2504                                       | - 支持dtk25.04系列软件栈<br />- 支持python3.8、python3.10和python3.11版本<br />- 新增支持BW系列卡<br />- 新增支持send recv 与计算kernel 同流，SENDRECV_STREAM_WITH_COMPUTE=1 使能<br />- 针对K100_AI/BW 新增支持torch.cuda.tunable（blaslt 后端）<br />- 新增BN 算子的mipoen NHWC 实现<br />- 优化torch.nn.functional.scaled dot_product_attention性能<br />- 新增USE_FA_BHSD，USE_FA_BHSD=1 使能调用bhsd layout 实现，提升部分性能 | - 修复torch.cdist donot_use_mm_for_euclid_dist 模式报错<br />- 修复foreach_expm1 complex128的计算崩溃问题 | - 针对K100_AI，不支持FP64<br />- MIOpen RNN函数(lstm, RNN, gru)不能处理空张量，若有需求可通过torch.backends.cudnn.enabled=False 禁用MIOpen调用，使用原生方法实现<br />- 部分torch.linalg相关操作存在计算误差 |
| **vLLM**                                                     | 用于大型语言模型 (LLM) 推理和服务的高效库。                  | 大语言模型的高速推理、在线服务、聊天机器人、文本生成、代码生成等。 | 0.6.2+das.opt3.dtk2504                                       | - 支持dtk25.04系列软件栈<br />- 支持对应官方的vllm-0.6.2版本<br />- 支持torch2.4.1软件栈<br />- 支持python3.8、python3.10和python3.11版本<br />- 新增支持BW，兼容K100_AI |                                                              | - 不支持K100以及Z100，Z100L系列卡<br />- 针对K100_AI，不支持FP64<br />- 不支持Marlin量化和FP8相关功能<br />- 不支持custom all-reduce kernel |
| (推测) DCU平台上的模型推理加速引擎或框架，与Torch等配合使用。 | 在DCU硬件上加速AI模型的推理部署，提高特定应用的推理性能。    | 0.0.2+das.dtk2504                                            | - 支持dtk25.04系列软件栈<br />- 支持torch2.4.1软件栈<br />- 支持python3.10<br />- 新增支持BW，兼容K100_AI |                                                              | - 不支持K100以及Z100，Z100L系列卡<br />- 针对K100_AI，不支持FP64<br />- 不支持custom all-reduce kernel<br />- 不支持fp8 |
| **LMSlim**                                                   | (推测) 用于大型语言模型压缩和优化的工具集，支持量化等技术。  | 减小大模型体积、降低显存占用、加速模型推理速度，尤其适用于资源受限的部署环境。 | 0.2.1+das.dtk2504                                            | - 支持dtk25.04系列软件栈<br />- 支持torch2.4.1软件栈<br />- 支持python3.8、python3.10和python3.11版本<br />- 新增支持BW<br />- 新增W8A8 triton支持，新增W8A8 tuning相关工具<br />- 新增GPTQ INT8 tensorcore 支持 | - 修复120cu卡awq rocblas策略反量化乱码问题<br />- 修复AWQ INT4支持在部分高并发应用场景崩溃问题 | - AWQ INT4模型量化暂不支持BW<br />- W8A8 仅支持激活per token+权重per channel量化格式 |
| **Triton**                                                   | 开源推理服务平台，用于在生产环境中部署和管理多种框架训练的AI模型。 (此为NVIDIA Triton的DAS适配版) | AI模型推理服务的统一部署、模型版本管理、动态批处理、多模型并发推理、跨框架模型支持。 | 3.0.0+das.opt4.dtk2504                                       | - 支持dtk25.04系列软件栈<br />- 支持python3.8、python3.10和python3.11版本<br />- 新增BW卡支持，兼容K100_AI<br />- 适配Torch2.4<br />- 增加ldmatrix指令<br />- 优化gemm、fa、bn算子性能 |                                                              | - 针对K100_AI，不支持FP64<br />- 新版本triton不支持Z100、K100<br />- BW卡打开max autotune 功能后 gemm算子会有偶发性精度问题 |
| **Vision**                                                   | PyTorch的官方计算机视觉库，包含流行的模型、数据集和图像转换工具。(此为TorchVision的DAS适配版) | 图像分类、目标检测、图像分割、视频分析、光流估计等计算机视觉任务。 | 0.19.1+das.opt2.dtk2504                                      | - 支持dtk25.04系列软件栈<br />- 支持python3.8、python3.10和python3.11版本<br />- 新增BW卡支持，兼容K100_AI<br />- 支持对应官方的torchvision-0.19.1版本<br />- 支持torch2.4版本 |                                                              | - 针对K100_AI，不支持FP64                                    |
| **Apex**                                                     | PyTorch的扩展库，提供混合精度训练和分布式训练等优化功能。(此为NVIDIA Apex的DAS适配版) | 加速大规模神经网络训练、减少显存占用、提升训练效率。         | 1.4.0+das.opt1.dtk2504                                       | - 支持dtk25.04系列软件栈<br />- 支持torch2.4.1软件栈<br />- 支持python3.10<br />- 新增支持BW、兼容K100_AI、K100、Z100系列卡 |                                                              | - 暂不支持transducer大部分<br />- 暂不支持部分混合精度训练   |
| **DGL**                                                      | 易于使用、高效且可扩展的Python包，用于在图结构数据上进行深度学习。 | 图神经网络（GNN）研究与应用、社交网络分析、推荐系统、知识图谱、药物发现、生物信息学等。 | 2.2.1+das.opt1.dtk2504                                       | - 支持dtk25.04系列软件栈<br />- 支持torch2.4.1软件栈         |                                                              | - K100_AI 下部分FP64结果异常<br />- UniqueAndCompactBatched不支持hash table的方法，graphbolt下NeighborSampler、LayerNeighborSampler下的结果应当参考device_capability &lt; 7的结果<br />- 单测中test_SubgraphSampler_HyperLink_Hetero、test_SubgraphSampler_without_deduplication_Homo_HyperLink报错 |
| **JAX**                                                      | Google开发的用于高性能数值计算和机器学习研究的Python库，具有自动微分和XLA编译能力。 | 机器学习研究、高性能科学计算、自定义数值算法实现、函数变换（如自动微分、JIT编译、向量化）。 | 0.4.34+das.opt1.dtk2504                                      | - 支持dtk25.04系列软件栈<br />- 支持python3.10和python3.11版本<br />- 支持BW、K100_AI、K100、Z100系列卡 | - 修复SYEVJ/HEEVJ方面的精度问题                              | - pallas相关部分算子存在精度问题                             |
| **Fastpt**                                                   | 针对PyTorch模型的优化或编译工具，可能用于提升在特定硬件（如DCU）上的性能。 | PyTorch模型优化、编译加速、提高模型在特定硬件上的执行效率。  | 2.0.0+das.dtk2504                                            | - 支持dtk25.04系列软件栈<br />- 支持torch2.4.1软件栈<br />- 支持python3.10 |                                                              | - 暂不支持处理三方库<br />- 暂不支持PTX编译指令              |
| **Flash_attn**                                               | 一种快速且内存高效的精确注意力算法实现，尤其适用于长序列。   | Transformer模型（如GPT、BERT）的训练和推理加速，特别是在处理长文本或高分辨率图像等长序列数据时。 | 2.6.1+das.opt4.dtk2504                                       | - 支持dtk25.04系列软件栈<br />- 支持python3.8、python3.10和python3.11版本<br />- 新增BW卡支持，兼容K100_AI |                                                              |                                                              |
| **TensorFlow**                                               | Google开发的端到端开源机器学习平台。                         | 各种机器学习和深度学习任务，包括模型构建、训练、评估和部署，应用于图像识别、语音识别、自然语言处理等。 | 2.13.1+das.opt1.dtk2504                                      | - 支持dtk25.04系列软件栈<br />- 支持hipblaslt autotuning，可通过export USE_CUBLAS_LT=1打开，默认关闭<br />- 支持python3.10<br />- 新增支持BW，兼容K100_AI、K100、Z100系列卡 |                                                              | - hipblaslt库性能目前较差，默认不打开hipblaslt autotuning<br />- kme相关单测会有FP64相关报错；新增报错测例RecomputeGradMemoryTest.testRecomputeGradXla，新合入xla相关功能导致测例增加2%的内存使用，经过评估不影响功能。 |
| **Deepspeed**                                                | 微软推出的深度学习优化库，旨在提高大规模模型训练的效率和规模。 | 大规模模型训练（如万亿参数模型）、分布式训练、模型并行、流水线并行、混合精度训练、ZeRO显存优化。 | 0.14.2+das.opt2.dtk2504                                      | - 支持dtk25.04系列软件栈<br />- 支持python3.8、python3.10和python3.11版本<br />- 随官方版本升级到0.14.2，新增功能参看官方 |                                                              | - Intel cpu 编译可能导致精度问题<br />- 针对K100_AI，不支持FP64<br />- 暂不支持官方0.14.2版本中新增的fp量化功能 |
| **ONNX Runtime**                                             | 用于ONNX (Open Neural Network Exchange) 格式模型的高性能推理引擎。 | 跨多种框架（PyTorch, TensorFlow, scikit-learn等）训练的模型的统一部署和加速推理，支持云端和边缘设备。 | 1.15.0+das.opt1.dtk2504                                      | - 合入MIGraphX后端相关功能<br />- 增加scatterND算子性能优化<br />- 增加ROCm后端多线程支持<br />- 新增支持BW，兼容K100_AI、K100、Z100 |                                                              | - 针对K100_AI，不支持FP64                                    |
| **Transformer Engine**                                       | (NVIDIA Transformer Engine的DAS适配版) 用于加速Transformer模型训练和推理的库，支持FP8等先进技术。 | 高效训练和部署大规模Transformer模型，利用FP8等混合精度技术进一步提升性能和降低显存占用，用于大语言模型、计算机视觉等。 | 1.9.0+das.opt2.dtk2504                                       | - 支持dtk25.04系列软件栈<br />- 支持python3.8、python3.10和python3.11版本<br />- 支持BW、K100_AI、K100、Z100系列卡<br />- 支持torch2.4、fa2.6.1、triton3.0版本<br />- 针对MOE，新增batchgemm,可替换相同矩阵size的group gemm使用<br />- 支持两种外置fa，默认走cutlass版本fa,设置export NVTE_FLASH_ATTN_TRITON=1走外置triton版本fa<br />- 纯gemm默认走blas,可以export NVTE_FORCE_BLASLT=1强制所有gemm走blaslt |                                                              | - 不支持FP8相关功能<br />- 不支持fused attention功能<br />- 不支持TP comm overlap功能<br />- 暂只支持pytorch |

## 参考

1. [DAS介绍](https://das.sourcefind.cn:55011/portal/#/docs/DAS%E4%BB%8B%E7%BB%8D)
2. [DAS代码仓库](https://developer.sourcefind.cn/codes/OpenDAS)
3. [镜像仓库](https://sourcefind.cn/#/model-zoo/list)