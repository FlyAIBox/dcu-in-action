# 01-DCU 环境及 ROCm 软件栈简介

**海光DCU** 采用 **ROCm** (Radeon Open Computing Platform) 作为其核心软件支撑。ROCm 是一个开源的、轻量级、模块化的 GPU 通用计算平台，为开发者提供了丰富的工具和运行时环境，适用于大规模计算和异构应用开发。

------

### ROCm 核心特性与架构 🛠️

![ROCm组件结构](/docs/img/ROCm组件结构.png)

- 分层构建: ROCm 采用自底向上的模块化设计，主要组件包括：
  - **底层驱动**: 加速器驱动 **ROCk**。
  - **运行时系统**: **ROCt** 和 **ROCr**。
  - **编程模型**: 主要是 **HIP** (Heterogeneous-Compute Interface for Portability)，也支持 OpenCL 和 C++ AMP。
  - **库与工具**: 基础数学库、管理工具、优化和调试工具。
- **异构计算**: ROCm 旨在简化在 CPU、DCU、GPU 等不同设备上的异构计算应用程序开发。

------

### ROCm 与 CUDA 对比 🔄

ROCm 在架构上与 NVIDIA 的 CUDA 有相似之处，但关键区别在于 **ROCm 是开源的**。

| **ROCm 组件**                    | **对应 CUDA 组件** | **功能说明**                                      |
| -------------------------------- | ------------------ | ------------------------------------------------- |
| ROCk (内核态驱动) 和 ROCt (接口) | CUDA Driver        | 底层驱动及接口 (CUDA Driver API)                  |
| HIP Runtime (或其他模型Runtime)  | CUDA Runtime       | 用户态驱动，负责资源管理与调度 (CUDA Runtime API) |

------

### HIP 编程模型详解 💻

- **核心地位**: HIP 是 ROCm 中最主要的异构并行编程模型。
- **跨平台兼容**: 目标是让代码能在 NVIDIA GPU、DCU 等多个平台上编译运行。
- **CUDA 迁移**: 提供 `hipify` 工具，可将 CUDA 源码转换为 HIP 模型。
- **高度相似**: HIP 的数据结构、接口模型及核函数语法与 CUDA 基本一致。
- 编译流程:
  - 使用 **HIPCC** 编译器（内含 Clang，可调用 NVCC）。
  - 主机端和设备端代码由 Clang 分别编译为 LLVM IR，再生成各自的可执行代码，最终合并。

------

### ROCm 生态：数学库与工具集 📊🔬

ROCm 提供了丰富的数学库和工具，以支持高效开发、调试和优化。

- 主要数学库

   (及其 CUDA 对应):

  - `hipblas` (cuBLAS): 基础矩阵运算
  - `hiprand` (cuRAND): 随机数
  - `hipsparse` (cuSPARSE): 稀疏矩阵
  - `hipfft` (cuFFT): 快速傅里叶变换
  - `miopen` (cuDNN): 深度学习
  - `hipcub` (CUB): 基础算法
  - `rccl` (NCCL): 通信库
  - `rocThrust` (Thrust): 并行库

- 主要优化与调试工具:

  - `rocprofiler`: 程序分析和时间线绘制
  - `roctracer`: 程序跟踪

- 管理工具:

  - `rocm-smi`: 查看 DCU 系统状态

总而言之，ROCm 为 DCU 提供了一个功能完整、开源的软件生态环境，旨在对标并提供 CUDA 的替代方案，尤其强调其可移植性和开放性。

## DCU软硬件平台架构
![ROCm组件结构](/docs/img/DCU计算平台软硬件架构.png)

## DCU资源工具

| **名称**                     | **下载地址**                                   |
| ---------------------------- | ---------------------------------------------- |
| 驱动                         | `https://download.sourcefind.cn:65024/6/main`  |
| DTK(DCU Toolkit)             | `https://download.sourcefind.cn:65024/1/main`  |
| DAS（DCU AI Software Stack） | `https://download.sourcefind.cn:65024/4/main/` |
| 科学计算                     | `https://download.sourcefind.cn:65024/9/main`  |
| 光源（镜像、模型仓库）       | `https://sourcefind.cn/`                       |
| 工具包                       | `https://download.sourcefind.cn:65024/5/main`  |
| 隐私计算                     | `https://download.sourcefind.cn:65024/8/main`  |
| DCU编程实训 代码             | `https://download.sourcefind.cn:65024/7/main`  |



## 参考资料
1. [DCU环境简介](https://developer.sourcefind.cn/gitbook//dcu_developer/OperationManual/1_ROCmIntro/ROCmIntro.html)
2. [DCU硬件安装参考](https://developer.sourcefind.cn/gitbook//dcu_tutorial/index.html)
3. [DCU驱动安装参考](https://developer.sourcefind.cn/gitbook//dcu_tutorial/index.html)
4. [DTK安装包及部署文档下载](https://cancon.hpccube.com:65024/1/main/)
5. [DTK安装步骤参考教程](https://developer.sourcefind.cn/gitbook//dcu_tutorial/index.html)