# ROCm 常用命令运维手册 

`rocm-smi` (ROCm System Management Interface) 是一个命令行实用工具，用于监控和管理 AMD GPU 设备。它类似于 NVIDIA 的 `nvidia-smi` 工具，为系统管理员和开发人员提供了查看和控制 GPU 状态及资源占用的能力。

------

## 第一部分：基础监控与状态查看

### 1.1 显示所有设备信息及显存占用

此命令用于列出系统中所有可用的 AMD GPU 设备及其当前的显存使用、温度、功耗等基本状态信息。

**命令:**

```Bash
rocm-smi
```

或 (等效命令):

```Bash
rocm-smi --alldevices
```

预期输出:

会显示类似以下格式的信息，包含每个 GPU 的 ID、固件版本、VBIOS 版本、显存总量、已用显存、温度、功耗、性能级别、风扇转速等。

### 1.2 显示独立显存占用情况

#### 1.2.1 详细显存分类占用 (VRAM, Visible VRAM, GTT)

此命令用于更详细地展示 GPU 的显存使用情况，包括 VRAM、可见 VRAM (Visible VRAM) 和 GTT (Graphics Translation Table) 内存。

**关键参数说明:**

- **`vram`**: GPU 专用的板载高速显存。
- **`vis_vram`**: CPU 可直接访问的 GPU 专用显存部分。
- **`gtt`**: 图形转换表内存，通常是系统内存中被 GPU 用于特定地址转换和数据暂存的区域。

**命令:**

```Bash
rocm-smi --showmeminfo vram vis_vram gtt
```

预期输出:

会针对每个 GPU 列出其 VRAM 总量/已用量、可见 VRAM (vis_vram) 总量/已用量以及 GTT 内存总量/已用量。

例如：

```Bash
========================== ROCm System Management Interface ==========================
GPU[0]		: vram Total Memory (MiB): 65520
GPU[0]		: vram Total Used Memory (MiB): 2048
GPU[0]		: vis_vram Total Memory (MiB): 256
GPU[0]		: vis_vram Total Used Memory (MiB): 50
GPU[0]		: gtt Total Memory (MiB): 16384
GPU[0]		: gtt Total Used Memory (MiB): 100
... (其他 GPU 的信息) ...
================================= End of ROCm SMI Log ==================================
```

#### 1.2.2 仅查看 VRAM 显存占用

此命令专门用于显示 GPU 的 **VRAM (Video RAM)** 使用情况。

**命令:**

```Bash
rocm-smi --showmeminfo vram
```

预期输出:

会针对每个 GPU 单独列出其 VRAM 的总量 (Total Memory) 和已用量 (Total Used Memory)。

例如：

```Bash
========================== ROCm System Management Interface ==========================
GPU[0]		: vram Total Memory (MiB): 65520
GPU[0]		: vram Total Used Memory (MiB): 2048
... (其他 GPU 的 VRAM 信息) ...
================================= End of ROCm SMI Log ==================================
```

### 1.3 统计所有加速卡的总显存和已用显存

此命令组合通过解析 `rocm-smi` 的显存信息输出来汇总当前机器上所有 AMD 加速卡的总显存容量以及当前已使用的显存总量。

**命令:**

```Bash
rocm-smi --showmeminfo vram | awk '
/vram Total Memory \(MiB\):/ { total_vram += $NF }
/vram Total Used Memory \(MiB\):/ { used_vram += $NF }
END {
  printf "所有加速卡总显存 (Total VRAM): %s MiB\n", total_vram;
  printf "所有加速卡已用显存 (Total Used VRAM): %s MiB\n", used_vram;
}'
```

预期输出:

命令会输出两行结果，分别显示所有加速卡累加的总显存大小和已使用显存大小，单位是 MiB。

例如：

```Bash
所有加速卡总显存 (Total VRAM): 524160 MiB
所有加速卡已用显存 (Total Used VRAM): 16384 MiB
```

### 1.4 显示占用 GPU 的进程 ID

#### 1.4.1 基础进程信息

此命令用于识别当前正在使用 GPU 资源的进程及其对应的进程 ID (PID)。这对于排查 GPU 资源占用过高或定位特定 GPU 应用非常有用。

**命令:**

```Bash
rocm-smi --showpids
```

预期输出:

会列出每个 GPU 上正在运行的进程的 GPU ID 和进程 ID (PID)。

#### 1.4.2 详细进程信息 (包括进程名)

此命令在基础进程信息之上，额外显示进程的名称。

**关键参数说明:**

- `-P` 或 `--showprocess`: 在输出中包含进程的名称。

**命令:**

```Bash
rocm-smi --showpids -P
```

预期输出:

会列出每个 GPU 上正在运行的进程的 GPU ID、进程 ID (PID) 以及进程名称。

例如：

```Bash
========================== ROCm System Management Interface ==========================
GPU[0] Process ID: 12345, Process Name: my_gpu_app
GPU[0] Process ID: 67890, Process Name: another_app
... (其他 GPU 上的进程信息) ...
================================= End of ROCm SMI Log ==================================
```

*(备注: `-u` 参数在标准 `rocm-smi` 文档中不常见，其功能可能取决于特定版本或环境。)*

------

## 第二部分：硬件信息与识别

### 2.1 查看 GPU 时钟频率

此命令用于显示 GPU 上各种关键时钟（如核心时钟、显存时钟等）的当前频率。

**命令:**

```Bash
rocm-smi --showclocks
```

预期输出:

会列出每张 HCU (GPU) 的各项时钟频率，例如 fclk (Fabric Clock), mclk (Memory Clock), sclk (Shader/Core Clock), socclk (System on Chip Clock) 和 pcie (PCI Express Clock) 等。

例如：

```Bash
============================ System Management Interface =============================
======================================================================================
HCU[0]		: fclk clock level: 0 (1250Mhz)
HCU[0]		: mclk clock level: 0 (875Mhz)
HCU[0]		: sclk clock level: 0 (600Mhz)
HCU[0]		: socclk clock level: 2 (566Mhz)
HCU[0]		: pcie clock level 1 (16.0GT/s, x16 800Mhz)
... (其他 HCU 的信息) ...
======================================================================================
```

这些信息对于性能分析和功耗管理非常重要。`mclk` 是计算理论显存带宽的关键参数之一。

### 2.2 检查 GPU 类型

此命令组合用于从硬件和完整信息中筛选并显示 GPU 卡的型号或市场名称。

**命令:**

```Bash
rocm-smi --showhw --showallinfo | grep -i card
```

**说明:**

- `rocm-smi --showhw --showallinfo`: 显示关于 GPU 硬件的非常详细的静态信息。
- `| grep -i card`: 通过管道搜索包含 "card" 关键词的行（忽略大小写）。

预期输出:

输出会是包含 "card" 的若干行信息，其中可能包含显卡的市场名称 (Marketing Name)、型号 (Card model) 或系列 (Card series) 等。

例如:

```Bash
HCU[0]		: Card Series:		 K100_AI
HCU[0]		: Card Vendor:		 Chengdu Haiguang IC Design Co., Ltd.
```

### 2.3 检查 GPU PCIe 总线 ID

此命令用于列出系统中 ROCm 可识别的 GPU 及其对应的 PCIe 总线地址。

**命令:**

```Bash
rocm-smi --showid
```

预期输出:

输出内容通常包括 GPU ID, DEVICE ID (可选), 以及 PCI Bus (GPU 的 PCIe 总线地址，例如 0000:c1:00.0)。

例如：

```rocm-smi --showid
============================ System Management Interface =============================
======================================================================================
HCU[0]		: Device ID: 0x6210
HCU[1]		: Device ID: 0x6210
HCU[2]		: Device ID: 0x6210
HCU[3]		: Device ID: 0x6210
HCU[4]		: Device ID: 0x6210
HCU[5]		: Device ID: 0x6210
HCU[6]		: Device ID: 0x6210
HCU[7]		: Device ID: 0x6210
======================================================================================
=================================== End of SMI Log ===================================
```

------

## 第三部分：性能与带宽分析

### 3.1 确定显存理论带宽

此操作结合 `rocm-smi --showclocks` 命令的输出和 GPU 的官方规格文档，用于确定或估算显存的理论峰值带宽。理论带宽是评估显存性能的重要指标。

**步骤与关键参数:**

1. 使用 `rocm-smi --showclocks` (参考 2.1节) 获取 `mclk` (显存时钟频率)。
2. 查阅 GPU 的官方规格文档获取：
   - **显存位宽 (Memory Bus Width)**: GPU 与显存之间数据总线的宽度 (如 2048-bit, 4096-bit for HBM)。
   - **显存类型 (Memory Type)**: 如 HBM2, HBM3, GDDR6 等，并确定其数据传输倍率 (通常为 2，因为是 DDR)。

理论带宽计算公式示例 (具体细节取决于显存类型):

理论带宽 (GB/s) = (mclk频率 (MHz) × 数据传输倍率 × 显存位宽 (bits) / 8) / 1000

**示例 (假设):**

- `mclk` = 875 MHz (来自 `rocm-smi --showclocks`)
- 显存位宽 = 4096 bits (来自官方规格)
- 显存类型为 HBM2 (数据传输倍率 = 2)

```Bash
理论带宽 = (875 × 2 × 4096 / 8) / 1000 = (875 × 1024) / 1000 = 896000 / 1000 = 896 GB/s
```

### 3.2 查看卡间连接拓扑和类型

此命令用于显示系统中各 GPU 之间以及 GPU 与 CPU 之间的连接拓扑结构和物理连接类型。

**命令:**

```Bash
rocm-smi --showtopo
```

预期输出:

输出通常包含 Weight Matrix (通信权重), Link Type Matrix (连接类型，如 XGMI 或 PCIE), 和 Hops Matrix (跳数)。

示例输出片段：

```Bash

============================ System Management Interface =============================
======================================================================================
Link accessible between HCUs
          HCU[0]    HCU[1]    HCU[2]    HCU[3]    HCU[4]    HCU[5]    HCU[6]    HCU[7]    
HCU[0]    TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      
HCU[1]    TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      
HCU[2]    TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      
HCU[3]    TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      
HCU[4]    TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      
HCU[5]    TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      
HCU[6]    TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      
HCU[7]    TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      TRUE      
======================================================================================
======================================================================================
Weight between HCUs
          HCU[0]    HCU[1]    HCU[2]    HCU[3]    HCU[4]    HCU[5]    HCU[6]    HCU[7]    
HCU[0]    0         40        40        40        40        40        40        40        
HCU[1]    40        0         40        40        40        40        40        40        
HCU[2]    40        40        0         40        40        40        40        40        
HCU[3]    40        40        40        0         40        40        40        40        
HCU[4]    40        40        40        40        0         40        40        40        
HCU[5]    40        40        40        40        40        0         40        40        
HCU[6]    40        40        40        40        40        40        0         40        
HCU[7]    40        40        40        40        40        40        40        0         
======================================================================================
======================================================================================
Hops between HCUs
          HCU[0]    HCU[1]    HCU[2]    HCU[3]    HCU[4]    HCU[5]    HCU[6]    HCU[7]    
HCU[0]    0         2         2         2         2         2         2         2         
HCU[1]    2         0         2         2         2         2         2         2         
HCU[2]    2         2         0         2         2         2         2         2         
HCU[3]    2         2         2         0         2         2         2         2         
HCU[4]    2         2         2         2         0         2         2         2         
HCU[5]    2         2         2         2         2         0         2         2         
HCU[6]    2         2         2         2         2         2         0         2         
HCU[7]    2         2         2         2         2         2         2         0         
======================================================================================
======================================================================================
Link Type between HCUs
          HCU[0]    HCU[1]    HCU[2]    HCU[3]    HCU[4]    HCU[5]    HCU[6]    HCU[7]    
HCU[0]    None      PCIE      PCIE      PCIE      PCIE      PCIE      PCIE      PCIE      
HCU[1]    PCIE      None      PCIE      PCIE      PCIE      PCIE      PCIE      PCIE      
HCU[2]    PCIE      PCIE      None      PCIE      PCIE      PCIE      PCIE      PCIE      
HCU[3]    PCIE      PCIE      PCIE      None      PCIE      PCIE      PCIE      PCIE      
HCU[4]    PCIE      PCIE      PCIE      PCIE      None      PCIE      PCIE      PCIE      
HCU[5]    PCIE      PCIE      PCIE      PCIE      PCIE      None      PCIE      PCIE      
HCU[6]    PCIE      PCIE      PCIE      PCIE      PCIE      PCIE      None      PCIE      
HCU[7]    PCIE      PCIE      PCIE      PCIE      PCIE      PCIE      PCIE      None      
======================================================================================
======================================================================================
HCU[0]		: Numa Node:  0
HCU[0]		: Numa Affinity:  0
HCU[1]		: Numa Node:  0
HCU[1]		: Numa Affinity:  0
HCU[2]		: Numa Node:  0
HCU[2]		: Numa Affinity:  0
HCU[3]		: Numa Node:  0
HCU[3]		: Numa Affinity:  0
HCU[4]		: Numa Node:  0
HCU[4]		: Numa Affinity:  0
HCU[5]		: Numa Node:  0
HCU[5]		: Numa Affinity:  0
HCU[6]		: Numa Node:  0
HCU[6]		: Numa Affinity:  0
HCU[7]		: Numa Node:  0
HCU[7]		: Numa Affinity:  0
======================================================================================
=================================== End of SMI Log ===================================
```

### 3.3 实际测量卡间互联带宽

此命令通过运行实际的数据传输测试来测量系统中不同设备对（尤其是 GPU 与 GPU 之间）的有效带宽。

**命令:**

```Bash
rocm-bandwidth-test
```

预期输出:

命令执行后会输出一个带宽矩阵，显示各个源设备到目标设备的单向和双向数据拷贝带宽，单位通常是 GB/s。重点关注 GPU 到 GPU (D2D) 的测试结果。

示例输出片段：

```Bash
RocmBandwidthTest Version: 2.6.0

          Launch Command is: rocm-bandwidth-test (rocm_bandwidth -a + rocm_bandwidth -A)

....
```

### 3.4 检查 GPU PCIe 接口带宽

此命令用于监控 GPU 通过 PCIe 总线与系统其他部分进行数据传输的实时带宽。

**命令:**

```bash
rocm-smi -b
```

或 (完整参数名):

```Bash
rocm-smi --showbw
```

预期输出:

命令会为每个 GPU（或指定的 GPU）持续刷新显示的 PCIe 带宽数据，包括发送 (Sent) 和接收 (Received) 的速率 (MB/s 或 GB/s)。按 Ctrl+C 停止。

```Bash
========================== ROCm System Management Interface ==========================
GPU[0]		: PCIe Bus Bandwidth Received: 150 MiB/s
GPU[0]		: PCIe Bus Bandwidth Sent: 75 MiB/s
...
================================= End of ROCm SMI Log ==================================
```

------

## 第四部分：特定厂商工具 (示例)

### 4.1 使用 `hy-smi` 列出 Hygon DCU (GPU) 设备

`hy-smi` 是针对海光 (Hygon) DCU (GPU) 的命令行管理工具，功能上类似于 `rocm-smi`，专门用于监控和管理海光加速卡。

说明:

此命令用于列出系统中所有已安装并被识别的海光 DCU 设备及其状态。具体支持的参数和输出格式请参考 `hy-smi --help` 或海光提供的官方文档。

**命令 (通用示例):**

```Bash
hy-smi
```

预期输出:

通常会显示系统中每块海光 DCU 的设备 ID、型号、温度、功耗、显存使用情况等信息。

------

## 总结

以上是 `rocm-smi` 及其相关工具的一些常用命令，覆盖了从基本状态监控、硬件识别到性能带宽分析等多个方面。掌握这些命令可以帮助运维人员和开发人员更有效地管理和调试基于 ROCm 平台的 AMD GPU 系统。根据具体需求，`rocm-smi` 和其他 ROCm 工具还提供了更多高级功能和选项，建议通过 `--help` 选项或查阅官方文档来获取完整的帮助信息。