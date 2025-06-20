# LLaMA Factory 参数体系详解与核心配置（上篇）

## 摘要

LLaMA Factory 作为当前最受欢迎的大语言模型微调框架，其参数体系的复杂性和专业性要求开发者具备深入的理论理解和实践经验。本文基于[LLaMA Factory 官方文档](https://llamafactory.readthedocs.io/zh-cn/latest/advanced/arguments.html#id2)，深入解析其参数体系架构，详细阐述核心参数的作用机制、配置策略和最佳实践。

本文分为上下两篇：上篇重点讲解参数体系架构和核心参数配置；下篇聚焦高级优化策略和生产环境实践。

**关键词**: LLaMA Factory, 参数配置, 模型微调, LoRA, 大语言模型

## 1. 引言

### 1.1 背景与意义

随着 Transformer 架构在自然语言处理领域的广泛应用，大语言模型微调已成为AI应用开发的核心技术。LLaMA Factory 框架通过统一的参数接口，支持 100+ 种预训练模型的高效微调，极大降低了技术门槛。

然而，其丰富的参数选项（400+ 个配置项）也带来了配置复杂性。**不当的参数设置可能导致：**
- **训练不收敛或过拟合**
- **显存溢出（OOM）错误**
- **训练效率低下**
- **模型性能不达预期**

### 1.2 文章贡献

本文的主要贡献包括：
1. **系统性参数分类体系**：将复杂参数分为8大类别，建立清晰的层次结构
2. **参数依赖关系分析**：揭示参数间的相互影响和约束关系
3. **生产级配置模板**：提供针对不同硬件和任务的最佳实践配置
4. **科学化调优方法**：建立基于实验和监控的参数优化流程

## 2. LLaMA Factory 参数体系架构

### 2.1 整体架构设计

LLaMA Factory 参数体系采用分层架构设计，共分为五个层次，每层负责不同的功能职责：

**核心控制层（Core Control Layer）**：定义微调方法和训练策略的高级控制参数
**算法配置层（Algorithm Config Layer）**：配置具体的微调算法和优化策略
**资源管理层（Resource Management Layer）**：管理计算资源、数据资源和模型资源
**执行引擎层（Execution Engine Layer）**：实际执行训练、推理和评估任务
**监控评估层（Monitoring & Evaluation Layer）**：提供实验跟踪、性能监控和模型评估

### 2.2 参数分类体系

根据官方文档和实际应用需求，我们将 LLaMA Factory 的参数分为以下8大类别：

| 参数类别 | 英文名称 | 参数数量 | 核心作用 | 配置复杂度 |
|---------|----------|----------|----------|------------|
| 微调参数 | FinetuningArguments | 50+ | 控制微调方法和策略 | ⭐⭐⭐⭐⭐ |
| 训练参数 | TrainingArguments | 80+ | 控制训练超参数和优化器 | ⭐⭐⭐⭐⭐ |
| 数据参数 | DataArguments | 40+ | 控制数据加载和预处理 | ⭐⭐⭐⭐ |
| 模型参数 | ModelArguments | 60+ | 控制模型加载和配置 | ⭐⭐⭐ |
| 生成参数 | GenerationArguments | 20+ | 控制文本生成策略 | ⭐⭐⭐ |
| 评估参数 | EvaluationArguments | 15+ | 控制模型评估设置 | ⭐⭐ |
| 监控参数 | MonitoringArguments | 10+ | 控制实验跟踪和可视化 | ⭐⭐ |
| 环境变量 | Environment Variables | 30+ | 控制系统环境和调试 | ⭐⭐ |

### 2.3 参数优先级和依赖关系

参数间存在复杂的依赖关系和优先级约束。理解这些关系对于正确配置至关重要：

**一级依赖（强制约束）**：

![基本参数](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202506201641265.png)

- `model_name_or_path` → `template`：模型路径决定对话模板
- `finetuning_type` → `lora_*`：微调类型决定LoRA参数是否生效
- `stage` → `dataset`：训练阶段决定数据集要求

**二级依赖（影响性能参数-显存消耗与训练稳定性）**：

![性能影响-显存消耗与训练稳定性](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202506201710133.png)

| **参数 (Parameter)**                   | **对显存消耗的影响 (Impact on VRAM Consumption)**            | **对训练稳定性的影响 (Impact on Training Stability)**        |
| -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **截断长度 (Cutoff Length)**           | **核心影响**: 显存消耗与此值的**平方**成正比。是主要的动态显存（激活值）占用源。 | **次要影响**: 过长的序列会增加模型学习上下文的难度，但对稳定性的直接影响不如学习率。 |
| **批处理大小 (Per Device Batch Size)** | **核心影响**: 显存消耗与此值成**正比**。`1` 是最节省显存的设置，直接决定了单次计算的激活值大小。 | **高影响**: 与“梯度累积”共同决定有效批量大小。单独看，极小的批处理大小会导致梯度噪声大，不利于稳定。 |
| **计算类型 (Compute Type)**            | **主要影响**: 相比 `fp32`，`bf16` 混合精度能将模型权重、梯度、优化器状态的显存占用**减半**。 | **次要影响**: `bf16` 数值范围广，稳定性优于 `fp16`。相比 `fp32`，任何混合精度都有微小的数值不确定性风险，但在现代框架下已基本可忽略。 |
| **LoRA 秩 (lora_rank)**                | **中等影响**: **直接决定 LoRA 适配器的大小**。`rank` 越高，注入的新参数就越多，相应的显存占用也会增加。`16` 是一个兼顾效果与资源的均衡选择。 | **中等影响**: `rank` 关系到模型的“学习容量”。过高的 `rank` 在小数据集上可能导致过拟合，这也是一种广义上的不“稳定”。`16` 相比 `8` 赋予了模型更强的学习能力。 |
| **LoRA 作用模块 (lora_target)**        | **中等影响**: 作用的模块范围越广（如 `all`），需要创建的 LoRA 适配器就越多，总的 LoRA 参数量和显存占用也会相应增加。 | **高影响**: `all` 选项能让 LoRA 在模型中更广泛、更全面地进行调整，通常能带来更稳定和有效的学习过程。相比只作用于少数几个模块，`all` 的收敛路径通常更优。 |
| **梯度累积 (Gradient Accumulation)**   | **优化手段**: **不增加**显存消耗。它是一种用计算时间换取显存空间的优化策略。 | **核心影响**: **稳定训练的基石**。它将多个小批次的梯度进行平均，极大地平滑了梯度更新的噪声，从而让 `批处理大小=1` 的训练过程变得可行且稳定。 |
| **学习率 (Learning Rate)**             | **无影响**                                                   | **核心影响**: **最敏感**的稳定性参数。过高导致训练发散（崩溃），过低导致训练停滞（不收敛）。`5e-5` 是 LoRA 微调的黄金参考值。 |
| **最大梯度范数 (Max Grad Norm)**       | **无影响**                                                   | **高影响**: **直接的稳定器**。通过梯度裁剪，防止因异常数据导致的“梯度爆炸”现象，为训练过程提供“安全护栏”。 |
| **学习率调度器 (LR Scheduler)**        | **无影响**                                                   | **高影响**: **过程稳定器**。`cosine` 调度器平滑地衰减学习率，能帮助模型在训练后期更稳定地收敛到最优解，避免在终点附近因学习率过高而“反复横跳”。 |
| **LoRA 缩放系数 (lora_alpha)**         | **无影响**: `alpha` 只是一个计算中的缩放因子，不直接分配显存。 | **中等影响**: `alpha` 与 `rank` 的**比率** (`alpha/rank`) 共同决定 LoRA 的生效强度。`32/16=2` 是一个标准且稳健的配置，能有效缩放 LoRA 的学习效果。过高的比率可能让模型调整过于剧烈，影响稳定性。 |
| **LoRA 随机丢弃 (lora_dropout)**       | **无影响**                                                   | **中等影响 (通过防止过拟合)**: Dropout 是一种正则化技术，用于提升模型的泛化能力和训练稳定性。设为 `0` 表示不使用。如果后续观察到过拟合，将其设为 `0.05` 或 `0.1` 是提升稳定性的有效手段。 |
| **部分参数微调设置**                   | **不适用 (Inactive)**: 由于下方已配置并启用了 LoRA 参数，当前微调方法为 `lora`。因此，这个属于 Freeze-Tuning 方法的设置模块是**无效**的，其设定值将被忽略。 | **不适用 (Inactive)**                                        |

- `cutoff_len` × `per_device_train_batch_size` → 显存消耗
- `learning_rate` × `optimizer` → 训练稳定性
- `lora_rank` × `lora_alpha( LoRA 缩放系数)` → LoRA表达能力

**三级依赖（优化效果）**：

`flash_attn` +`Unsloth`+ `enable_liger_kernel` → 计算效率

![加速方式](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202506201715617.png)

| **技术名称**       | **核心作用**                                                 | **启用参数**                |
| ------------------ | ------------------------------------------------------------ | --------------------------- |
| **FlashAttention** | 加快注意力机制的运算速度，同时减少对内存的使用。             | `flash_attn: fa2`           |
| **Unsloth**        | 支持多种大语言模型的 4-bit 和 16-bit QLoRA/LoRA 微调，在提高运算速度的同时还减少了显存占用。 | `use_unsloth: True`         |
| **Liger Kernel**   | 一个大语言模型训练的性能优化框架，可有效地提高吞吐量并减少内存占用。 | `enable_liger_kernel: True` |

- `deepspeed` → 分布式效率

  ![分布式训练](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202506201718893.png)

  **DeepSpeed ZeRO 阶段对比**

  | **ZeRO 阶段 (Stage)** | **划分内容 (Partitioned Components)** | **显存占用 (VRAM Usage)** | **训练速度 (Training Speed)** | **适用场景/建议 (Recommendation)**                         |
  | --------------------- | ------------------------------------- | ------------------------- | ----------------------------- | ---------------------------------------------------------- |
  | **ZeRO-1**            | 仅划分优化器参数。                    | 较高                      | **最快**                      | 显存充足时的首选，以获得最佳训练速度。                     |
  | **ZeRO-2**            | 划分优化器参数与梯度。                | 中等                      | 中等                          | 在显存不足以使用 ZeRO-1 时，一个兼顾速度与显存的平衡选择。 |
  | **ZeRO-3**            | 划分优化器参数、梯度与模型参数。      | **最低**                  | 较慢                          | 显存极度受限，或需要训练远超单卡容量的大模型时的选择。     |

  ------

  **关于 CPU Offload 的说明**

  | **参数**                | **作用与影响**                                               |
  | ----------------------- | ------------------------------------------------------------ |
  | **`offload_param=cpu`** | 通过将参数卸载到 CPU 内存，可以**大幅减小** GPU 显存需求，但同时会**极大地减慢**训练速度，因为数据需要在 GPU 和 CPU 之间来回传输。这是一个用极致的速度牺牲换取显存的最终手段。 |

## 3. 核心参数详解

### 3.1 微调参数（FinetuningArguments）

微调参数是 LLaMA Factory 的核心，决定了模型微调的方法和策略。

#### 3.1.1 基础微调参数

**`stage`** - 训练阶段

```yaml
# 参数定义
stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto"]
# 默认值: "sft"

# 各阶段详解
pt:   # 预训练 (Pre-training)
  - 用途: 从头训练或继续预训练
  - 数据: 大规模无标注文本
  - 目标: 学习语言建模能力
  
sft:  # 监督微调 (Supervised Fine-tuning)
  - 用途: 指令跟随和对话能力训练
  - 数据: 指令-回答对
  - 目标: 提升任务执行能力
  
rm:   # 奖励模型 (Reward Model)
  - 用途: 训练用于RLHF的奖励模型
  - 数据: 偏好对比数据
  - 目标: 学习人类偏好评估
  
ppo:  # 近端策略优化 (Proximal Policy Optimization)
  - 用途: 基于奖励模型的强化学习
  - 数据: 查询prompt集合
  - 目标: 优化生成策略
  
dpo:  # 直接偏好优化 (Direct Preference Optimization)
  - 用途: 无需奖励模型的偏好学习
  - 数据: 偏好对比数据
  - 目标: 直接优化人类偏好
  
kto:  # KTO偏好学习 (Kahneman-Tversky Optimization)
  - 用途: 基于前景理论的偏好优化
  - 数据: 二元偏好标签
  - 目标: 更精确的偏好建模
```

**生产环境配置建议**：
- **对话任务**：使用 `sft` 阶段，配合高质量对话数据
- **代码生成**：使用 `sft` 阶段，配合代码指令数据
- **人类对齐**：先 `sft` 后 `dpo`，构建两阶段训练流程
- **创新探索**：使用 `kto` 进行更精细的偏好学习

**`finetuning_type`** - 微调方法
```yaml
# 参数定义
finetuning_type: Literal["lora", "freeze", "full"]
# 默认值: "lora"

# 方法对比分析
lora:    # 低秩适应 (Low-Rank Adaptation)
  原理: 通过低秩矩阵近似权重更新
  优势: 参数高效、显存友好、训练快速
  适用: 资源受限、快速迭代、多任务适配
  参数量: 原模型的0.1%-1%
  
freeze:  # 冻结微调 (Freeze Fine-tuning)
  原理: 冻结大部分层，仅训练特定层
  优势: 保持预训练知识、防止灾难性遗忘
  适用: 领域适配、少量数据、知识保持
  参数量: 原模型的10%-50%
  
full:    # 全参数微调 (Full Fine-tuning)
  原理: 更新所有模型参数
  优势: 最大化性能提升、完全适配任务
  适用: 充足资源、关键应用、性能导向
  参数量: 原模型的100%
```

**性能-资源权衡分析**：

| 方法 | 显存需求 | 训练速度 | 性能上限 | 部署便利性 | 推荐场景 |
|------|----------|----------|----------|------------|----------|
| LoRA | 低 | 快 | 中-高 | 优秀 | 快速原型、资源受限 |
| Freeze | 中 | 中 | 中 | 良好 | 领域适配、知识保持 |
| Full | 高 | 慢 | 最高 | 一般 | 关键应用、性能优先 |

#### 3.1.2 LoRA 参数详解

LoRA（Low-Rank Adaptation）是目前最流行的参数高效微调方法，其核心思想是通过低秩矩阵分解来近似权重更新。

**理论基础**：
```
ΔW = BA
其中：B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)
最终权重：W' = W₀ + αBA
```

![image-20250620165149526](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202506201651656.png)

**核心参数配置**：

**`lora_rank`** - LoRA 秩
```yaml
# 参数定义
lora_rank: int
# 默认值: 8

# 秩选择策略
基础任务 (rank=8-16):
  - 简单对话、翻译、摘要
  - 数据量: <10K样本
  - 显存要求: 最低
  
中等任务 (rank=32-64):
  - 复杂推理、代码生成、多轮对话
  - 数据量: 10K-100K样本
  - 显存要求: 中等
  
复杂任务 (rank=128-256):
  - 专业领域、多模态、长文本
  - 数据量: >100K样本
  - 显存要求: 较高

# 理论分析
表达能力 ∝ rank × (d + k)
计算复杂度 ∝ rank × sequence_length
显存消耗 ∝ rank × model_layers
```

**`lora_alpha`** - LoRA 缩放系数

```yaml
# 参数定义
lora_alpha: Optional[int]
# 默认值: None (自动设为 lora_rank * 2)

# 缩放系数作用
实际学习率 = learning_rate × (lora_alpha / lora_rank)

# 配置策略
lora_alpha = lora_rank × 2:     # 标准配置
  - 适用于大多数任务
  - 平衡训练稳定性和学习能力
  
lora_alpha = lora_rank × 1:     # 保守配置  
  - 适用于敏感任务或小数据集
  - 降低过拟合风险
  
lora_alpha = lora_rank × 4:     # 激进配置
  - 适用于大数据集或需要快速收敛
  - 需要仔细监控训练稳定性
```

**`lora_dropout`** - LoRA Dropout率
```yaml
# 参数定义
lora_dropout: float
# 默认值: 0

# Dropout配置策略
0.0:      # 无Dropout
  - 适用于大数据集 (>50K样本)
  - 模型容量充分利用
  
0.05:     # 轻度Dropout
  - 适用于中等数据集 (10K-50K样本)
  - 轻微正则化效果
  
0.1-0.2:  # 中度Dropout
  - 适用于小数据集 (<10K样本)
  - 防止过拟合
```

**`lora_target`** - LoRA 目标模块
```yaml
# 参数定义
lora_target: str
# 默认值: "all"

# 模块选择策略
"all":                    # 全模块 (推荐)
  - 应用到所有线性层
  - 最大化表达能力
  
"q_proj,v_proj":         # 注意力模块
  - 仅应用到查询和值投影
  - 平衡性能和参数量
  
"q_proj,k_proj,v_proj":  # 完整注意力
  - 全注意力机制适配
  - 注意力模式优化
  
"gate_proj,up_proj":     # FFN模块
  - 仅应用到前馈网络
  - 知识存储优化
```

#### 3.1.3 高级 LoRA 算法

**`use_rslora`** - 秩稳定 LoRA
```yaml
# 参数定义
use_rslora: bool
# 默认值: False

# RSLoRA 原理
标准LoRA: ΔW = αBA/r
RSLoRA:   ΔW = αBA/√r

# 适用场景
- 大rank值 (r ≥ 64) 训练
- 训练不稳定情况
- 需要更好的rank扩展性
```

**`use_dora`** - 权重分解 LoRA
```yaml
# 参数定义
use_dora: bool
# 默认值: False

# DoRA 原理
W' = m · (W₀ + BA) / ||W₀ + BA||
其中 m 是可学习的幅度参数

# 性能提升
- 通常比标准LoRA提升2-5%性能
- 增加约10%的参数量和计算量
- 适用于性能敏感任务
```

**`pissa_init`** - PiSSA 初始化
```yaml
# 参数定义
pissa_init: bool
# 默认值: False

# PiSSA 原理
通过主成分分析初始化LoRA矩阵
W₀ = USVᵀ ≈ USᵣVᵣᵀ + USᵣ₊₁:ₙVᵣ₊₁:ₙᵀ

# 优势分析
- 更快的收敛速度
- 更好的初始化质量  
- 适用于复杂任务微调
```

### 3.2 训练参数（TrainingArguments）

训练参数控制模型训练的核心超参数，直接影响训练效果和计算效率。

#### 3.2.1 学习率调度

**`learning_rate`** - 基础学习率
```yaml
# 参数定义
learning_rate: float
# 默认值: 5e-5

# 学习率选择原理
LoRA微调: 5e-5 到 1e-4
  - LoRA层需要相对较大的学习率
  - 快速适配新任务需求
  
全参数微调: 1e-6 到 5e-6  
  - 避免破坏预训练知识
  - 渐进式参数调整
  
冻结微调: 1e-5 到 5e-5
  - 平衡新知识学习和知识保持
```

**`lr_scheduler_type`** - 学习率调度器
```yaml
# 参数定义
lr_scheduler_type: str
# 默认值: "linear"

# 调度器类型分析
linear:     # 线性衰减
  公式: lr(t) = lr₀ × (1 - t/T)
  特点: 简单稳定，适用于短期训练
  
cosine:     # 余弦退火
  公式: lr(t) = lr₀ × (1 + cos(πt/T))/2  
  特点: 平滑衰减，更好的收敛性质
  推荐: 生产环境首选
  
constant:   # 常数学习率
  公式: lr(t) = lr₀
  特点: 无衰减，适用于持续学习
  
polynomial: # 多项式衰减
  公式: lr(t) = lr₀ × (1 - t/T)^power
  特点: 可控制衰减速率
```

**`warmup_steps`** - 预热步数
```yaml
# 参数定义
warmup_steps: int
# 默认值: 0

# 预热策略
预热步数 = 总训练步数 × 预热比例

小数据集 (<1K样本):
  预热比例: 5-10%
  作用: 避免初期梯度爆炸
  
大数据集 (>10K样本):
  预热比例: 1-5%  
  作用: 稳定初期训练

# 预热类型
linear_warmup:     lr(t) = lr₀ × t/warmup_steps
cosine_warmup:     lr(t) = lr₀ × (1 - cos(πt/warmup_steps))/2
constant_warmup:   lr(t) = lr₀ × warmup_ratio
```

#### 3.2.2 批量大小与梯度处理

**批量大小原理**：
```
有效批量大小 = per_device_batch_size × gradient_accumulation_steps × num_gpus
```

**`per_device_train_batch_size`** - 单设备批量大小
```yaml
# 参数定义
per_device_train_batch_size: int
# 默认值: 1

# 显存-批量大小关系
batch_size = 1:  基础显存需求 (推荐起点)
batch_size = 2:  显存需求 +30%
batch_size = 4:  显存需求 +70%
batch_size = 8:  显存需求 +150%

# 选择策略
显存充足: 优先增加batch_size
显存紧张: 保持batch_size=1，通过gradient_accumulation_steps增加有效批量
```

**`gradient_accumulation_steps`** - 梯度累积步数
```yaml
# 参数定义
gradient_accumulation_steps: int
# 默认值: 1

# 梯度累积原理
∇L = (1/N) Σᵢ ∇Lᵢ
其中 N = batch_size × accumulation_steps

# 配置策略
小模型 (7B): accumulation_steps = 4-8
中模型 (14B): accumulation_steps = 8-16  
大模型 (70B): accumulation_steps = 16-32

# 优势分析
- 模拟大批量训练效果
- 避免显存不足问题
- 保持训练稳定性
```

**`max_grad_norm`** - 梯度裁剪
```yaml
# 参数定义
max_grad_norm: float
# 默认值: 1.0

# 梯度裁剪原理
if ||∇θ|| > max_grad_norm:
    ∇θ = ∇θ × max_grad_norm / ||∇θ||

# 参数选择
max_grad_norm = 1.0:    # 标准配置
  - 适用于大多数任务
  - 有效防止梯度爆炸
  
max_grad_norm = 0.5:    # 保守配置
  - 适用于不稳定训练
  - 更强的梯度约束
  
max_grad_norm = 2.0:    # 宽松配置
  - 适用于需要大梯度更新
  - 谨慎使用
```

### 3.3 数据参数（DataArguments）

数据参数控制数据集的加载、预处理和增强策略，对训练效果有重要影响。

#### 3.3.1 序列长度管理

**`cutoff_len`** - 序列截断长度
```yaml
# 参数定义
cutoff_len: int
# 默认值: 2048

# 长度选择策略
基于数据分析:
  P50长度: 基础参考
  P90长度: 平衡配置  
  P99长度: 最优配置

# 显存影响分析
序列长度对显存的影响呈二次关系：
Memory ∝ sequence_length²

cutoff_len = 1024:  基础显存需求
cutoff_len = 2048:  显存需求 ×4
cutoff_len = 4096:  显存需求 ×16
cutoff_len = 8192:  显存需求 ×64
```

**数据长度分析脚本**：
```bash
# 官方提供的长度分析工具
torchrun --nproc_per_node=1 \
    scripts/stat_utils/length_cdf.py \
    --model_name_or_path /path/to/model \
    --dataset your_dataset \
    --template your_template \
    --cutoff_len 8192

# 输出示例
50%ile: 512 tokens
90%ile: 1024 tokens  
95%ile: 2048 tokens
99%ile: 4096 tokens
99.9%ile: 8192 tokens
```

#### 3.3.2 数据增强策略

**`packing`** - 序列打包
```yaml
# 参数定义
packing: Optional[bool]
# 默认值: None (预训练时自动启用)

# 打包原理
将多个短序列连接成一个长序列:
[seq1][EOS][seq2][EOS][seq3][EOS]...

# 适用场景
预训练任务: 建议启用
  - 提高序列利用率
  - 减少padding浪费
  
微调任务: 根据情况
  - 对话任务: 不推荐 (语义隔离)
  - 代码任务: 可以考虑
```

**`neat_packing`** - 清洁打包
```yaml
# 参数定义  
neat_packing: bool
# 默认值: False

# 清洁打包特点
- 避免跨序列的注意力计算
- 保持序列间的语义隔离
- 提供打包的效率优势
- 避免cross-contamination
```

**`mix_strategy`** - 数据集混合策略
```yaml
# 参数定义
mix_strategy: Literal["concat", "interleave_under", "interleave_over"]
# 默认值: "concat"

# 混合策略对比
concat:           # 顺序连接
  - 按顺序使用各数据集
  - 简单直接，但可能导致数据不平衡
  
interleave_under: # 下采样交错
  - 按比例交错采样  
  - 确保数据平衡，推荐使用
  
interleave_over:  # 上采样交错
  - 重复采样小数据集
  - 适用于数据集大小差异巨大的情况
```

### 3.4 模型参数（ModelArguments）

模型参数控制预训练模型的加载、配置和优化设置。

#### 3.4.1 模型加载配置

**`model_name_or_path`** - 模型路径
```yaml
# 参数定义
model_name_or_path: Optional[str]

# 路径类型
本地路径: /path/to/local/model
HuggingFace Hub: meta-llama/Llama-2-7b-hf  
ModelScope: qwen/Qwen2.5-7B-Instruct
```

**`trust_remote_code`** - 信任远程代码
```yaml
# 参数定义
trust_remote_code: bool
# 默认值: False

# 安全考虑
True:  允许执行模型仓库中的自定义代码
  - 适用于: 最新模型、自定义架构
  - 风险: 代码执行安全风险
  
False: 仅使用Transformers内置代码
  - 适用于: 标准模型架构
  - 优势: 更高安全性
```

#### 3.4.2 性能优化配置

**`flash_attn`** - FlashAttention配置
```yaml
# 参数定义
flash_attn: Literal["auto", "disabled", "sdpa", "fa2"]
# 默认值: "auto"

# 选项对比
auto:     自动选择最佳实现
disabled: 禁用FlashAttention
sdpa:     使用PyTorch SDPA实现
fa2:      使用FlashAttention-2实现 (推荐)

# 性能提升
fa2 vs 标准注意力:
- 内存节省: 50-80%
- 速度提升: 150-300%
- 支持更长序列
```

**`enable_liger_kernel`** - Liger内核优化
```yaml
# 参数定义
enable_liger_kernel: bool  
# 默认值: False

# Liger Kernel优势
- 融合操作内核，减少内存访问
- 优化RMSNorm、RoPE、SwiGLU等操作
- 显存节省: 20-40%
- 速度提升: 10-30%

# 生产环境建议
推荐始终启用: enable_liger_kernel: true
```

## 4. 参数配置实践

### 4.1 基础配置模板

#### 4.1.1 对话任务配置
```yaml
# ===== 基础设置 =====
model_name_or_path: /models/Qwen2.5-7B-Instruct
stage: sft
finetuning_type: lora
template: qwen

# ===== 数据配置 =====
dataset: alpaca_zh
cutoff_len: 4096
train_on_prompt: false
mask_history: true

# ===== LoRA配置 =====
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all

# ===== 训练配置 =====
learning_rate: 5e-05
num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
max_grad_norm: 1.0
lr_scheduler_type: cosine
warmup_steps: 100

# ===== 优化配置 =====
bf16: true
flash_attn: fa2
enable_liger_kernel: true

# ===== 监控配置 =====
logging_steps: 10
save_steps: 500
eval_steps: 500
plot_loss: true
val_size: 0.1
```

#### 4.1.2 代码生成配置
```yaml
# ===== 基础设置 =====
model_name_or_path: /models/CodeLlama-7b-Python
stage: sft
finetuning_type: lora
template: codellama

# ===== 数据配置 =====
dataset: code_alpaca
cutoff_len: 2048
packing: true

# ===== LoRA配置 =====
lora_rank: 32
lora_alpha: 64
lora_target: "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# ===== 训练配置 =====
learning_rate: 1e-04
num_train_epochs: 2
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
```

### 4.2 硬件适配策略

#### 4.2.1 单卡配置（24GB显存）
```yaml
# 资源受限配置
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
cutoff_len: 2048
lora_rank: 16
enable_liger_kernel: true
```

#### 4.2.2 多卡配置（48GB显存×2）
```yaml
# 高性能配置  
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
cutoff_len: 4096
lora_rank: 32
ddp_find_unused_parameters: false
```

### 4.3 参数调优策略

#### 4.3.1 基线建立
```yaml
# 最小可行配置
model_name_or_path: /path/to/model
stage: sft
finetuning_type: lora
dataset: your_dataset
lora_rank: 8
learning_rate: 5e-05
num_train_epochs: 1
per_device_train_batch_size: 1
cutoff_len: 1024
```

#### 4.3.2 渐进优化
1. **确保基础运行**：最小配置成功训练
2. **优化序列长度**：根据数据分布调整`cutoff_len`
3. **调整学习率**：基于loss曲线fine-tune
4. **扩展LoRA能力**：增加`lora_rank`和`lora_alpha`
5. **优化批量大小**：最大化硬件利用率

## 5. 小结

本文详细解析了 LLaMA Factory 的参数体系架构和核心参数配置。主要结论包括：

1. **分层架构设计**：五层架构清晰分离不同层次的配置责任
2. **参数依赖关系**：理解参数间的约束关系是正确配置的基础  
3. **LoRA优化策略**：rank、alpha、dropout的科学配置方法
4. **训练超参数调优**：学习率、批量大小、梯度处理的最佳实践
5. **数据处理优化**：序列长度、打包策略、混合方法的选择原则

在下篇文章中，我们将深入探讨显存优化策略、高级算法配置、生产环境部署等高级主题。

## 参考文献

1. LLaMA Factory Documentation. https://llamafactory.readthedocs.io/
2. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685.
3. Liu, S., et al. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation. arXiv preprint arXiv:2402.09353.
4. Hao, Y., et al. (2024). LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models. arXiv preprint arXiv:2403.13372. 