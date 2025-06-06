# Easy Dataset × LLaMA Factory: 让大模型在海光DCU k100-AI上高效学习领域知识

[Easy Dataset](https://github.com/ConardLi/easy-dataset) 是一个专为创建大型语言模型（LLM）微调数据集而设计的应用程序。它提供了直观的界面，用于上传特定领域的文件，智能分割内容，生成问题，并为模型微调生成高质量的训练数据。支持使用 OpenAI、DeepSeek、火山引擎等大模型 API 和 Ollama 本地模型调用。

[LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) 是一款开源低代码大模型微调框架，集成了业界最广泛使用的微调技术，支持通过 Web UI 界面零代码微调大模型，目前已经成为开源社区最热门的微调框架之一，GitHub 星标超过 4.6 万。支持全量微调、LoRA 微调、以及 SFT 和 DPO 等微调算法。

**🚀 本教程专门针对海光DCU k100-AI加速卡优化**，使用 Easy Dataset 从五家互联网公司的公开财报构建 SFT 微调数据，并使用 LLaMA Factory 微调 Qwen2.5-3B-Instruct 模型，充分发挥DCU硬件的计算优势，使微调后的模型能学习到财报数据集中的知识。

## 📋 DCU k100-AI 运行环境要求

### 🔧 硬件配置
- **DCU型号**：海光 k100-AI 加速卡
- **显存要求**：≥ 16 GB HBM2E（k100-AI标配64GB）
- **系统内存**：≥ 64 GB DDR4
- **存储空间**：≥ 100 GB SSD（用于模型和数据存储）

### 💻 软件环境
- **操作系统**：Ubuntu 22.04.4 LTS
- **内核版本**：5.15.0-94-generic
- **DCU驱动**：ROCK 6.3.8
- **DTK版本**：25.04
- **Python版本**：3.10
- **Docker版本**：28.1.1+
- **Easy Dataset版本**：1.2.3
- **LLaMA Factory版本**：0.9.3.dev0 

### ⚡ DCU性能特性
**k100-AI具有以下优势：**

- **高带宽内存**：HBM2E提供更高的内存带宽
- **超大容量显存**：64GB显存支持超大模型训练和微调
- **优化架构**：专为AI工作负载设计的计算单元
- **生态兼容**：兼容PyTorch、PaddlePaddle等主流框架

## 🛠️ 使用 Easy Dataset 生成微调数据

### 📦 安装 Easy Dataset

#### 方法一：使用Docker（推荐用于DCU环境）

**为DCU环境优化的Docker方式**，避免依赖冲突：

```bash
# 1. 拉取Easy Dataset仓库
git clone https://github.com/ConardLi/easy-dataset.git
cd easy-dataset

# 2. 构建DCU优化的Docker镜像
cat > Dockerfile.dcu << 'EOF'
FROM ubuntu:22.04

# 设置时区和语言环境
ENV TZ=Asia/Shanghai
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    python3 \
    python3-pip \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# 安装pnpm
RUN npm install -g pnpm

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . .

# 安装依赖
RUN pnpm install

# 构建项目
RUN pnpm build

# 暴露端口
EXPOSE 1717

# 启动命令
CMD ["pnpm", "start"]
EOF

# 3. 构建镜像
docker build -f Dockerfile.dcu -t easy-dataset:dcu .

# 4. 运行容器（映射数据目录到宿主机）
docker run -d \
    -p 1717:1717 \
    -v $(pwd)/local-db:/app/local-db \
    -v $(pwd)/datasets:/app/datasets \
    --name easy-dataset-dcu \
    easy-dataset:dcu
```

#### 方法二：直接安装（适用于开发环境）

```bash
# 1. 确保Node.js版本 >= 18.0
# https://nodejs.org/en/download
node -v  # 应显示 v22.14.0

# 2. 克隆仓库
git clone https://github.com/ConardLi/easy-dataset.git
cd easy-dataset

# 3. 安装依赖（使用国内镜像加速）
# 安装pnpm
curl -fsSL https://get.pnpm.io/install.sh | sh -
npm config set registry https://registry.npmmirror.com
pnpm config set registry https://registry.npmmirror.com
pnpm install

# 4. 构建并启动
pnpm build
pnpm start
```

**✅ 启动成功标志**：
```bash
> easy-dataset@1.2.3 start
> next start -p 1717

  ▲ Next.js 14.2.25
  - Local:        http://localhost:1717
  - Network:      http://0.0.0.0:1717

 ✓ Ready in 287ms
```

### 📊 示例数据下载

本教程提供了专门针对中国互联网公司财报的高质量数据集：

```bash
# 下载财报数据集
git clone https://github.com/llm-factory/FinancialData-SecondQuarter-2024.git
cd FinancialData-SecondQuarter-2024

# 查看数据集结构
ls -la
# 包含：阿里巴巴、腾讯、京东、美团、快手等公司2024年Q2财报
```

**📈 数据集特点**：

- **高质量内容**：专业财经分析师整理
- **结构化信息**：包含营收、利润、业务分析等
- **时效性强**：2024年最新财报数据
- **领域专业**：互联网行业深度洞察

### 🎯 微调数据生成

#### 第一步：创建项目并配置参数

> 在浏览器进入 Easy Dataset 主页后，点击**创建项目**

1. **项目初始化**

   ```
   浏览器访问：http://localhost:1717
   点击"创建项目" → 输入项目名称："DCU-Financial-Dataset"
   ```

2. **模型配置**

   > 项目创建后会跳转到**项目设置**页面，打开**模型配置**，选择数据生成时需要调用的大模型 API 接口

   ![模型配置](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202506061805814.png)

   > 这里以 QWEN模型为例，修改模型**提供商**和**模型名称**，填写 **API 密钥**，点击**保存**后将数据保存到本地，在右上角选择配置好的模型

   ![image-20250606181008317](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202506061810423.png)

   - **提供商**：阿里云百炼（推荐，性价比高）
   - **模型名称**：qwen-max-latest
   - **API密钥**：[阿里云百炼API KEY](https://bailian.console.aliyun.com/?tab=model#/api-key)
   - **请求配置**：
     
     ```json
     {
       "temperature": 0.7,
       "max_tokens": 8192
     }
     ```

3. **任务配置**

   > 打开**任务配置**页面，设置文本分割长度为最小 500 字符，最大 1000 字符。在问题生成设置中，修改为每 10 个字符生成一个问题，修改后在页面最下方**保存任务配置**

   ![任务配置](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202506061821826.png)

   ```
   文本分割设置：
   - 最小长度：800字符
   - 最大长度：1500字符（保证上下文完整性）
   
   问题生成设置：
   - 生成密度：每240个字符1个问题（平衡质量与数量）
   ```

#### 第二步：批量处理数据文件

1. **并行上传文件**

   > 1. 打开**文献处理**页面，选择并上传示例数据文件，选择文件后点击**上传并处理文件**
   > 2. 上传后会调用大模型解析文件内容并分块，耐心等待文件处理完成，示例数据通常需要 2 分钟左右

   ```
   同时上传多个文件：
   - 选择所有财报文件（txt + markdown格式）
   ```

2. **生成微调数据**

   > 1. 待文件处理结束后，可以看到文本分割后的文本段，选择全部文本段，点击**批量生成问题**
   > 2. 点击后会调用大模型根据文本块来构建问题，耐心等待处理完成。视 API 速度，处理时间可能在 20-40 分钟不等
   > 3. 处理完成后，打开**问题管理**页面，选择全部问题，点击**批量构造数据集**，耐心等待数据生成。视 API 速度，处理时间可能在 20-40 分钟不等

   ![批量生成问题](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202506061838184.png)

   ![批量构造数据集](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202506061839250.png)

#### 第四步：导出LLaMA Factory格式的数据集

> 1. 答案全部生成结束后，打开**数据集管理**页面，点击**导出数据集**
> 2. 在导出配置中选择**在** **LLaMA Factory** **中使用**，点击**更新** **LLaMA Factory** **配置**，即可在对应文件夹下生成配置文件，点击**复制**按钮可以将配置路径复制到粘贴板。
> 3. 在配置文件路径对应的文件夹中可以看到生成的数据文件，其中主要关注以下三个文件
>    1. dataset_info.json：LLaMA Factory 所需的数据集配置文件
>    2. alpaca.json：以 Alpaca 格式组织的数据集文件
>    3. sharegpt.json：以 Sharegpt 格式组织的数据集文件
>
> 其中 alpaca 和 sharegpt 格式均可以用来微调，两个文件内容相同。

![导出LLaMA Factory格式的数据集](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202506062215106.png)

**文件结构验证**

```bash
ls -la exported/
# dataset_info.json  - LLaMA Factory配置
# alpaca.json       - Alpaca格式数据
# sharegpt.json     - ShareGPT格式数据
# 其中 alpaca 和 sharegpt 格式均可以用来微调，两个文件内容相同。
```

## 🚀 使用 LLaMA Factory 在DCU k100-AI上微调模型

### 📦 安装DCU优化的LLaMA Factory

#### 环境准备

```bash
# 1. 创建DCU专用虚拟环境
conda create -n dcu_env python=3.10
conda activate dcu_env

# 2. 安装DCU版本PyTorch
# pip install https://download.sourcefind.cn:65024/file/4/pytorch/DAS1.5/torch-2.4.1+das.opt2.dtk2504-cp310-cp310-manylinux_2_28_x86_64.whl

# 3. 验证DCU环境
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'DCU可用: {torch.cuda.is_available()}')
print(f'DCU设备数: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'当前DCU: {torch.cuda.get_device_name(0)}')
    print(f'显存容量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

#### 安装LLaMA Factory

```bash
# 1. 克隆仓库（使用国内镜像）
git clone https://gitee.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 2. 安装依赖（DCU优化版本）
pip install -e ".[torch,metrics]" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 验证安装
llamafactory-cli version
```

**✅ 安装成功输出**：
```bash
(dcu_env) root@Ubuntu2204:~/AI-BOX/code/dcu# llamafactory-cli version
[2025-06-06 22:22:17,021] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
----------------------------------------------------------
| Welcome to LLaMA Factory, version 0.9.3.dev0           |
|                                                        |
| Project page: https://github.com/hiyouga/LLaMA-Factory |
----------------------------------------------------------
```

### 🎯 启动LLaMA Factory的微调任务

#### 启动LLaMA Board

```bash
# DCU环境变量设置
export CUDA_VISIBLE_DEVICES=0  # 指定使用第一块DCU
export USE_MODELSCOPE_HUB=1   # 使用国内模型源

# 启动Web UI
llamafactory-cli webui
```

**🌐 访问地址**：http://localhost:7860

#### 微调参数配置

1. **模型选择与优化**

   > 进入 Web UI 界面后，选择模型为 Qwen2.5-3B-Instruct，模型路径可填写本地绝对路径，不填则从互联网下载

   ```bash
   模型：Qwen2.5-3B-Instruct
   模型路径：/path/to/models/Qwen2.5-3B-Instruct
   
   DCU优化设置：
   - 精度：bf16（k100-AI原生支持）
   - Flash Attention：启用（减少显存占用）
   - Gradient Checkpointing：启用（大模型必备）
   ```

2. **数据配置**

   > 将**数据路径**改为使用 Easy Dataset 导出的配置路径，选择 Alpaca 格式数据集

   ```bash
   数据路径：/path/to/exported（如/root/AI-BOX/code/dcu/easy-dataset/local-db/BRSJUcZdjjho）
   数据集：选择生成的财报数据集
   数据格式：alpaca
   
   预处理优化：
   - Max Length：2048（充分利用DCU显存）
   - Cutoff Length：1024
   - Preprocessing：8进程并行
   ```

3. **训练参数**

   > 为了让模型更好地学习数据知识，将**学习率**改为 1e-4，**训练轮数**提高到 8 轮。批处理大小和梯度累计则根据设备显存大小调整，在显存允许的情况下提高批处理大小有助于加速训练，**一般保持批处理大小×梯度累积×显卡数量等于 32 即可**

   ```bash
   基础参数：
   - 学习率：1e-4（k100-AI优化值）
   - 训练轮数：8轮
   - 批处理大小：2（64GB显存最优）
   - 梯度累积：16步
   - 有效批处理：1×16=32
   
   内存优化：
   - LoRA 秩：16（平衡效果与效率）
   - LoRA 缩放系数：32
   - LoRA 随机丢弃：0
   - LoRA+ 学习率比例：0
   
   ```

![微调参数配置](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202506062305466.png)

#### 1. 核心选择（要做什么？）

![](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202506062325641.png)

这部分决定了我们选择哪个“大学生”以及用什么“培训方法”。

- 模型名称 (Model Name)
  - **`Qwen2-5.7B-Instruct`**: 这是你选择的**基础模型**。Qwen2（通义千问2）是阿里巴巴开发的大模型系列，`5.7B` 代表它有大约57亿个参数（可以理解为模型的复杂程度或“脑容量”），`Instruct` 表示这是一个经过指令微调的版本，本身就比较擅长听从指令。
- 微调方法 (Finetuning Method)
  - `lora`: 这是目前最流行的一种**参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）**方法。
    - **小白理解**：传统的“全量微调”像是给大学生“重塑大脑”，计算量巨大，需要非常好的显卡。而 `LoRA` 就像是给了大学生一个“**特殊技能笔记本**” 📝。我们不改变他原来的大脑，只让他在这个小本本上记录和学习新技能。这个“笔记本”很小，所以训练起来非常快，对显卡要求也低很多。训练完成后，把这个“笔记本”和他原本的大脑结合，他就掌握了新技能。这是新手入门微调的**首选方法**。

------

#### 2. 训练数据（学什么？）

这部分定义了模型的“学习材料”。

- 数据路径 (Dataset Path)
  - `/root/AI-BOX/code/dcu/easy-dataset/local-db/BRSJUcZdjjho` 和 `[Easy Dataset][BRSJUcZdjjho]Alpaca` : 这里指定了你的数据集，也就是给模型学习的“教材”。
  - **关键点**：**数据质量决定了微调效果的上限**。好的数据能让模型学到真正的能力，而差的数据只会让模型“学坏”。

------

#### 3. 核心训练参数（怎么学？）

![](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202506062326513.png)

这部分是最关键的超参数，它们共同控制着整个“学习过程”。

- **学习率 (Learning Rate)**
  - **含义**: 模型在学习中每一步“更新知识”的幅度，可以理解为“**学习的步子大小**” 🏃。
  - **比喻**: 步子太大（学习率过高），容易“扯着蛋”，学不扎实，甚至可能越学越差（知识跑偏）。步子太小（学习率过低），学习速度太慢，要花很长时间才能学会。`1e-4` (即 0.0001) 是 LoRA 微调中一个比较常用的初始值。
- **训练轮数 (Epochs)**
  - **含义**: 把整个“教材”（数据集）从头到尾完整学习的次数。
  - **比喻**: `Epochs = 8.0` 就意味着让学生把整本教材读8遍。对于大型数据集，通常 1-3 个 Epoch 就足够了，太多了模型可能会“死记硬背”，失去泛化能力。
- **批处理大小 (Batch Size)**
  - **含义**: `per_device_train_batch_size`，即每张显卡一次性“看”多少条数据。
  - **比喻**: 学生一次看 16 道例题，然后总结一下规律，更新自己的知识。这个值越大，训练过程越稳定，但对**显存（VRAM）**的占用也越大。如果你的显存不够，就需要调小这个值。
- **梯度累积步(Gradient Accumulation)**
  - **含义**: 这是一个“**省显存的技巧**”。它可以让你在不增加显存消耗的情况下，达到和大 Batch Size 同样的效果。
  - **比喻**: 你的显存一次只能处理 4 条数据（Batch Size=4），但你希望达到一次看 16 条数据的稳定效果。你可以设置梯度累积步数为 4 (`16 / 4 = 4`)。模型会先看4条，再看4条... 看完四次（共16条）后，再统一总结、更新知识。
  - **公式**: `有效批处理大小 = 批处理大小 × 梯度累积步数`。
- **截断长度 (Cutoff Length)**
  - **含义**: 模型在一次处理中能够“看到”的文本最大长度（Token数量）。
  - **比喻**: 学生一次能读的文章长度上限。超过这个长度的文本会被截断。这个值越大，模型能处理的上下文越长，但同样会**消耗更多显存**。
- **学习率调节器 (Learning rate scheduler)`cosine`**
  - **含义**: 在整个训练过程中，动态调整“学习步子大小”的策略。
  - **比喻**: `cosine` 策略就像一个先快后慢的学习计划。刚开始时学习率较高（步子大），让模型快速进入状态；随着训练的进行，学习率会像余弦曲线一样平滑地下降，让模型在后期“精雕细琢”，稳定地收敛到最佳状态。这是一个非常常用且有效的策略。
- 计算类型(Compute type)`bf16`
  - **含义**: 训练时使用的数字精度。
  - `fp32` (32位浮点数): 全精度，最准确，但占用显存最大，速度最慢。
  - `fp16` (16位浮点数): 半精度，显存占用减半，速度加快，但在某些情况下可能导致训练不稳定。
  - `bf16` (16位脑浮点数): 也是半精度，但对数值范围的表达能力比 `fp16` 强，不容易出错。

------

#### 4. LoRA 特定参数（“笔记本”的规格）

![image-20250606232641497](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202506062326636.png)

这部分参数是 LoRA 方法独有的，用来定义那个“特殊技能笔记本”的属性。

- **LoRA 秩 (lora_rank / r)**

  - **含义**: LoRA 模块的“**核心维度**”，可以理解为“笔记本的厚度”或“复杂程度”。
  - **比喻**: `r` 越大，笔记本越厚，能记录的“新知识”就越复杂、越丰富。但同时，它占用的计算资源和显存也会增加。`8`, `16`, `32`, `64` 是常见取值。对于大多数任务，`16` 或 `32` 是一个不错的起点。

- **LoRA 缩放系数 (lora_alpha)**

  - **含义**: 一个缩放因子，用来调整 LoRA 模块（笔记本）对模型原始权重的影响力。`lora_alpha` 越大，模型在做决策时，就越重视这本笔记本里新学到的知识。
  - **经验法则**: 通常设为 `lora_rank` 的两倍（`alpha = 2 * r`）。这是一种被广泛验证有效的设置。

  > **新手入门**: **严格遵守 `alpha = 2 * rank` 的原则**。这是一个非常安全、有效且省心的起点。你甚至可以设置 `alpha = rank`，让缩放因子恒为 1。先不要把 `alpha` 作为一个独立变量来调整。
  >
  > **进阶调优**: 当你对模型和任务有了更深的理解后，可以尝试打破这个规则。
  >
  > - 如果发现模型微调后“固执己见”，仍然保留了太多原始模型的行为，可以尝试**提高 `alpha` 的比例**（比如 `alpha = 4 * rank`），让模型更重视新学到的知识。
  > - 如果发现模型“学过头了”，完全忘记了原始的通用能力（发生了“灾难性遗忘”），可以尝试**降低 `alpha` 的比例**（比如 `alpha = 0.5 * rank`），减小微调数据的影响。

------

#### 给小白的总结与建议 🚀

1. **从 LoRA 开始**：对于新手，微调方法永远首选 `lora`，它能在消费级显卡上实现很好的效果。
2. **关注核心参数**：重点调整 **学习率 (Learning Rate)**、**批处理大小 (Batch Size)** 和 **训练轮数 (Epochs)**。刚开始时，可以参考网上成熟的方案，不要一次性修改太多参数。
3. **显存是关键**：很多参数（如 Batch Size, Cutoff Length, LoRA Rank）都会影响显存。如果遇到 "Out of Memory" 错误，优先考虑降低这几个值。
4. **数据为王**：记住，再好的调参技巧也弥补不了垃圾数据。微调的上限由你的数据质量决定。
5. **先跑通，再优化**：第一次微调，目标是**让流程完整地跑下来**。使用默认或推荐的参数，用少量数据（比如100条）先测试一下。成功运行后，再用完整数据、逐步调整参数来追求更好的效果。

希望这份详细的解释能帮助你踏出大模型微调的第一步！

### 监控微调过程

```bash
#  实时监控DCU使用情况
watch -n 0.1 "rocm-smi"
```

点击**开始**按钮，等待模型下载，一段时间后应能观察到训练过程的损失曲线

![练过程的损失曲线](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202506062352943.png)

##### 阶段一：`23:02:20` - 环境启动与准备

代码段

```
(dcu_env) root@Ubuntu2204:~/AI-BOX/code/dcu/llama-factory# llamafactory-cli webui
[2025-06-06 23:02:20,386] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Visit http://ip:port for Web UI, e.g., http://127.0.0.1:7860
* Running on local URL:  http://0.0.0.0:7860
```

- **动作**: 你执行了 `llamafactory-cli webui` 命令。

- 日志含义

  :

  - `Setting ds_accelerator to cuda (auto detect)`: 系统自动检测到你拥有 NVIDIA GPU (CUDA 环境)，并准备使用它进行加速。这是成功的第一步。
  - `Running on local URL: http://0.0.0.0:7860`: LLaMA Factory 的网页界面已经成功启动。你可以通过服务器的 IP 地址加上 `7860` 端口在浏览器中访问它。

------

#### 阶段二：`23:02:39` ~ `23:06:08` - 模型下载

代码段

```bash
[INFO|2025-06-06 23:02:39] llamafactory.hparams.parser:401 >> Process rank: 0, world size: 1, device: cuda:0, distributed training: False, compute dtype: torch.bfloat16
Downloading Model from https://www.modelscope.cn to directory: /root/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct
...
Downloading [model-00001-of-00002.safetensors]: 100%|██████████| 3.70G/3.70G [03:26<00:00, 19.3MB/s]
...
2025-06-06 23:06:08,245 - modelscope - INFO - Download model 'Qwen/Qwen2.5-3B-Instruct' successfully.
```

- **动作**: 你在 Web UI 界面上选择了 `Qwen/Qwen2.5-3B-Instruct` 模型，并点击了“开始”或“预览命令”，触发了模型的加载流程。
- 日志含:
  - `Process rank: 0, world size: 1`: 这表示你正在使用单张显卡进行训练。
  - `compute dtype: torch.bfloat16`: 确认了训练将使用 `bfloat16` 精度，这在现代 GPU 上能很好地平衡效率和稳定性。
  - `Downloading Model from https://www.modelscope.cn ...`: 程序发现本地没有这个模型，于是从魔搭（ModelScope）社区开始下载。这个过程耗时约 3 分半，下载了模型权重 (`.safetensors` 文件)、配置文件等。
  - `successfully`: 模型已成功下载到本地缓存目录 `/root/.cache/modelscope/hub/`，下次再使用就不用重新下载了。

------

#### 阶段三：`23:06:08` - 数据处理与Token化

```bash
[INFO|2025-06-06 23:06:08] llamafactory.data.loader:143 >> Loading dataset alpaca.json...
Generating train split: 120 examples [00:00, 9694.26 examples/s]
Running tokenizer on dataset (num_proc=16): 100%|██████████| 120/120 [00:02<00:00, 54.41 examples/s]
training example:
input_ids: [151644, 8948, 198, ...]
inputs: <|im_start|>system\nYou are Qwen...
label_ids: [-100, -100, -100, ..., 30440, 99677, ...]
labels: 可灵AI在快手未来战略...<|im_end|>
```

- **动作**: 模型下载完毕，开始处理你指定的数据集 `alpaca.json`。

- 日志含义 (非常关键！):

  - `Loading dataset alpaca.json...`: 加载了你的数据集，日志显示共有 120 条样本。

  - `Running tokenizer on dataset`: 正在将文本数据转换为模型可以理解的数字（Token ID）。

  - `training example`

    : 这里展示了一条处理好的数据样本，完美解释了监督微调的原理。

    - `inputs`: 这是完整的输入文本，包含了 system prompt (系统指令)、user prompt (用户问题)。
    - `labels`: 这是模型需要学习回答的内容，也就是你的答案部分。
    - `input_ids`: 是 `inputs` 文本被 Tokenizer 转换后的数字序列。
    - `label_ids`: **这是精髓所在**。你会发现它的开头有大量的 `-100`。`-100` 是一个特殊的标签，它告诉模型：“这部分内容（也就是你的问题）你只需要读，作为上下文理解，但**不需要计算它的损失**，你不需要为这部分内容学着去生成。” 模型只会根据非 `-100` 的部分（也就是 `labels` 对应的部分）来计算损失和学习。

> `label_ids` 中的 `-100` 是一个特殊的“**占位符**”或“**遮罩 (Mask)**”，它的唯一作用就是告诉模型：“**在计算损失（衡量对错）时，请完全忽略这个位置的token。**”
>
> 可以把它想象成学生考试卷上的“**此题不计分**”标记。

------

#### 阶段四：`23:06:14` ~ `23:06:17` - 加载模型与注入LoRA

```bash
[INFO|modeling_utils.py:3723] 2025-06-06 23:06:14,165 >> loading weights file ...
Loading checkpoint shards: 100%|██████████████████████████████████████| 2/2 [00:02<00:00,  1.46s/it]
...
[INFO|2025-06-06 23:06:17] llamafactory.model.adapter:143 >> Fine-tuning method: LoRA
[INFO|2025-06-06 23:06:17] llamafactory.model.loader:143 >> trainable params: 29,933,568 || all params: 3,115,872,256 || trainable%: 0.9607
```

- **动作**: 数据准备就绪，现在正式将模型加载到 GPU 显存中，并应用 LoRA 配置。

- 日志含义:

  - `Loading checkpoint shards`: 正在将刚才下载的 3B 模型的权重文件加载进来。

  - `Fine-tuning method: LoRA`: 确认了本次微调使用 LoRA 方法。

  - `trainable params: 29,933,568 || all params: 3,115,872,256 || trainable%: 0.9607`: 

    这是最能体现 LoRA 优势的一行日志！

    - `all params`: 基础模型总共有约 31 亿个参数。
    - `trainable params`: 我们实际需要训练的参数只有约 3000 万个。
    - `trainable%`: 需要训练的参数量**不到总参数量的 1%**！这极大地降低了对计算资源和显存的需求。

------

#### 阶段五：`23:06:18` ~ `23:09:08` - 核心：执行训练

```bash
[INFO|trainer.py:2243] 2025-06-06 23:06:18,499 >> ***** Running training *****
[INFO|trainer.py:2245] 2025-06-06 23:06:18,499 >>   Num Epochs = 8
[INFO|trainer.py:2246] 2025-06-06 23:06:18,499 >>   Instantaneous batch size per device = 2
[INFO|trainer.py:2249] 2025-06-06 23:06:18,499 >>   Total train batch size (w. parallel, distributed & accumulation) = 32
[INFO|trainer.py:2250] 2025-06-06 23:06:18,499 >>   Gradient Accumulation steps = 16
[INFO|trainer.py:2251] 2025-06-06 23:06:18,500 >>   Total optimization steps = 24
...
[INFO|2025-06-06 23:06:59] llamafactory.train.callbacks:143 >> {'loss': 1.8934, 'learning_rate': 8.9668e-05, 'epoch': 1.33, ...}
...
[INFO|2025-06-06 23:08:07] llamafactory.train.callbacks:143 >> {'loss': 1.5472, 'learning_rate': 3.0866e-05, 'epoch': 4.00, ...}
...
Training completed. ...
***** train metrics *****
  epoch                   =        6.4
  train_loss              =     1.6646
  train_runtime           = 0:02:50.08
```

- **动作**: 一切准备就绪，训练正式开始。
- 日志含义:
  - `***** Running training *****`: 训练循环启动的标志。
  - 这里列出了我们讨论过的核心参数：`Epochs=8`, `batch size=2`, `Gradient Accumulation=16`。注意 `Total train batch size = 2 * 16 = 32`，符合我们之前讨论的经验法则。
  - `'loss': 1.8934 ... 'loss': 1.5472`: 这是**训练过程中的核心监控指标**。“loss”（损失）代表模型的预测与真实答案之间的差距，这个值越低越好。可以看到，随着训练的进行，loss 从 1.89 稳步下降到 1.54，**这表明模型确实在学习并且学得很好！**
  - `'learning_rate'`: 学习率也在按照 `cosine` 策略平滑下降。
  - `Training completed`: 训练成功结束！
  - `***** train metrics *****`: 训练的最终总结。总共耗时约 2 分 50 秒，最终的平均 `train_loss` 为 1.6646。

------

#### 阶段六：`23:09:08` ~ `23:09:09` - 收尾：保存结果

```bash
[INFO|trainer.py:3705] 2025-06-06 23:09:08,595 >> Saving model checkpoint to saves/Qwen2.5-3B-Instruct/lora/train_2025-06-06-22-48-42
...
Figure saved at: saves/Qwen2.5-3B-Instruct/lora/train_2025-06-06-22-48-42/training_loss.png
```

- **动作**: 保存训练成果。
- 日志含义:
  - `Saving model checkpoint to ...`: 系统正在将训练好的 LoRA 适配器（也就是那个“特殊技能笔记本”）保存到指定的目录中。这是你本次训练最有价值的产出。
  - `training_loss.png`: 系统还为你绘制了损失曲线图并保存，方便你直观地分析训练过程。

------

#### 阶段七：`23:31:56` 以后 - 应用：加载微调后的模型进行测试

```bash
Downloading Model from https://www.modelscope.cn ... (This is just checking cache)
...
[INFO|2025-06-06 23:32:10] llamafactory.model.adapter:143 >> Merged 1 adapter(s).
[INFO|2025-06-06 23:32:10] llamafactory.model.adapter:143 >> Loaded adapter(s): saves/Qwen2.5-3B-Instruct/lora/train_2025-06-06-22-48-42
```

- **动作**: 在训练完成后，你可能切换到了“聊天”或“评估”页面，准备测试模型效果。
- 日志含义:
  - `Merged 1 adapter(s)` 和 `Loaded adapter(s)` 是关键。这表明程序重新加载了**原始的** `Qwen2.5-3B-Instruct` 模型，然后将你刚刚训练好的、保存在 `saves/...` 目录下的 LoRA 适配器权重**合并**了进去。
  - 现在，你得到的模型就是一个**既拥有通义千问原有强大能力，又精通你投喂的 `alpaca.json` 数据特定风格和知识**的、更强大的定制化模型了。

这份日志清晰地展示了一次非常成功和标准的 LoRA 微调流程：

1. **环境就绪**：成功启动并检测到 GPU。
2. **准备资源**：自动下载了基础模型。
3. **处理数据**：正确地加载、转换了你的数据集，并应用了 `-100` 标签来忽略提示词的损失。
4. **高效训练**：使用 LoRA 方法，只训练了不到 1% 的参数，并且从 Loss 的稳步下降可以看出学习过程非常健康。
5. **保存成果**：成功保存了训练好的 LoRA 适配器。
6. **应用测试**：成功加载了微调后的模型，可以开始对话测试了。

### 🧪 验证微调效果

#### 模型加载与测试

> 1. 选择**检查点路径**为刚才的输出目录，打开 **Chat** 页面，点击**加载模型**
> 2. 在下方的对话框中输入问题后，点击提交与模型进行对话，经与原始数据比对发现微调后的模型回答正确

![模型加载与测试](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202506062336362.png)

```bash
测试问题示例：
Q: "快手2024年第二季度的净利润增长率是多少？"


微调后回答：快手2024年第二季度的净利润增长率约为48.7%。这一增长主要得益于快手的电商业务以及其在广告业务上的表现。
```

点击**卸载模型**将微调后的模型卸载，清空**检查点路径**，点击**加载模型**加载微调前的原始模型

![](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202506062341841.png)

```bash
测试问题示例：
Q: "快手2024年第二季度的净利润增长率是多少？"


微调前回答：关于快手2024年第二季度的具体财务数据，包括净利润增长率，我目前无法直接获取最新的官方披露信息。通常这类详细财务数据需要参考快手发布的财报报告。由于我无法访问互联网以获取最新的实时数据，建议直接访问快手的官方网站或查阅其最新发布的财报报告以获得准确信息。如果您需要了解相关信息，我也可以帮助您了解如何查找这些信息。
```

## 💡常见问题

### 1. 显存不足

```bash
# 解决方案
- 减小batch_size
- 启用gradient_checkpointing
- 使用deepspeed zero-2
```

### 2. 训练速度慢

```bash
# 优化方案
- 启用flash_attention
- 调整DataLoader参数
- 使用混合精度训练
```

### 3. 模型效果不佳

```bash
# 改进策略
- 增加训练轮数
- 调整学习率
- 扩充训练数据
```

### 4. 目前微调的 `train_loss: 1.6646` 是不是有点高？

**直接回答：不，对于你这个任务来说，1.66 的 Loss 不仅不高，而且是一个相当健康和合理的结果。**

“Loss 值是高是低”是一个相对概念，**绝对不能只看数字本身**，它至少取决于以下几个因素：

1. **基础模型的能力**：不同的模型起点不同，其初始的 Loss 也千差万别。
2. **任务的复杂度**：如果你微调的数据集非常专业、晦涩，充满了不常见的词汇和逻辑，那么 Loss 自然会比微调简单问答数据要高。模型需要更努力才能学会。
3. **词汇表大小 (Vocabulary Size)**：**这是最关键的一点**。Loss 的计算与词汇表大小密切相关。在你日志中，Qwen2 模型的 `vocab_size` 是 **151,936**。这意味着模型在预测下一个词时，是从超过15万个选项中选一个！这是一个极其庞大的分类任务。在这种情况下，能达到 1.x 的 Loss 已经非常出色了。

为了更直观地理解，我们可以把 Loss 转换成另一个指标——**困惑度 (Perplexity, PPL)**。

- **计算公式**: PPL=eloss (即 e 的 loss 次方)
- **你的 PPL**: e1.6646≈(2.71828)1.6646≈5.28
- **PPL 的通俗解释**: 困惑度可以粗略理解为“模型在预测下一个词时，平均在多少个词之间感到困惑和不确定”。
- **结果解读**: 你的模型 PPL 约等于 5.28，意味着它在预测下一个词时，能把范围从 **15万多个**选项中，精准地缩小到平均只有 **5-6 个**备选项。这足以证明你的模型学得非常好，所以 **1.66 的 Loss 一点也不高**。

------

### 5. 如何评判 Loss 到了最优值？

这是一个更重要的问题。评判“最优”不能只依靠单一的训练损失，而需要一套组合拳。以下是评判是否达到最优的几种核心方法，按重要性排序：

#### 方法一：引入“验证集损失 (Evaluation Loss)”—— 最核心的指标 🏆

这是**最科学、最标准**的方法。

- 概念:
  - **训练集 (Training Set)**: 用来训练模型的数据，好比是给学生做的“练习册”。`train_loss` 就是学生在练习册上的得分。
  - **验证集 (Validation/Evaluation Set)**: 从原始数据中**留出**一小部分（比如10%-20%），这部分数据**不参与训练**，专门用来在训练过程中“考核”模型，好比是“模拟考试”。`eval_loss` 就是学生在模拟考上的得分。
- 评判标准:
  - 我们追求的是模型在**没见过的数据**上的表现，所以 **`eval_loss` 比 `train_loss` 更重要**。
  - **最优状态**：`train_loss` 在稳步下降，同时 `eval_loss` 也跟着下降。当 `eval_loss` **下降到最低点并开始有回升迹象**时，这个最低点对应的模型就是“最优模型”。
- 图形解读:
  - **A点 (欠拟合)**: 还没学好，两个Loss都还有下降空间。
  - **B点 (最佳点)**: `eval_loss` 达到最低。这是模型的泛化能力最强的时刻，应该保存这个版本的模型作为最佳选择。
  - **C点 (过拟合)**: 模型在练习册上分数越来越高（`train_loss` 持续下降），但在模拟考上分数开始变差（`eval_loss` 回升）。这说明模型开始“死记硬背”训练数据，失去了泛化能力。
- 在 LLaMA Factory 中如何做:
  1. 准备两个数据集文件，一个 `train.json`，一个 `eval.json`。
  2. 在“数据”设置中，分别指定“数据集”和“验证集大小/验证集文件”。
  3. 在“训练参数”中，将“评估策略”设置为 `steps`，并设置一个合理的“评估步数”（比如每10步或50步评估一次）。

#### 方法二：观察训练损失的“形状”和趋势

如果你没有验证集，只能看 `train_loss`，那么你需要关注它的变化趋势。

- **理想趋势**: 像你的日志里那样，Loss 在训练初期快速下降，然后下降速度逐渐减缓，最后趋于平稳（收敛）。
- **何时停止**: 当 Loss 曲线变得非常平坦，连续很多步都不再有明显下降时，就可以认为训练已经饱和，可以停止了。继续训练不仅效果提升有限，还可能导致过拟合。

#### 方法三：进行“人工评测”—— 最直观的方法

数字终究是数字，模型好不好用，最终还是要人来判断。

- **操作**: 在训练的不同阶段（比如 Loss 下降到不同水平时）都保存一个模型副本。训练结束后，分别加载这些模型，在“聊天”界面用各种问题去测试它们。
- 评判标准:
  - 它能理解你的指令吗？
  - 它的回答风格符合你的预期吗？
  - 它是否还在“胡说八道”（幻觉）？
  - 它的回答质量是否比微调前有实质性提升？

有时候，`eval_loss` 最低的版本不一定在主观感受上是最好的。所以，人工评测是最终的、也是最重要的“验金石”。

## 🎉 总结

通过本教程，我们成功实现了：

✅ **高效数据生成**：使用Easy Dataset创建高质量财报QA数据集  
✅ **DCU优化训练**：充分发挥k100-AI硬件优势进行模型微调  
✅ **效果显著提升**：微调后模型在财经领域表现优异  
✅ **生产就绪**：提供完整的部署和监控方案  

**🚀 下一步探索**：

- 尝试更大规模模型（32B/70B）
- 探索多卡分布式训练

---

## 📚 相关资源

1. [Easy Dataset](https://github.com/ConardLi/easy-dataset) 
2. [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)
3. [海光DCU开发社区](https://developer.sourcefind.cn/)