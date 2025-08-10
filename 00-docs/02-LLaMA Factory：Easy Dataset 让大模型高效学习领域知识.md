# Easy Dataset × LLaMA Factory: 让大模型在海光DCU k100-AI上高效学习领域知识

[Easy Dataset](https://github.com/ConardLi/easy-dataset) 是一个专为创建大型语言模型（LLM）微调数据集而设计的应用程序。它提供了直观的界面，用于上传特定领域的文件，智能分割内容，生成问题，并为模型微调生成高质量的训练数据。支持使用 OpenAI、DeepSeek、火山引擎等大模型 API 和 Ollama 本地模型调用。

[LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) 是一款开源低代码大模型微调框架，集成了业界最广泛使用的微调技术，支持通过 Web UI 界面零代码微调大模型，目前已经成为开源社区最热门的微调框架之一，GitHub 星标超过 4.6 万。支持全量微调、LoRA 微调、以及 SFT 和 DPO 等微调算法。

**🚀 本教程专门针对海光DCU k100-AI加速卡优化**，使用 Easy Dataset 从五家互联网公司的公开财报构建 SFT 微调数据，并使用 LLaMA Factory 微调 DeepSeek-R1-Distill-Qwen-32B 模型，充分发挥DCU硬件的计算优势，使微调后的模型能学习到财报数据集中的知识。

## 📋 DCU k100-AI 运行环境要求

### 🔧 硬件配置
- **DCU型号**：海光 k100-AI 加速卡
- **显存要求**：≥ 450 GB HBM2E（k100-AI标配64GB）
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
- **LLaMA Factory版本**：v0.9.2

### ⚡ DCU性能特性
**k100-AI具有以下优势：**

- **高带宽内存**：HBM2E提供更高的内存带宽
- **超大容量显存**：64GB显存支持超大模型训练和微调
- **优化架构**：专为AI工作负载设计的计算单元
- **生态兼容**：兼容PyTorch、PaddlePaddle等主流框架

## 🛠️ 使用 Easy Dataset 生成微调数据

### 📦 安装 Easy Dataset

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
# 前台启动
pnpm start
# 后台启动
nohup pnpm start > output.log 2>&1 &
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

# 或者直接使用
01-llm-fine-tuning/dataset/FinancialData-SecondQuarter-2024

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
   - **模型名称**：[qwen-max模型](https://bailian.console.aliyun.com/?spm=a2c4g.11186623.0.0.375c657b3BTO7w&tab=model#/model-market/detail/qwen-max?modelGroup=qwen-max)

     ![qwen-max模型](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202508090855415.png)
   - **API密钥**：[阿里云百炼API KEY](https://bailian.console.aliyun.com/?tab=model#/api-key)
   - **请求配置**：

     ```json
     {
       "temperature": 0.7,
       "max_tokens": 8192
     }
     ```

   > 这里以 DeepSeek 模型为例，修改模型**提供商**和**模型名称**，填写 **API 密钥**，点击**保存**后将数据保存到本地，在右上角选择配置好的模型

   ![image-20250809100618667](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202508091006956.png)

   - **提供商**：阿里云百炼（推荐，性价比高）

   - **模型名称**：[deepseek-chat](https://api-docs.deepseek.com/zh-cn/)

   - **API密钥**：[deepseek API KEY](https://platform.deepseek.com/api_keys)

   - **请求配置**：

     ```json
     {
       "temperature": 0.7,
       "max_tokens": 8192
     }
     ```

3. **任务配置**

   > 打开**任务配置**页面，设置

   ![任务配置](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202506061821826.png)

   ```bash
   文本分割设置：
   - 最小长度：800字符
   - 最大长度：1500字符（保证上下文完整性）
   
   问题生成设置：
   - 生成密度：每240个字符1个问题（平衡质量与数量）
   ```

#### 第二步：批量处理数据文件

1. **并行上传文件**

   > 1. 打开**文献处理**页面，**点击右上角选择刚配置的模型**
   > 2. 选择并上传示例数据文件，选择文件后点击**上传并处理文件**
   > 3. 上传后会调用大模型解析文件内容并分块，耐心等待文件处理完成，示例数据通常需要 2 分钟左右

   ```bash
   同时上传多个文件：
   - 选择所有财报文件（txt + markdown格式）
   ```

   ![并行上传文件](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202508091408431.png)

2. **生成微调数据**

   > 1. 待文件处理结束后，可以看到文本分割后的文本段，选择全部文本段，点击**批量生成问题**
   > 2. 点击后会调用大模型根据文本块来构建问题，耐心等待处理完成。视 API 速度，处理时间可能在 20-40 分钟不等
   > 3. 处理完成后，打开**问题管理**页面，选择全部问题，点击**批量构造数据集**，耐心等待数据生成。视 API 速度，处理时间可能在 20-40 分钟不等

   ![批量生成问题](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202508091435357.png)

   ![批量构造数据集](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202508091437010.png)

#### 第四步：导出LLaMA Factory格式的数据集

> 1. 答案全部生成结束后，打开**数据集管理**页面，点击**导出数据集**
> 2. 在导出配置中选择**在** **LLaMA Factory** **中使用**，点击**生成 LLaMA Factory** **配置**，即可在对应文件夹下生成配置文件，点击**复制**按钮可以将配置路径路径复制到粘贴板。
> 3. 在配置文件路径对应的文件夹中可以看到生成的数据文件，其中主要关注以下三个文件
>    1. dataset_info.json：LLaMA Factory 所需的数据集配置文件
>    2. alpaca.json：以 Alpaca 格式组织的数据集文件
>    3. sharegpt.json：以 Sharegpt 格式组织的数据集文件
>
> 其中 alpaca 和 sharegpt 格式均可以用来微调，两个文件内容相同。

![导出数据集](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202508091440669.png)

![在LLaMA Factory中使用](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202508091440143.png)

![数据集路径](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202508091443836.png)

**文件结构验证**

```bash
ls -la exported/
# dataset_info.json  - LLaMA Factory配置
# alpaca.json       - Alpaca格式数据
# sharegpt.json     - ShareGPT格式数据
# 其中 alpaca 和 sharegpt 格式均可以用来微调，两个文件内容相同。
```

服务日志输出

```bash
rebuild tags [
  { label: '1 医学基础', child: [ [Object], [Object], [Object] ] },
  {
    label: '2 临床医学',
    child: [ [Object], [Object], [Object], [Object] ]
  },
  { label: '3 诊断技术', child: [ [Object], [Object], [Object] ] },
  { label: '4 治疗手段', child: [ [Object], [Object], [Object] ] },
  { label: '5 公共卫生', child: [ [Object], [Object], [Object] ] },
  { label: '6 医学伦理' },
  { label: '7 医疗管理' }
]
检查到 0 个激活的GA pairs，不启用GA扩展生成
标准模式：生成 2 个问题
检查到 0 个激活的GA pairs，不启用GA扩展生成
标准模式：生成 2 个问题
检查到 0 个激活的GA pairs，不使用增强提示词
使用标准提示词生成答案
检查到 0 个激活的GA pairs，不使用增强提示词
使用标准提示词生成答案
检查到 0 个激活的GA pairs，不使用增强提示词
使用标准提示词生成答案
检查到 0 个激活的GA pairs，不使用增强提示词
使用标准提示词生成答案
检查到 0 个激活的GA pairs，不使用增强提示词
使用标准提示词生成答案
成功生成标准数据集: 初产妇第二产程延长后出现阴道持续鲜红色出血最可能的原因是什么
检查到 0 个激活的GA pairs，不使用增强提示词
使用标准提示词生成答案
成功生成标准数据集: 老年男性在输液过程中突发急性肺水肿的首选治疗方案是什么
检查到 0 个激活的GA pairs，不使用增强提示词
使用标准提示词生成答案
成功生成标准数据集: 1岁儿童夏季头皮出现多发结节、溃破流脓并伴有头皮下空洞的中医诊断名称是什么
成功生成标准数据集: 18岁女性右侧血性胸腔积液且ADA显著升高，最可能的诊断是什么？
成功生成标准数据集: 37岁女性闭经2月且孕激素试验阴性，首先应考虑什么诊断？
成功生成标准数据集: 60岁男性肺结核患者出现右侧胸腔积液时，为明确胸水性质应首选哪种检测指标
成功生成标准数据集: 51岁女性围绝经期样症状伴阴道持续出血，临床首先应进行什么检查
```



## 🚀 使用 LLaMA Factory 在DCU k100-AI上微调模型

### 📦 安装DCU优化的LLaMA Factory

#### 环境准备

```bash
# 1. 创建DCU专用虚拟环境
conda create -n llm-finetune-dtk2504 python=3.10 -y
conda activate llm-finetune-dtk2504
# 环境删除
## conda env remove -n llm-finetune-dtk2504

# 2. 安装DCU版本PyTorch等深度学习框架依赖
git clone -b finetuning0808 https://github.com/FlyAIBox/dcu-in-action.git
cd dcu-in-action
pip install -r requirements-dtk2504.txt

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

### 输出信息
PyTorch版本: 2.4.1
DCU可用: True
DCU设备数: 8
当前DCU: K100_AI
显存容量: 64.0 GB
```

#### 安装LLaMA Factory

```bash
# 1. 克隆仓库（使用国内镜像）
git clone -b v0.9.2 https://gitee.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 2. 安装依赖
pip install -e ".[torch,metrics]" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 验证安装
llamafactory-cli version
```

**✅ 安装成功输出**：
```bash
(llm-finetune-dtk2504) root@Ubuntu2204:~/AI-BOX/code/dcu/LLaMA-Factory# llamafactory-cli version
[2025-08-09 16:14:47,923] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/root/anaconda3/envs/llm-finetune-dtk2504/lib/python3.10/site-packages/jieba/_compat.py:18: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
----------------------------------------------------------
| Welcome to LLaMA Factory, version 0.9.2                |
|                                                        |
| Project page: https://github.com/hiyouga/LLaMA-Factory |
----------------------------------------------------------
```

### 📈 AI模型显存需求精准计算器

- Github: https://github.com/st-lzh/vram-wuhrai
- 官网：https://vram.wuhrai.com/

**🧠 核心功能**

- 🎯 **精准显存计算** - 基于最新AI工程实践的精确计算公式
- 🔧 **三种工作模式** - 支持训练、推理、微调场景的显存需求分析
- ⚡ **优化技术支持** - 涵盖混合精度、梯度检查点、量化等先进优化技术
- 🤖 **热门模型库** - 内置Qwen、DeepSeek、Llama、ChatGLM等主流模型参数

**✨ 智能特性**

- 📊 **GPU自动匹配** - 基于显存需求智能推荐最适合的GPU配置
- 💰 **价格对比分析** - 提供GPU价格对比和性价比评估
- 📈 **利用率分析** - 计算GPU资源利用率，优化硬件投资
- 🔄 **实时更新** - 模型参数规格持续更新，确保计算准确性

**🎯 适用场景**

- AI研究人员进行模型训练前的硬件规划
- 企业级AI部署的GPU选型决策
- 个人开发者的显卡升级参考
- 教育机构的AI实验室硬件配置

![AI模型显存需求精准计算器](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202508091541457.png)



*专业的训练显存计算界面，支持模型参数、批次大小、序列长度、精度等配置，实时显示显存需求和GPU推荐*

### 🎯 启动LLaMA Factory的微调任务

#### 启动LLaMA Board

```bash
# DCU环境变量设置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # 指定使用8块DCU
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # 指定使用8块DCU
# export USE_MODELSCOPE_HUB=1   # 使用国内模型源

# 启动Web UI
llamafactory-cli webui

# 后台运行
nohup llamafactory-cli webui > webui.log 2>&1 &
tail -f webui.log
# 结束服务
pkill -f "llamafactory-cli webui"
```

**🌐 访问地址**：http://localhost:7860

#### 微调参数配置

1. **模型选择与优化**

   > 进入 Web UI 界面后，选择模型为 `DeepSeek-R1-Distill-Qwen-32B`，模型路径可填写本地绝对路径，不填则从互联网下载

   ```bash
   模型：DeepSeek-R1-Distill-Qwen-32B
   模型路径：/path/to/models/DeepSeek-R1-Distill-Qwen-32B
   ```

2. **数据配置**

   > 将**数据路径**改为使用 Easy Dataset 导出的配置路径，选择 Alpaca 格式数据集

    **Easy Daset数据集配置**

   ```bash
   数据路径：/path/to/exported（如/root/AI-BOX/code/dcu/easy-dataset/local-db/H_LPE1u7tKkP）
   数据集：选择生成的财报数据集
   数据格式：alpaca
   
   预处理优化：
   - Max Length：2048（充分利用DCU显存）
   - Cutoff Length：1024
   - Preprocessing：8进程并行
   ```

   **在线数据集**

   1. 下载微调数据集后解压到本地，将`jsonl`文件拖拽到`LLaMA Factory/data`文件下或其他位置。

   2. 修改`LLaMA Factory/data`文件夹下`dataset_json.info`文件，在json文件开始位置添加如下所示脚本。

      > **注意：**修改`file_name`路径，为你的数据集的绝对路径

   ```shell
     "medical_sft": {
       "file_name": "/root/AI-BOX/code/dcu/LLaMA-Factory/data/medical_o1_sft_Chinese_alpaca_cot.jsonl",
       "columns": {
         "prompt": "prompt",
         "response": "response"
       }
     },
   ```

   ![添加数据集](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202508091616645.png)

3. **训练参数**

   > 为了让模型更好地学习数据知识，将**学习率**改为 5e-5，**训练轮数**提高到 3 轮。
   >
   > 批处理大小和梯度累计则根据设备显存大小调整，在显存允许的情况下提高批处理大小有助于加速训练，**一般保持批处理大小×梯度累积×显卡数量等于 32 即可**

   以下是本次训练的核心超参数配置，这些参数共同决定了模型的最终表现。

   | 参数分类       | 参数项                        | 页面显示名称    | 设定值 | 目的与考量                                                   |
   | -------------- | ----------------------------- | --------------- | ------ | ------------------------------------------------------------ |
   | **数据与模型** | `cutoff_len`                  | 截断长度        | 2048   | 定义输入序列的最大长度。太小会丢失信息，太大则增加计算成本和显存消耗。2048 是一个常见的平衡选择。 |
   | **训练策略**   | `num_train_epochs`            | 训练轮数        | 3.0    | 模型在整个数据集上的迭代次数。太少会导致欠拟合，太多会导致过拟合。需要根据实验来调整。 |
   |                | `learning_rate`               | 学习率          | 5e-05  | 参数更新的步长。太高会导致震荡无法收敛，太低则训练缓慢。5e-05 是一个常用且安全的初始值。 |
   |                | `lr_scheduler_type`           | 学习率调节器    | cosine | 动态调整学习率的策略。余弦退火可以在训练初期快速收敛，后期精细微调，提高泛化能力。 |
   | **批处理**     | `per_device_train_batch_size` | 批处理大小      | 1      | 单个设备每次处理的样本数。此处设为 1，通常是为了节省显存，以支持大模型或长序列训练。 |
   |                | `gradient_accumulation_steps` | 梯度累积        | 32     | 累积梯度 32 次后才进行一次参数更新。这可以在显存有限时，通过模拟更大的批次大小来获得更稳定的训练效果。 |
   | **LoRA**       | `lora_rank`                   | LoRA 秩         | 16     | 决定 LoRA 模块的参数量和表达能力。秩越高，效果可能越好，但计算成本也越高。16 是一个实践中常用的值。 |
   |                | `lora_alpha`                  | LoRA 缩放系数   | 32     | 用于调整 LoRA 模块对原始模型的影响程度。通常与 `lora_rank` 成正比，以确保 LoRA 在微调中起足够作用。 |
   | **DeepSpeed**  | `deepspeed `                  | DeepSpeed stage | 3      | 指的是 DeepSpeed ZeRO-3 显存优化等级。通过跨设备分片模型状态，可以在显存有限的情况下训练更大的模型。 |

#### 1. 核心选择（要做什么？）

![image-20250809170426740](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202508091704908.png)

这部分决定了我们选择哪个“大学生”以及用什么“培训方法”。

- 模型名称 (Model Name)
  - **`DeepSeek-R1-Distill-Qwen-32B`**: 这是你选择的**基础模型**。
- 微调方法 (Finetuning Method)
  - `lora`: 这是目前最流行的一种**参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）**方法。
    - **小白理解**：传统的“全量微调”像是给大学生“重塑大脑”，计算量巨大，需要非常好的显卡。而 `LoRA` 就像是给了大学生一个“**特殊技能笔记本**” 📝。我们不改变他原来的大脑，只让他在这个小本本上记录和学习新技能。这个“笔记本”很小，所以训练起来非常快，对显卡要求也低很多。训练完成后，把这个“笔记本”和他原本的大脑结合，他就掌握了新技能。这是新手入门微调的**首选方法**。

------

#### 2. 训练数据（学什么？）

这部分定义了模型的“学习材料”。

- 数据路径 (Dataset Path)
  - `/root/AI-BOX/code/dcu/easy-dataset/local-db/H_LPE1u7tKkP` 和 `[Easy Dataset][H_LPE1u7tKkP]Alpaca` 
  - 或`medical_sft` 这里指定了你的数据集，也就是给模型学习的“教材”。
  - **关键点**：**数据质量决定了微调效果的上限**。好的数据能让模型学到真正的能力，而差的数据只会让模型“学坏”。

------

#### 3. 核心训练参数（怎么学？）

![image-20250809170455248](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202508091704442.png)

这部分是最关键的超参数，它们共同控制着整个“学习过程”。

- **学习率 (Learning Rate)**
  - **含义**: 模型在学习中每一步“更新知识”的幅度，可以理解为“**学习的步子大小**” 🏃。
  - **比喻**: 步子太大（学习率过高），容易“扯着蛋”，学不扎实，甚至可能越学越差（知识跑偏）。步子太小（学习率过低），学习速度太慢，要花很长时间才能学会。`5e-5` (即 0.0001) 是 LoRA 微调中一个比较常用的初始值。
- **训练轮数 (Epochs)**
  - **含义**: 把整个“教材”（数据集）从头到尾完整学习的次数。
  - **比喻**: `Epochs = 3.0` 就意味着让学生把整本教材读3遍。对于大型数据集，通常 1-3 个 Epoch 就足够了，太多了模型可能会“死记硬背”，失去泛化能力。
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

![](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202508091705927.png)

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

#### SwanLab监控

[SwanLab](https://swanlab.cn/)是一款开源且轻量级的AI模型训练可视化追踪工具。在本次微调任务中，我们将使用SwanLab记录整个微调过程。在开始之前，请确保您已经登录[SwanLab](https://swanlab.cn/)平台。登录后，点击`设置/常规`，即可获取`API Key`

![image-20250809172722147](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202508091727413.png)

```bash
#  实时监控DCU使用情况
watch -n 0.1 "rocm-smi"

# SwanLab
pip install swanlab==0.6.4
swanlab login
```

点击**开始**按钮，等待模型下载，一段时间后应能观察到训练过程的损失曲线

![观察到训练过程的损失曲线](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202508091828879.png)

------

#### 1. 模型加载与硬件环境配置的深层解读

这个阶段的日志是训练的起点，它揭示了模型本身的规模和如何通过技术手段适配硬件。

- 模型规模和架构:

  `[INFO|2025-08-09 17:37:10] ... Model config Qwen2Config ... "hidden_size": 5120, "num_hidden_layers": 64, "num_attention_heads": 40`

  - 这行日志不仅告诉我们模型是 **Qwen2** 架构，更明确了其核心参数：
    - `num_hidden_layers: 64` 意味着模型有64个Transformer层。
    - `hidden_size: 5120` 表示每个token的隐藏层维度是5120。
    - `num_attention_heads: 40` 说明在每一层中，注意力机制被分割成40个独立的“头”来并行计算。
  - 这些参数共同决定了模型320亿的庞大参数量。

- 内存优化与计算精度:

  `"torch_dtype": "bfloat16"`

  - 这是一个关键的内存和性能配置。`bfloat16`（Brain Floating Point）是一种半精度浮点数格式，相比于常见的`fp16`，它在数值范围上更广，在训练中能更好地避免梯度下溢或上溢的问题，同时将模型和梯度所需的显存减半，使320亿参数模型在有限显存的DCU上成为可能。

- 分布式训练策略:

  `[INFO|2025-08-09 17:37:24] ... Detected DeepSpeed ZeRO-3: activating zero.init() for this model`

  - 这行日志是整个训练能够启动的决定性因素。**DeepSpeed ZeRO-3** 是一种分片数据并行技术。对于一个320亿参数的模型，其**参数、梯度和优化器**状态的总大小远超单张DCU的显存。ZeRO-3的作用是：
    - 将 **模型参数** (Parameters)、**梯度** (Gradients) 和 **优化器状态** (Optimizer States) 分片，并分别存储在8张DCU上。
    - 在需要时（例如前向或反向传播），它会自动在所有DCU之间通信，只将需要的参数部分加载到当前计算的DCU上，从而实现了“用多卡的显存共同存储一个大模型”的目标。这正是320亿参数模型能被训练的根本原因。

------

#### 2. 微调技术与效率指标的量化分析

这一部分日志用精确的数字展示了LoRA微调的效率。

- LoRA的参数效率:

  `[INFO|2025-08-09 17:39:56] ... trainable params: 134,217,728 || all params: 32,898,094,080 || trainable%: 0.4080`

  - 日志中的`trainable params`（可训练参数）和`all params`（总参数）对比，直观地量化了LoRA的优势。
  - `32,898,094,080` 是整个模型的总参数量。
  - `134,217,728` 是实际参与微调更新的LoRA矩阵参数量。
  - `trainable%: 0.4080` 表明我们只更新了 **不到0.5%** 的参数。这极大地减少了反向传播的计算量和所需的显存，使得训练速度更快，同时也能有效防止灾难性遗忘。

------

#### 3. 训练过程与性能指标的动态观察

此阶段日志记录了训练过程中的实时数据，提供了关于训练稳定性和效率的证据。

- 训练吞吐量（Throughput）:

  `[INFO|2025-08-09 17:45:09] ... 'throughput': 603.50`

  - `throughput` 是衡量训练效率的关键指标，单位是 **tokens/s**。这表示每秒钟训练过程处理了603.5个token。在整个训练过程中，这个值稳定在600左右，这说明DCU硬件和软件栈（DeepSpeed等）结合工作稳定，训练性能没有出现剧烈波动。

- **损失值（Loss）变化**:

  - `[INFO|2025-08-09 17:45:09] ... 'loss': 0.7071` (训练初期)
  - `[INFO|2025-08-09 18:08:36] ... 'loss': 0.6772` (训练后期)
  - 损失值从训练初期的 `0.7071` 逐步下降到后期的 `0.6772`，这表明模型在学习训练集中的数据，参数在不断优化。损失值的持续下降是训练有效的直接证明。

- 最终评估结果:

  `train_loss: 0.6971`

  `eval_loss: 0.6747`

  - **`eval_loss` 小于 `train_loss`** 是一个非常好的现象，这意味着模型在未见过的验证集上的表现甚至略好于训练集。这强有力地证明了模型没有过拟合，具备良好的泛化能力。

- 技术警告解析:

  [WARNING|2025-08-09 17:37:24] ... Using the SDPA attention implementation on multi-gpu setup with ROCM may lead to performance issues ...

  - 这条警告表明在PyTorch的特定版本中，针对SDPA（Scaled Dot-Product Attention）的优化实现可能与ROCM（海光DCU的GPU驱动和库）存在兼容性或性能问题。因此，代码自动禁用了该优化。这提示我们，在使用特定硬件（如DCU）时，需要关注框架和驱动的兼容性，可能需要通过参数调整来绕过潜在的性能瓶颈。

#### 4. 关键技术参数计算与深层分析

我们将结合日志中的具体数字，计算并解释一些核心参数，以更精确地理解训练过程。

- **1. 总的训练批次大小 (Total Training Batch Size)**
  - **日志参数**:
    - `train_micro_batch_size_per_gpu = 1` (每个 GPU 的微批次大小)
    - `gradient_accumulation_steps = 32` (梯度累积步数)
    - `world_size = 8` (分布式训练中的 GPU 数量)
    - `[INFO|trainer.py:2249] ... Total train batch size ... = 256`
  - **计算过程**:
    - 总批次大小 = `train_micro_batch_size_per_gpu` × `world_size` × `gradient_accumulation_steps`
    - 总批次大小 = `1` × `8` × `32` = **256**
  - **深层分析**:
    - 这意味着在每次模型参数更新（即一个优化步）之前，训练框架会处理 **256个样本**。
    - 这种设置在大模型微调中非常常见，因为单个大样本（长文本）可能无法完全放入单卡显存中，即使是使用 ZeRO-3 这样的技术。通过将微批次设置为1，然后通过梯度累积来模拟更大的批次大小（256），可以有效利用硬件资源，同时保证训练的稳定性。
- **2. 总的优化步数 (Total Optimization Steps)**
  - **日志参数**:
    - `Num examples = 900` (训练集总样本数)
    - `Num Epochs = 2` (总的训练周期数)
    - `Total train batch size = 256` (总批次大小)
    - `[INFO|trainer.py:2251] ... Total optimization steps = 6`
  - **计算过程**:
    - 总优化步数 = `Num examples × Num Epochs / Total train batch size`
    - 总优化步数 = `900` × `2` / `256`  ≈ **7**
  - **深层分析**:
    - 日志显示的最终步数是6，与理论计算的7稍有出入。这可能是因为训练集大小不能被总批次大小整除，导致最后一个 epoch 的步数略少。但这个数字基本吻合，表明训练过程严格按照预定配置执行。在整个训练过程中，模型总共只进行了6次参数更新，这再次凸显了 LoRA 微调在小数据集上的快速收敛能力。
- **3. 训练总 Token 数 (Total Input Tokens Seen)**
  - **日志参数**:
    - `[INFO|2025-08-09 18:08:36] ... 'epoch': 1.7, 'num_input_tokens_seen': 1093008`
  - **深层分析**:
    - `num_input_tokens_seen` 表示模型在整个训练过程中见过的总 token 数量。这个指标比样本数更能反映模型的学习量。
    - 在训练结束时，总共处理了`1,093,008`个 token，这对于一个320亿参数的模型来说，学习量相对有限，但 LoRA 微调的优势在于能够通过少量数据和参数高效地迁移知识，因此仍然可以获得良好的效果。
- **4. 训练时长与效率 (Training Runtime and Efficiency)**
  - **日志参数**:
    - `train_runtime = 0:29:18.52`
    - `train_steps_per_second = 0.003`
    - `train_samples_per_second = 1.024`
    - `'throughput': 603.50`
  - **深层分析**:
    - **训练时长** (29分18秒) 针对6个优化步和109万个 token 而言，相对较长。
    - **`train_steps_per_second`** (每秒0.003步) 和 **`train_samples_per_second`** (每秒1.024样本) 看起来很低。这是因为计算的是整个训练周期内的平均值，而大部分时间都花在了单个梯度累积批次（256个样本）的前向和反向传播上。
    - 真正反映瞬时性能的是 **`throughput`** (吞吐量)，其值稳定在 **600 tokens/s** 左右。这是衡量硬件计算效率的关键参数。它说明在实际的计算过程中，DCU的性能是稳定的。

![](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202508091844569.png)

通过这些具体参数的计算和分析，我们能够更全面地理解这次微调：它不仅成功地使用了 **DeepSpeed ZeRO-3** 和 **LoRA** 两种关键技术在海光DCU上运行了大型模型，而且在配置上通过梯度累积来平衡硬件限制，最终在可控的时间内完成了训练，并获得了优秀的泛化效果。

#### 5. SwanLab监测效果

![SwanLab监测效果](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202508091847659.png)

##### 1. `train/loss` (训练损失)

- **图表含义**: 这条曲线描绘了在每个优化步（Step）上，模型在训练集上的损失值。
- **图表趋势**: 损失值从大约0.73开始，在前两步略有波动，然后开始持续下降，在第6步结束时降至0.68左右。
- **分析**: 损失持续下降是训练成功的核心指标。它表明模型正在有效地学习数据集中的模式和知识，其预测结果越来越接近真实标签。这个下降趋势证明了LoRA微调过程是稳定且有效的。

------

##### 2. `train/grad_norm` (梯度范数)

- **图表含义**: 这条曲线显示了在每次参数更新之前，所有梯度的范数（magnitude），可以理解为“梯度的大小”。
- **图表趋势**: 梯度范数从第一步的0.117左右稳步上升，在第四步之后达到一个平台期，稳定在0.17左右。
- **分析**: 在使用梯度累积和学习率衰减的训练中，梯度范数逐渐增大是正常的。这可能意味着：
  - **模型在探索更复杂的参数空间**：随着训练的深入，模型可能需要更大的梯度来调整参数。
  - **训练是稳定的**：梯度范数没有出现失控的暴涨（即梯度爆炸），也没有迅速归零（即梯度消失），这表明训练过程是健康的。

##### 3. `train/learning_rate` (学习率)

- **图表含义**: 这条曲线展示了在每个优化步中，优化器使用的学习率。
- **图表趋势**: 学习率从约5e-5的高值开始，随着步数的增加，平滑地、非线性地下降，在第6步结束时降至接近0。
- **分析**: 这是一个典型的**学习率衰减**策略，很可能采用了**余弦退火（Cosine Decay）**或其他类似方法。
  - **初期高学习率**：让模型快速探索参数空间，进行较大步长的更新。
  - **后期低学习率**：让模型在训练末期进行精细调整，防止在最优解附近震荡，有助于模型更精确地收敛。这个图表是训练配置合理的有力证明。

##### 4. `train/epoch` (训练周期)

- **图表含义**: 这条曲线显示了在每个优化步完成时，当前所处的训练周期进度。
- **图表趋势**: 随着步数从1增加到6，周期进度从0.28线性增加到1.7左右。
- **分析**: 这个图表与我们之前计算的总步数和总周期数相吻合。它直观地展示了训练进度，验证了每个优化步确实对应着训练数据的一个特定子集。

##### 5. `train/num_input_tokens_seen` (已处理的输入token数)

- **图表含义**: 这条曲线显示了从训练开始到每个优化步结束时，模型总共处理的token数量。
- **图表趋势**: 该曲线呈一条完美的斜直线，从第一步开始平稳上升，到**第6步达到约1.1百万**。
- **分析**: 这条直线是训练过程稳定性的一个重要“健康检查”。它表明每个优化步处理的token数量是恒定的，这符合我们之前分析的“总批次大小”和“梯度累积”的配置。

##### 6. `train/global_step` (全局步数)

- **图表含义**: 这是从训练开始至今，优化步数的累计计数。
- **图表趋势**: 曲线从1到6呈一条完美的直线。
- **分析**: 这是最基础的计数器，用于确认训练确实按照预定的6个优化步完成了。它为所有其他图表提供了X轴的基准，确保了我们对训练流程的理解是正确的。

所有曲线都表现出健康、稳定的趋势，没有出现损失暴涨、梯度爆炸或训练停滞等异常情况。这套图表是本次 LoRA 微调任务成功的有力佐证。

## 🧪 验证微调效果

#### 模型加载与测试

> 1. 选择**检查点路径**为刚才的输出目录，打开 **Chat** 页面，点击**加载模型**
> 2. 在下方的对话框中输入问题后，点击提交与模型进行对话，经与原始数据比对发现微调后的模型回答正确

![image-20250809193615409](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202508091936664.png)

```bash
测试问题示例：
Q: "一名40岁男性突发上腹剧痛伴恶心半小时，查体显示上腹部深压痛，但Murphy征阴性，血常规显示白细胞计数略增，尿便常规正常，考虑其最可能的诊断是什么？ 请逐步推理并给出答案。"


微调后模型回答：

一名40岁男性突发上腹剧痛伴恶心半小时，查体显示上腹部深压痛，但Murphy征阴性，血常规显示白细胞计数略增，尿便常规正常。综合分析，最可能的诊断是：

急性胰腺炎

解释：

症状和体征：上腹剧痛、恶心，符合急性胰腺炎的表现。Murphy征阴性并不排除胰腺炎，因为该征阳性通常与胆囊炎相关。
实验室检查：白细胞计数略增高，提示可能存在炎症反应，但胰腺炎的确诊通常依赖于淀粉酶和脂肪酶的升高，这些在病例中未提及。
鉴别诊断：排除了胆囊炎（Murphy征阴性）、胃十二指肠穿孔（无明显腹膜刺激征）、阑尾炎（无转移性右下腹痛）等。
进一步检查：建议进行血液淀粉酶和脂肪酶检测，以及腹部超声或CT扫描以确认诊断。
因此，综合考虑，急性胰腺炎是最可能的诊断，需进一步检查确认。
```

点击**卸载模型**将微调后的模型卸载，清空**检查点路径**，点击**加载模型**加载微调前的原始模型

![image-20250809190117929](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202508091901133.png)

```bash
测试问题示例：
Q: "一名40岁男性突发上腹剧痛伴恶心半小时，查体显示上腹部深压痛，但Murphy征阴性，血常规显示白细胞计数略增，尿便常规正常，考虑其最可能的诊断是什么？ 请逐步推理并给出答案。"


原生模型回答：

根据患者的症状和检查结果，最可能的诊断是急性胃炎或十二指肠溃疡。这些情况通常表现为上腹痛，伴有恶心，且Murphy征阴性。白细胞略增可能提示轻度感染或炎症。为了进一步确认，建议进行胃镜检查或腹部超声等检查。

```

![image-20250809192651591](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202508091926977.png)

数据集相关内容

```bash
{"prompt": "一名40岁男性突发上腹剧痛伴恶心半小时，查体显示上腹部深压痛，但Murphy征阴性，血常规显示白细胞计数略增，尿便常规正常，考虑其最可能的诊断是什么？ 请逐步推理并给出答案。\n哦，这位40岁的男士上腹部突然很痛，还伴有恶心。嗯，这么剧烈的上腹痛，肯定是腹腔里哪里出了问题。\n\n先来看看这个Murphy征是阴性，说明胆囊炎可能性小。Murphy征不是用来检查胆囊的吗？嗯，胆囊方面可能不是问题。\n\n白细胞增多，这倒是有些炎症或感染的征兆，尿和便倒是很正常，嗯，看看哪个器官出问题会这样的。\n\n这个痛，要是急性胰腺炎呢？胰腺炎可是会导致很剧烈的疼痛，还会恶心呢。\n\n但等等，胰腺炎的痛常会往后背放射吧，这里没说有背痛啊，可能性就减小了。\n\n那会不会是个什么溃疡，比如胃溃疡或十二指肠溃疡，穿孔的话会极痛啊，但问题是他有没提到腹膜刺激征呢？\n\n说不定是胆道里面有事。胆结石的话这个Murphy征不太符合，那蛔虫呢。是的，胆道蛔虫病会剧烈的上腹痛，也会恶心，还可能有白细胞增高。\n\n对，胆道蛔虫病看着很像。这种情况，蛔虫在胆道周转的时候会很痛，但不见得有典型Murphy征，挺合适的。\n\n那就暂时这么看，可能是胆道蛔虫病，毕竟解释得挺通的，先定个结论再说。", 


"response": "<think>\n根据提供的信息，这位40岁的男性患者出现了突发性的上腹部剧痛，并伴有恶心。查体发现上腹部深部压痛，但Murphy征阴性。这表明胆囊炎的可能性较小。血常规显示白细胞计数略增，提示可能存在炎症或感染，但尿便常规是正常的。\n\n在这种情况下，我们需要分析上腹痛的常见原因。急性胰腺炎通常会导致严重的上腹痛和恶心，但常见的特征之一是疼痛会向背部放射，而这里没有提到背痛，这减少了胰腺炎的可能性。此外，尽管胃或十二指肠溃疡穿孔会导致剧烈的疼痛，但通常会伴有腹膜刺激征，如果存在的话，通常在体检中会被提到。\n\n因此，我们需要考虑其他可能导致这些症状的消化道或胆道疾病。胆道蛔虫病是一个合理的假设。该病可能导致剧烈的上腹痛和恶心，也可能导致白细胞增多，但不一定会有典型的Murphy征。病因是在蛔虫进入胆道或胆管时可能引发剧烈的疼痛。\n\n综上所述，结合患者的症状和检查结果，最可能的诊断是胆道蛔虫病。这一诊断可以解释患者的上腹部剧痛、恶心和白细胞增高，而Murphy征阴性并不排除这个诊断。建议进一步的影像学检查，如腹部超声或CT扫描，以确认诊断。\n</think>\n\n根据提供的信息，这位40岁的男性患者出现了突发性的上腹部剧痛，并伴有恶心。查体发现上腹部深部压痛，但Murphy征阴性。这表明胆囊炎的可能性较小。血常规显示白细胞计数略增，提示可能存在炎症或感染，但尿便常规是正常的。\n\n在这种情况下，我们需要分析上腹痛的常见原因。急性胰腺炎通常会导致严重的上腹痛和恶心，但常见的特征之一是疼痛会向背部放射，而这里没有提到背痛，这减少了胰腺炎的可能性。此外，尽管胃或十二指肠溃疡穿孔会导致剧烈的疼痛，但通常会伴有腹膜刺激征，如果存在的话，通常在体检中会被提到。\n\n因此，我们需要考虑其他可能导致这些症状的消化道或胆道疾病。胆道蛔虫病是一个合理的假设。该病可能导致剧烈的上腹痛和恶心，也可能导致白细胞增多，但不一定会有典型的Murphy征。病因是在蛔虫进入胆道或胆管时可能引发剧烈的疼痛。\n\n综上所述，结合患者的症状和检查结果，最可能的诊断是胆道蛔虫病。这一诊断可以解释患者的上腹部剧痛、恶心和白细胞增高，而Murphy征阴性并不排除这个诊断。建议进一步的影像学检查，如腹部超声或CT扫描，以确认诊断。"}

```

根据用户提供的测试问题及微调前后模型的回答，我对两种模型的表现进行如下解读：

##### 微调后模型回答

- **诊断思路清晰**：微调后的模型首先给出了明确的结论**“急性胰腺炎”**，然后分点解释了推理过程，包括症状和体征、实验室检查、鉴别诊断和进一步检查。这种结构化的回答方式非常符合临床诊断的逻辑。
- **专业性强**：
  - 它正确地指出**Murphy征阴性**可以排除胆囊炎，但不能排除胰腺炎，这显示了对该体征临床意义的准确理解。
  - 它还提到急性胰腺炎的确诊需要**淀粉酶和脂肪酶**的升高，这一点非常关键，体现了扎实的医学知识。
  - 在鉴别诊断中，它排除了胃十二指肠穿孔（**腹膜刺激征**）、阑尾炎（**转移性右下腹痛**）等，推理严谨。
  - 最后，它给出了符合临床实践的下一步建议：**血液淀粉酶和脂肪酶检测**以及**腹部超声或CT扫描**。

##### 原生模型回答

- **诊断结论不准确**：原生模型给出的诊断是“急性胃炎或十二指肠溃疡”，这个结论不够严谨。虽然这些疾病可以引起上腹痛，但“突发上腹**剧痛**”更倾向于一些急腹症，而急性胃炎通常是胀痛或灼痛，且疼痛程度不至于如此剧烈。
- **推理过程模糊**：原生模型的推理过于笼统，仅凭“上腹痛、恶心、Murphy征阴性”就得出结论，缺乏对更严重疾病的鉴别和排除。
- **缺乏专业深度**：原生模型没有提到关键的体征和实验室检查，比如**腹膜刺激征、淀粉酶和脂肪酶**等。它给出的进一步检查建议（胃镜或腹部超声）虽然正确，但在急诊情况下，对于疑似急性胰腺炎的患者，更重要的是先查血清淀粉酶和脂肪酶。

### 总结

微调后的模型在处理这个医学案例时，展现出了更**严谨的逻辑推理**、**更强的专业知识**和**更贴近临床实践**的诊断思路。它能够准确抓住关键信息，进行有效的鉴别诊断，并给出合理的下一步诊疗建议。相比之下，原生模型的回答显得过于宽泛，诊断结论也缺乏足够的依据，可能会导致漏诊或延误治疗。这表明微调过程在提升模型处理复杂医学问题时的准确性和专业性方面取得了显著成效。

## 🧪模型导出

微调效果满意并需要导出模型，您可以在导出界面通过指定 **模型**、 **适配器**、 **分块大小**、 **导出量化等级及校准数据集**、 **导出设备**、 **导出目录** 等参数后点击 `导出` 按钮导出模型。

![image-20250809193901341](https://cdn.jsdelivr.net/gh/Fly0905/note-picture@main/imag/202508091939619.png)

导出结果

```bash
(llm-finetune-dtk2504) root@Ubuntu2204:/data/model/deepseek32b-medical# ll -G
total 64003384
drwxr-xr-x 2 root       4096 Aug  9 19:43 ./
drwxr-xr-x 5 root       4096 Aug  9 19:38 ../
-rw-r--r-- 1 root        749 Aug  9 19:41 config.json
-rw-r--r-- 1 root        181 Aug  9 19:41 generation_config.json
-rw-r--r-- 1 root 4891730992 Aug  9 19:41 model-00001-of-00014.safetensors
-rw-r--r-- 1 root 4876059352 Aug  9 19:41 model-00002-of-00014.safetensors
-rw-r--r-- 1 root 4876059384 Aug  9 19:42 model-00003-of-00014.safetensors
-rw-r--r-- 1 root 4876059416 Aug  9 19:42 model-00004-of-00014.safetensors
-rw-r--r-- 1 root 4876059416 Aug  9 19:42 model-00005-of-00014.safetensors
-rw-r--r-- 1 root 4876059416 Aug  9 19:42 model-00006-of-00014.safetensors
-rw-r--r-- 1 root 4876059416 Aug  9 19:42 model-00007-of-00014.safetensors
-rw-r--r-- 1 root 4876059416 Aug  9 19:43 model-00008-of-00014.safetensors
-rw-r--r-- 1 root 4876059416 Aug  9 19:43 model-00009-of-00014.safetensors
-rw-r--r-- 1 root 4876059416 Aug  9 19:43 model-00010-of-00014.safetensors
-rw-r--r-- 1 root 4876059416 Aug  9 19:43 model-00011-of-00014.safetensors
-rw-r--r-- 1 root 4876059416 Aug  9 19:43 model-00012-of-00014.safetensors
-rw-r--r-- 1 root 4876059416 Aug  9 19:43 model-00013-of-00014.safetensors
-rw-r--r-- 1 root 2123397800 Aug  9 19:43 model-00014-of-00014.safetensors
-rw-r--r-- 1 root        392 Aug  9 19:43 Modelfile
-rw-r--r-- 1 root      63248 Aug  9 19:43 model.safetensors.index.json
-rw-r--r-- 1 root        485 Aug  9 19:43 special_tokens_map.json
-rw-r--r-- 1 root       6793 Aug  9 19:43 tokenizer_config.json
-rw-r--r-- 1 root   11422778 Aug  9 19:43 tokenizer.json

(llm-finetune-dtk2504) root@Ubuntu2204:/data/model/deepseek32b-medical# du -sh /data/model/deepseek32b-medical
62G	/data/model/deepseek32b-medical

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

------

### 4. 如何评判 Loss 到了最优值？

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

✅ **高效数据生成**：使用Easy Dataset创建高质量医疗QA数据集  
✅ **DCU优化训练**：充分发挥k100-AI硬件优势进行模型微调  
✅ **效果显著提升**：微调后模型在财经领域表现优异  
✅ **生产就绪**：提供完整的部署和监控方案  

---

## 📚 相关资源

1. [Easy Dataset](https://github.com/ConardLi/easy-dataset) 
2. [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)
3. [海光DCU开发社区](https://developer.sourcefind.cn/)
4. 医疗数据集
   - https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot
   - https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT/blob/main/medical_o1_sft_Chinese.json