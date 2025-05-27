# LLaMA Factory：02-基于DCU进行LoRA微调

> 仓库地址：https://developer.sourcefind.cn/codes/OpenDAS/llama-factory

## DCU支持模型结构列表

| 模型名                                                       | 参数量                           | Template  |
| ------------------------------------------------------------ | -------------------------------- | --------- |
| [Baichuan 2](https://huggingface.co/baichuan-inc)            | 7B/13B                           | baichuan2 |
| [ChatGLM3](https://huggingface.co/THUDM)                     | 6B                               | chatglm3  |
| [Gemma 2](https://huggingface.co/google)                     | 2B/9B                            | gemma     |
| [GLM-4/GLM-4-0414/GLM-Z1](https://huggingface.co/THUDM)      | 9B/32B                           | glm4      |
| [Llama 2](https://huggingface.co/meta-llama)                 | 7B/13B/70B                       | llama2    |
| [Llama 3/Llama 3.1](https://huggingface.co/meta-llama)       | 8B/70B                           | llama3    |
| [Llama 4](https://huggingface.co/meta-llama)                 | 109B/402B                        | llama4    |
| [OLMo](https://hf-mirror.com/allenai)                        | 1B/7B                            | olmo      |
| [Qwen (1-2.5) (Code/Math/MoE/QwQ)](https://huggingface.co/Qwen) | 0.5B/1.5B/3B/7B/14B/32B/72B/110B | qwen      |
| [Qwen3 (MoE)](https://huggingface.co/Qwen)                   | 0.6B/1.7B/4B/8B/14B/30B/32B/235B | qwen3     |
| [XVERSE](https://hf-mirror.com/xverse)                       | 7B/13B                           | xverse    |

持续更新中...

> [!NOTE]
>
> 注意：本版本仅支持deepseek蒸馏模型的监督微调(SFT)，可参考[deepseek-r1-distill_vllm](https://developer.sourcefind.cn/codes/modelzoo/deepseek-r1-distill_vllm)
>
> 对于所有“基座”（Base）模型，`template` 参数可以是 `default`, `alpaca`, `vicuna` 等任意值。但“对话”（Instruct/Chat）模型请务必使用**对应的模板**。
>
> 请务必在训练和推理时采用**完全一致**的模板。 您也可以在 [template.py]() 中添加自己的对话模板。
>
> **已知问题及解决方案**
>
> 1. `Baichuan 2` 需要卸载掉环境中的xformers库，当前仅支持Lora方式训练。
> 2. `XVERSE`在`tokenizer > 0.19`的版本下有兼容性问题报错`Exception: data did not match any variant of untagged enum PyPreTokenizerTypeWrappe`，需要使用[XVERSE-13B-256K-hf](https://huggingface.co/xverse/XVERSE-13B-256K/tree/main)中的`tokenizer_config.json.update`/`tokenizer.json.update`替换原有模型文件中的对应tokenizer文件，具体解决方法参考[xverse-ai/XVERSE-7B issues](https://github.com/xverse-ai/XVERSE-7B/issues/1)
> 3. `Qwen2`训练仅支持bf16格式，**fp16会出现loss为0，lr为0的问题**，参考[issues](https://github.com/hiyouga/LLaMA-Factory/issues/4848)
> 4. `deepspeed-cpu-offload-stage3`出现`RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`错误，是deepspeed本身bug，解决办法参考官方[issuse](https://github.com/microsoft/DeepSpeed/issues/5634)

## 如何使用

#### Anaconda

```bash
# 创建虚拟环境
conda create -n dcu_llm_fine python=3.10
conda activate dcu_llm_fine
```

关于本项目DCU显卡所需的特殊深度学习库可从[光合](https://developer.hpccube.com/tool/)开发者社区下载安装。

```shell
DTK: 25.04

python: 3.10

#  torch: 2.4.1 / 2.4.1+das.opt2.dtk2504
pip install torch-2.4.1+das.opt2.dtk2504-cp310-cp310-manylinux_2_28_x86_64.whl

# lmslim==0.2.1
wget --content-disposition 'https://download.sourcefind.cn:65024/file/4/lmslim/DAS1.5/lmslim-0.2.1+das.dtk2504-cp310-cp310-manylinux_2_28_x86_64.whl'

# flash-attn：2.6.1+das.opt4.dtk2504
wget --content-disposition 'https://download.sourcefind.cn:65024/file/4/flash_attn/DAS1.5/flash_attn-2.6.1+das.opt4.dtk2504-cp310-cp310-manylinux_2_28_x86_64.whl'

# vllm: ≥0.4.3 / 
wget --content-disposition 'https://download.sourcefind.cn:65024/file/4/vllm/DAS1.5/vllm-0.6.2+das.opt3.dtk2504-cp310-cp310-manylinux_2_28_x86_64.whl'

#deepspeed: 0.14.2+das.opt2.dtk2504
wget --content-disposition 'https://download.sourcefind.cn:65024/file/4/deepspeed/DAS1.5/deepspeed-0.14.2+das.opt2.dtk2504-cp310-cp310-manylinux_2_28_x86_64.whl'


# Tips：以上dtk驱动、python、torch等DCU相关工具版本需要严格一一对应
```

### 安装 LLaMA Factory

> [!TIP]
>
> 遇到包冲突时，可使用 `pip install --no-deps -e .` 解决。

```shell
git clone http://developer.hpccube.com/codes/OpenDAS/llama-factory.git
cd /your_code_path/llama_factory
pip install -e ".[torch,metrics]"

## llama4 需要单独安装以下包
##pip install git+https://github.com/hiyouga/transformers.git@llama4_train

# （可选）deepspeed多机训练
# pdsh安装，若已安装，可忽略。
# pdsh 工具的主要作用是在多个远程主机上并行执行命令。它是一个高效的并行远程 shell 工具，对于需要同时管理和操作大量服务器的系统管理员和集群用户来说非常有用。
# 安装需要root权限
cd ../
#下载解压
wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/pdsh/pdsh-2.29.tar.bz2 && tar -xf pdsh-2.29.tar.bz2
#编译安装
cd pdsh-2.29 && ./configure --with-ssh --enable-static-modules --prefix=/usr/local && make && make install
#测试
pdsh -V
```

## 数据集

<details><summary>预训练数据集</summary>
<ul data-sourcepos="125:1-136:0">
<li data-sourcepos="125:1-125:38"><a href="/codes/OpenDAS/llama-factory/-/blob/master/data/wiki_demo.txt">Wiki Demo (en)</a></li>
<li data-sourcepos="126:1-126:77"><a href="https://huggingface.co/datasets/tiiuae/falcon-refinedweb" rel="nofollow noreferrer noopener" target="_blank">RefinedWeb (en)</a></li>
<li data-sourcepos="127:1-127:89"><a href="https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2" rel="nofollow noreferrer noopener" target="_blank">RedPajama V2 (en)</a></li>
<li data-sourcepos="128:1-128:78"><a href="https://huggingface.co/datasets/olm/olm-wikipedia-20221220" rel="nofollow noreferrer noopener" target="_blank">Wikipedia (en)</a></li>
<li data-sourcepos="129:1-129:90"><a href="https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered" rel="nofollow noreferrer noopener" target="_blank">Wikipedia (zh)</a></li>
<li data-sourcepos="130:1-130:62"><a href="https://huggingface.co/datasets/EleutherAI/pile" rel="nofollow noreferrer noopener" target="_blank">Pile (en)</a></li>
<li data-sourcepos="131:1-131:70"><a href="https://huggingface.co/datasets/Skywork/SkyPile-150B" rel="nofollow noreferrer noopener" target="_blank">SkyPile (zh)</a></li>
<li data-sourcepos="132:1-132:71"><a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb" rel="nofollow noreferrer noopener" target="_blank">FineWeb (en)</a></li>
<li data-sourcepos="133:1-133:79"><a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu" rel="nofollow noreferrer noopener" target="_blank">FineWeb-Edu (en)</a></li>
<li data-sourcepos="134:1-134:69"><a href="https://huggingface.co/datasets/bigcode/the-stack" rel="nofollow noreferrer noopener" target="_blank">The Stack (en)</a></li>
<li data-sourcepos="135:1-136:0"><a href="https://huggingface.co/datasets/bigcode/starcoderdata" rel="nofollow noreferrer noopener" target="_blank">StarCoder (en)</a></li>
</ul>
</details>

<details><summary>指令微调数据集</summary>
<ul data-sourcepos="141:1-196:0">
<li data-sourcepos="141:1-141:40"><a href="/codes/OpenDAS/llama-factory/-/blob/master/data/identity.json">Identity (en&amp;zh)</a></li>
<li data-sourcepos="142:1-142:70"><a href="https://github.com/tatsu-lab/stanford_alpaca" rel="nofollow noreferrer noopener" target="_blank">Stanford Alpaca (en)</a></li>
<li data-sourcepos="143:1-143:73"><a href="https://github.com/ymcui/Chinese-LLaMA-Alpaca-3" rel="nofollow noreferrer noopener" target="_blank">Stanford Alpaca (zh)</a></li>
<li data-sourcepos="144:1-144:83"><a href="https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM" rel="nofollow noreferrer noopener" target="_blank">Alpaca GPT4 (en&amp;zh)</a></li>
<li data-sourcepos="145:1-145:107"><a href="https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2" rel="nofollow noreferrer noopener" target="_blank">Glaive Function Calling V2 (en&amp;zh)</a></li>
<li data-sourcepos="146:1-146:56"><a href="https://huggingface.co/datasets/GAIR/lima" rel="nofollow noreferrer noopener" target="_blank">LIMA (en)</a></li>
<li data-sourcepos="147:1-147:97"><a href="https://huggingface.co/datasets/JosephusCheung/GuanacoDataset" rel="nofollow noreferrer noopener" target="_blank">Guanaco Dataset (multilingual)</a></li>
<li data-sourcepos="148:1-148:73"><a href="https://huggingface.co/datasets/BelleGroup/train_2M_CN" rel="nofollow noreferrer noopener" target="_blank">BELLE 2M (zh)</a></li>
<li data-sourcepos="149:1-149:73"><a href="https://huggingface.co/datasets/BelleGroup/train_1M_CN" rel="nofollow noreferrer noopener" target="_blank">BELLE 1M (zh)</a></li>
<li data-sourcepos="150:1-150:77"><a href="https://huggingface.co/datasets/BelleGroup/train_0.5M_CN" rel="nofollow noreferrer noopener" target="_blank">BELLE 0.5M (zh)</a></li>
<li data-sourcepos="151:1-151:92"><a href="https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M" rel="nofollow noreferrer noopener" target="_blank">BELLE Dialogue 0.4M (zh)</a></li>
<li data-sourcepos="152:1-152:94"><a href="https://huggingface.co/datasets/BelleGroup/school_math_0.25M" rel="nofollow noreferrer noopener" target="_blank">BELLE School Math 0.25M (zh)</a></li>
<li data-sourcepos="153:1-153:98"><a href="https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M" rel="nofollow noreferrer noopener" target="_blank">BELLE Multiturn Chat 0.8M (zh)</a></li>
<li data-sourcepos="154:1-154:55"><a href="https://github.com/thunlp/UltraChat" rel="nofollow noreferrer noopener" target="_blank">UltraChat (en)</a></li>
<li data-sourcepos="155:1-155:81"><a href="https://huggingface.co/datasets/garage-bAInd/Open-Platypus" rel="nofollow noreferrer noopener" target="_blank">OpenPlatypus (en)</a></li>
<li data-sourcepos="156:1-156:81"><a href="https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k" rel="nofollow noreferrer noopener" target="_blank">CodeAlpaca 20k (en)</a></li>
<li data-sourcepos="157:1-157:82"><a href="https://huggingface.co/datasets/QingyiSi/Alpaca-CoT" rel="nofollow noreferrer noopener" target="_blank">Alpaca CoT (multilingual)</a></li>
<li data-sourcepos="158:1-158:69"><a href="https://huggingface.co/datasets/Open-Orca/OpenOrca" rel="nofollow noreferrer noopener" target="_blank">OpenOrca (en)</a></li>
<li data-sourcepos="159:1-159:69"><a href="https://huggingface.co/datasets/Open-Orca/SlimOrca" rel="nofollow noreferrer noopener" target="_blank">SlimOrca (en)</a></li>
<li data-sourcepos="160:1-160:77"><a href="https://huggingface.co/datasets/TIGER-Lab/MathInstruct" rel="nofollow noreferrer noopener" target="_blank">MathInstruct (en)</a></li>
<li data-sourcepos="161:1-161:82"><a href="https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M" rel="nofollow noreferrer noopener" target="_blank">Firefly 1.1M (zh)</a></li>
<li data-sourcepos="162:1-162:57"><a href="https://huggingface.co/datasets/wiki_qa" rel="nofollow noreferrer noopener" target="_blank">Wiki QA (en)</a></li>
<li data-sourcepos="163:1-163:62"><a href="https://huggingface.co/datasets/suolyer/webqa" rel="nofollow noreferrer noopener" target="_blank">Web QA (zh)</a></li>
<li data-sourcepos="164:1-164:69"><a href="https://huggingface.co/datasets/zxbsmk/webnovel_cn" rel="nofollow noreferrer noopener" target="_blank">WebNovel (zh)</a></li>
<li data-sourcepos="165:1-165:69"><a href="https://huggingface.co/datasets/berkeley-nest/Nectar" rel="nofollow noreferrer noopener" target="_blank">Nectar (en)</a></li>
<li data-sourcepos="166:1-166:83"><a href="https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data" rel="nofollow noreferrer noopener" target="_blank">deepctrl (en&amp;zh)</a></li>
<li data-sourcepos="167:1-167:83"><a href="https://huggingface.co/datasets/HasturOfficial/adgen" rel="nofollow noreferrer noopener" target="_blank">Advertise Generating (zh)</a></li>
<li data-sourcepos="168:1-168:109"><a href="https://huggingface.co/datasets/totally-not-an-llm/sharegpt-hyperfiltered-3k" rel="nofollow noreferrer noopener" target="_blank">ShareGPT Hyperfiltered (en)</a></li>
<li data-sourcepos="169:1-169:79"><a href="https://huggingface.co/datasets/shibing624/sharegpt_gpt4" rel="nofollow noreferrer noopener" target="_blank">ShareGPT4 (en&amp;zh)</a></li>
<li data-sourcepos="170:1-170:85"><a href="https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k" rel="nofollow noreferrer noopener" target="_blank">UltraChat 200k (en)</a></li>
<li data-sourcepos="171:1-171:75"><a href="https://huggingface.co/datasets/THUDM/AgentInstruct" rel="nofollow noreferrer noopener" target="_blank">AgentInstruct (en)</a></li>
<li data-sourcepos="172:1-172:75"><a href="https://huggingface.co/datasets/lmsys/lmsys-chat-1m" rel="nofollow noreferrer noopener" target="_blank">LMSYS Chat 1M (en)</a></li>
<li data-sourcepos="173:1-173:98"><a href="https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k" rel="nofollow noreferrer noopener" target="_blank">Evol Instruct V2 (en)</a></li>
<li data-sourcepos="174:1-174:77"><a href="https://huggingface.co/datasets/HuggingFaceTB/cosmopedia" rel="nofollow noreferrer noopener" target="_blank">Cosmopedia (en)</a></li>
<li data-sourcepos="175:1-175:70"><a href="https://huggingface.co/datasets/hfl/stem_zh_instruction" rel="nofollow noreferrer noopener" target="_blank">STEM (zh)</a></li>
<li data-sourcepos="176:1-176:74"><a href="https://huggingface.co/datasets/hfl/ruozhiba_gpt4_turbo" rel="nofollow noreferrer noopener" target="_blank">Ruozhiba (zh)</a></li>
<li data-sourcepos="177:1-177:70"><a href="https://huggingface.co/datasets/m-a-p/neo_sft_phase2" rel="nofollow noreferrer noopener" target="_blank">Neo-sft (zh)</a></li>
<li data-sourcepos="178:1-178:104"><a href="https://huggingface.co/datasets/Magpie-Align/Magpie-Pro-300K-Filtered" rel="nofollow noreferrer noopener" target="_blank">Magpie-Pro-300K-Filtered (en)</a></li>
<li data-sourcepos="179:1-179:85"><a href="https://huggingface.co/datasets/argilla/magpie-ultra-v0.1" rel="nofollow noreferrer noopener" target="_blank">Magpie-ultra-v0.1 (en)</a></li>
<li data-sourcepos="180:1-180:81"><a href="https://huggingface.co/datasets/TIGER-Lab/WebInstructSub" rel="nofollow noreferrer noopener" target="_blank">WebInstructSub (en)</a></li>
<li data-sourcepos="181:1-181:74"><a href="https://huggingface.co/datasets/O1-OPEN/OpenO1-SFT" rel="nofollow noreferrer noopener" target="_blank">OpenO1-SFT (en&amp;zh)</a></li>
<li data-sourcepos="182:1-182:87"><a href="https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k" rel="nofollow noreferrer noopener" target="_blank">Open-Thoughts (en)</a></li>
<li data-sourcepos="183:1-183:79"><a href="https://huggingface.co/datasets/open-r1/OpenR1-Math-220k" rel="nofollow noreferrer noopener" target="_blank">Open-R1-Math (en)</a></li>
<li data-sourcepos="184:1-184:119"><a href="https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT" rel="nofollow noreferrer noopener" target="_blank">Chinese-DeepSeek-R1-Distill (zh)</a></li>
<li data-sourcepos="185:1-185:85"><a href="https://huggingface.co/datasets/BUAADreamer/llava-en-zh-300k" rel="nofollow noreferrer noopener" target="_blank">LLaVA mixed (en&amp;zh)</a></li>
<li data-sourcepos="186:1-186:99"><a href="https://huggingface.co/datasets/jugg1024/pokemon-gpt4o-captions" rel="nofollow noreferrer noopener" target="_blank">Pokemon-gpt4o-captions (en&amp;zh)</a></li>
<li data-sourcepos="187:1-187:79"><a href="https://huggingface.co/datasets/mayflowergmbh/oasst_de" rel="nofollow noreferrer noopener" target="_blank">Open Assistant (de)</a></li>
<li data-sourcepos="188:1-188:78"><a href="https://huggingface.co/datasets/mayflowergmbh/dolly-15k_de" rel="nofollow noreferrer noopener" target="_blank">Dolly 15k (de)</a></li>
<li data-sourcepos="189:1-189:82"><a href="https://huggingface.co/datasets/mayflowergmbh/alpaca-gpt4_de" rel="nofollow noreferrer noopener" target="_blank">Alpaca GPT4 (de)</a></li>
<li data-sourcepos="190:1-190:92"><a href="https://huggingface.co/datasets/mayflowergmbh/openschnabeltier_de" rel="nofollow noreferrer noopener" target="_blank">OpenSchnabeltier (de)</a></li>
<li data-sourcepos="191:1-191:86"><a href="https://huggingface.co/datasets/mayflowergmbh/evol-instruct_de" rel="nofollow noreferrer noopener" target="_blank">Evol Instruct (de)</a></li>
<li data-sourcepos="192:1-192:74"><a href="https://huggingface.co/datasets/mayflowergmbh/dolphin_de" rel="nofollow noreferrer noopener" target="_blank">Dolphin (de)</a></li>
<li data-sourcepos="193:1-193:74"><a href="https://huggingface.co/datasets/mayflowergmbh/booksum_de" rel="nofollow noreferrer noopener" target="_blank">Booksum (de)</a></li>
<li data-sourcepos="194:1-194:82"><a href="https://huggingface.co/datasets/mayflowergmbh/airoboros-3.0_de" rel="nofollow noreferrer noopener" target="_blank">Airoboros (de)</a></li>
<li data-sourcepos="195:1-196:0"><a href="https://huggingface.co/datasets/mayflowergmbh/ultra-chat_de" rel="nofollow noreferrer noopener" target="_blank">Ultrachat (de)</a></li>
</ul>
</details>

<details><summary>偏好数据集</summary>
<ul data-sourcepos="201:1-211:0">
<li data-sourcepos="201:1-201:76"><a href="https://huggingface.co/datasets/hiyouga/DPO-En-Zh-20k" rel="nofollow noreferrer noopener" target="_blank">DPO mixed (en&amp;zh)</a></li>
<li data-sourcepos="202:1-202:93"><a href="https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized" rel="nofollow noreferrer noopener" target="_blank">UltraFeedback (en)</a></li>
<li data-sourcepos="203:1-203:64"><a href="https://huggingface.co/datasets/m-a-p/COIG-P" rel="nofollow noreferrer noopener" target="_blank">COIG-P (en&amp;zh)</a></li>
<li data-sourcepos="204:1-204:71"><a href="https://huggingface.co/datasets/openbmb/RLHF-V-Dataset" rel="nofollow noreferrer noopener" target="_blank">RLHF-V (en)</a></li>
<li data-sourcepos="205:1-205:70"><a href="https://huggingface.co/datasets/Zhihui/VLFeedback" rel="nofollow noreferrer noopener" target="_blank">VLFeedback (en)</a></li>
<li data-sourcepos="206:1-206:77"><a href="https://huggingface.co/datasets/Intel/orca_dpo_pairs" rel="nofollow noreferrer noopener" target="_blank">Orca DPO Pairs (en)</a></li>
<li data-sourcepos="207:1-207:67"><a href="https://huggingface.co/datasets/Anthropic/hh-rlhf" rel="nofollow noreferrer noopener" target="_blank">HH-RLHF (en)</a></li>
<li data-sourcepos="208:1-208:69"><a href="https://huggingface.co/datasets/berkeley-nest/Nectar" rel="nofollow noreferrer noopener" target="_blank">Nectar (en)</a></li>
<li data-sourcepos="209:1-209:88"><a href="https://huggingface.co/datasets/mayflowergmbh/intel_orca_dpo_pairs_de" rel="nofollow noreferrer noopener" target="_blank">Orca DPO (de)</a></li>
<li data-sourcepos="210:1-211:0"><a href="https://huggingface.co/datasets/argilla/kto-mix-15k" rel="nofollow noreferrer noopener" target="_blank">KTO mixed (en)</a></li>
</ul>
</details>

部分数据集的使用需要确认，我们推荐使用下述命令登录您的 Hugging Face 账户。

```shell
pip install --upgrade huggingface_hub
huggingface-cli login
```

### 数据准备

关于数据集文件的格式，请参考 [data/README_zh.md]() 的内容。你可以使用 HuggingFace / ModelScope 上的数据集或加载本地数据集。

> [!NOTE]
>
> 使用自定义数据集时，请更新 `data/dataset_info.json` 文件。

## 如何使用

### 快速开始

下面三行命令分别对 Llama3-8B-Instruct 模型进行 LoRA **微调**、**推理**和**合并**。根据实际情况修改参数，如`model_name_or_path`/`dataset`/`template`等。

```shell
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

高级用法请参考 [examples/README_zh.md]()（包括多 GPU 微调）。

> [!TIP]
>
> 使用 `llamafactory-cli help` 显示帮助信息。
>
> 自有数据集推理精度验证方法推荐使用：`python scripts/vllm_infer.py`生成结果，`python scripts/eval_bleu_rouge.py`计算得分，具体参数信息请参考脚本内容。

## 参考资料

- [README_zh]()
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

# 附录

