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
> 请务必在训练和推理时采用**完全一致**的模板。 你也可以在 [template.py]() 中添加自己的对话模板。
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
DCU: K100-AI
DTK: 25.04
python: 3.10

#  torch: 2.4.1 / 2.4.1+das.opt2.dtk2504
pip install torch-2.4.1+das.opt2.dtk2504-cp310-cp310-manylinux_2_28_x86_64.whl

# lmslim==0.2.1
wget --content-disposition 'https://download.sourcefind.cn:65024/file/4/lmslim/DAS1.5/lmslim-0.2.1+das.dtk2504-cp310-cp310-manylinux_2_28_x86_64.whl'

# flash-attn：2.6.1+das.opt4.dtk2504
wget --content-disposition 'https://download.sourcefind.cn:65024/file/4/flash_attn/DAS1.5/flash_attn-2.6.1+das.opt4.dtk2504-cp310-cp310-manylinux_2_28_x86_64.whl'

# vllm: ≥0.4.3 / 0.6.2+das.opt3.dtk2504
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

# deepspeed多机训练（可选）
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

部分数据集的使用需要确认，我们推荐使用下述命令登录你的 Hugging Face 账户。

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
# LoRA 微调
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
# 推理
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
# 合并(模型导出)
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

高级用法请参考 [examples/README_zh.md]()（包括多 GPU 微调）。

> [!TIP]
>
> 使用 `llamafactory-cli help` 显示帮助信息。
>
> 自有数据集推理精度验证方法推荐使用：`python scripts/vllm_infer.py`生成结果，`python scripts/eval_bleu_rouge.py`计算得分，具体参数信息请参考脚本内容。

### 过程详解

#### Llama3-8B-Instruct 模型进行 LoRA 微调

**配置文件**

```bash
### model
#model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
model_name_or_path: /root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: identity,alpaca_en_demo
template: llama3
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/llama3-8b/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
# eval_dataset: alpaca_en_demo
# val_size: 0.1
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: 500
```

**输出信息**

```bash
(dcu_llm_fine) root@Ubuntu2204:~/AI-BOX/code/dcu/llama-factory# llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
[2025-05-27 15:16:40,207] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[INFO|2025-05-27 15:16:43] llamafactory.cli:143 >> Initializing 8 distributed tasks at: 127.0.0.1:54447
W0527 15:16:45.699000 140119427319616 torch/distributed/run.py:779] 
W0527 15:16:45.699000 140119427319616 torch/distributed/run.py:779] *****************************************
W0527 15:16:45.699000 140119427319616 torch/distributed/run.py:779] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W0527 15:16:45.699000 140119427319616 torch/distributed/run.py:779] *****************************************
[2025-05-27 15:16:51,217] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-27 15:16:51,246] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-27 15:16:51,246] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-27 15:16:51,291] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-27 15:16:51,296] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-27 15:16:51,524] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-05-27 15:16:51,616] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0527 15:16:51.665375 58626 ProcessGroupNCCL.cpp:869] [PG 0 Rank 2] ProcessGroupNCCL initialization options: size: 8, global rank: 2, TIMEOUT(ms): 180000000000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: 0, PG Name: 0
I0527 15:16:51.665462 58626 ProcessGroupNCCL.cpp:878] [PG 0 Rank 2] ProcessGroupNCCL environments: NCCL version: 2.18.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 600, TORCH_NCCL_TRACE_BUFFER_SIZE: 0, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0
[INFO|2025-05-27 15:16:51] llamafactory.hparams.parser:401 >> Process rank: 2, world size: 8, device: cuda:2, distributed training: True, compute dtype: torch.bfloat16
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0527 15:16:51.688997 58625 ProcessGroupNCCL.cpp:869] [PG 0 Rank 1] ProcessGroupNCCL initialization options: size: 8, global rank: 1, TIMEOUT(ms): 180000000000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: 0, PG Name: 0
I0527 15:16:51.689081 58625 ProcessGroupNCCL.cpp:878] [PG 0 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.18.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 600, TORCH_NCCL_TRACE_BUFFER_SIZE: 0, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0
[INFO|2025-05-27 15:16:51] llamafactory.hparams.parser:401 >> Process rank: 1, world size: 8, device: cuda:1, distributed training: True, compute dtype: torch.bfloat16
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0527 15:16:51.705945 58624 ProcessGroupNCCL.cpp:869] [PG 0 Rank 0] ProcessGroupNCCL initialization options: size: 8, global rank: 0, TIMEOUT(ms): 180000000000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: 0, PG Name: 0
I0527 15:16:51.706022 58624 ProcessGroupNCCL.cpp:878] [PG 0 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.18.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 600, TORCH_NCCL_TRACE_BUFFER_SIZE: 0, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0
[INFO|2025-05-27 15:16:51] llamafactory.hparams.parser:143 >> Set `ddp_find_unused_parameters` to False in DDP training since LoRA is enabled.
[INFO|2025-05-27 15:16:51] llamafactory.hparams.parser:401 >> Process rank: 0, world size: 8, device: cuda:0, distributed training: True, compute dtype: torch.bfloat16
[INFO|tokenization_utils_base.py:2212] 2025-05-27 15:16:51,714 >> loading file tokenizer.json
[INFO|tokenization_utils_base.py:2212] 2025-05-27 15:16:51,714 >> loading file tokenizer.model
[INFO|tokenization_utils_base.py:2212] 2025-05-27 15:16:51,714 >> loading file added_tokens.json
[INFO|tokenization_utils_base.py:2212] 2025-05-27 15:16:51,714 >> loading file special_tokens_map.json
[INFO|tokenization_utils_base.py:2212] 2025-05-27 15:16:51,714 >> loading file tokenizer_config.json
[2025-05-27 15:16:51,731] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0527 15:16:51.741318 58630 ProcessGroupNCCL.cpp:869] [PG 0 Rank 6] ProcessGroupNCCL initialization options: size: 8, global rank: 6, TIMEOUT(ms): 180000000000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: 0, PG Name: 0
I0527 15:16:51.741406 58630 ProcessGroupNCCL.cpp:878] [PG 0 Rank 6] ProcessGroupNCCL environments: NCCL version: 2.18.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 600, TORCH_NCCL_TRACE_BUFFER_SIZE: 0, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0
[INFO|2025-05-27 15:16:51] llamafactory.hparams.parser:401 >> Process rank: 6, world size: 8, device: cuda:6, distributed training: True, compute dtype: torch.bfloat16
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0527 15:16:51.762657 58629 ProcessGroupNCCL.cpp:869] [PG 0 Rank 5] ProcessGroupNCCL initialization options: size: 8, global rank: 5, TIMEOUT(ms): 180000000000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: 0, PG Name: 0
I0527 15:16:51.762769 58629 ProcessGroupNCCL.cpp:878] [PG 0 Rank 5] ProcessGroupNCCL environments: NCCL version: 2.18.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 600, TORCH_NCCL_TRACE_BUFFER_SIZE: 0, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0
[INFO|2025-05-27 15:16:51] llamafactory.hparams.parser:401 >> Process rank: 5, world size: 8, device: cuda:5, distributed training: True, compute dtype: torch.bfloat16
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0527 15:16:51.996441 58627 ProcessGroupNCCL.cpp:869] [PG 0 Rank 3] ProcessGroupNCCL initialization options: size: 8, global rank: 3, TIMEOUT(ms): 180000000000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: 0, PG Name: 0
I0527 15:16:51.996533 58627 ProcessGroupNCCL.cpp:878] [PG 0 Rank 3] ProcessGroupNCCL environments: NCCL version: 2.18.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 600, TORCH_NCCL_TRACE_BUFFER_SIZE: 0, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0
[INFO|2025-05-27 15:16:52] llamafactory.hparams.parser:401 >> Process rank: 3, world size: 8, device: cuda:3, distributed training: True, compute dtype: torch.bfloat16
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0527 15:16:52.091248 58628 ProcessGroupNCCL.cpp:869] [PG 0 Rank 4] ProcessGroupNCCL initialization options: size: 8, global rank: 4, TIMEOUT(ms): 180000000000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: 0, PG Name: 0
I0527 15:16:52.091320 58628 ProcessGroupNCCL.cpp:878] [PG 0 Rank 4] ProcessGroupNCCL environments: NCCL version: 2.18.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 600, TORCH_NCCL_TRACE_BUFFER_SIZE: 0, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0
[INFO|2025-05-27 15:16:52] llamafactory.hparams.parser:401 >> Process rank: 4, world size: 8, device: cuda:4, distributed training: True, compute dtype: torch.bfloat16
[INFO|tokenization_utils_base.py:2478] 2025-05-27 15:16:52,194 >> Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
[INFO|configuration_utils.py:670] 2025-05-27 15:16:52,195 >> loading configuration file /root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct/config.json
[INFO|configuration_utils.py:739] 2025-05-27 15:16:52,196 >> Model config LlamaConfig {
  "_name_or_path": "/root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128009,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.0",
  "use_cache": true,
  "vocab_size": 128256
}

[INFO|tokenization_utils_base.py:2212] 2025-05-27 15:16:52,197 >> loading file tokenizer.json
[INFO|tokenization_utils_base.py:2212] 2025-05-27 15:16:52,197 >> loading file tokenizer.model
[INFO|tokenization_utils_base.py:2212] 2025-05-27 15:16:52,197 >> loading file added_tokens.json
[INFO|tokenization_utils_base.py:2212] 2025-05-27 15:16:52,197 >> loading file special_tokens_map.json
[INFO|tokenization_utils_base.py:2212] 2025-05-27 15:16:52,197 >> loading file tokenizer_config.json
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0527 15:16:52.212648 58631 ProcessGroupNCCL.cpp:869] [PG 0 Rank 7] ProcessGroupNCCL initialization options: size: 8, global rank: 7, TIMEOUT(ms): 180000000000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: 0, PG Name: 0
I0527 15:16:52.212738 58631 ProcessGroupNCCL.cpp:878] [PG 0 Rank 7] ProcessGroupNCCL environments: NCCL version: 2.18.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 600, TORCH_NCCL_TRACE_BUFFER_SIZE: 0, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0
[INFO|2025-05-27 15:16:52] llamafactory.hparams.parser:401 >> Process rank: 7, world size: 8, device: cuda:7, distributed training: True, compute dtype: torch.bfloat16
I0527 15:16:52.585548 58626 ProcessGroupNCCL.cpp:3958] [PG 0 Rank 2]  using GPU 2 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device.
I0527 15:16:52.587344 58625 ProcessGroupNCCL.cpp:3958] [PG 0 Rank 1]  using GPU 1 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device.
[INFO|tokenization_utils_base.py:2478] 2025-05-27 15:16:52,636 >> Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
I0527 15:16:52.639250 58630 ProcessGroupNCCL.cpp:3958] [PG 0 Rank 6]  using GPU 6 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device.
[INFO|2025-05-27 15:16:52] llamafactory.data.template:143 >> Add pad token: <|eot_id|>
[INFO|2025-05-27 15:16:52] llamafactory.data.template:143 >> Add <|eot_id|>,<|eom_id|> to stop words.
[WARNING|2025-05-27 15:16:52] llamafactory.data.template:148 >> New tokens have been added, make sure `resize_vocab` is True.
[INFO|2025-05-27 15:16:52] llamafactory.data.loader:143 >> Loading dataset identity.json...
I0527 15:16:52.710393 58629 ProcessGroupNCCL.cpp:3958] [PG 0 Rank 5]  using GPU 5 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device.
I0527 15:16:52.878005 58627 ProcessGroupNCCL.cpp:3958] [PG 0 Rank 3]  using GPU 3 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device.
I0527 15:16:53.090049 58628 ProcessGroupNCCL.cpp:3958] [PG 0 Rank 4]  using GPU 4 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device.
I0527 15:16:53.116564 58631 ProcessGroupNCCL.cpp:3958] [PG 0 Rank 7]  using GPU 7 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device.
Converting format of dataset (num_proc=16): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 91/91 [00:00<00:00, 437.48 examples/s]
[INFO|2025-05-27 15:16:56] llamafactory.data.loader:143 >> Loading dataset alpaca_en_demo.json...
Converting format of dataset (num_proc=16): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 4385.32 examples/s]
I0527 15:16:57.216090 58624 ProcessGroupNCCL.cpp:3958] [PG 0 Rank 0]  using GPU 0 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device.
I0527 15:16:57.219332 58624 ProcessGroupNCCL.cpp:2074] [PG 0 (default_pg) Rank 0] ProcessGroupNCCL broadcast unique ID through store took 0.117509 ms
I0527 15:16:57.219606 58626 ProcessGroupNCCL.cpp:2074] [PG 0 (default_pg) Rank 2] ProcessGroupNCCL broadcast unique ID through store took 4633.94 ms
I0527 15:16:57.219616 58625 ProcessGroupNCCL.cpp:2074] [PG 0 (default_pg) Rank 1] ProcessGroupNCCL broadcast unique ID through store took 4632.15 ms
I0527 15:16:57.219633 58630 ProcessGroupNCCL.cpp:2074] [PG 0 (default_pg) Rank 6] ProcessGroupNCCL broadcast unique ID through store took 4580.25 ms
I0527 15:16:57.219646 58629 ProcessGroupNCCL.cpp:2074] [PG 0 (default_pg) Rank 5] ProcessGroupNCCL broadcast unique ID through store took 4509.13 ms
I0527 15:16:57.219699 58627 ProcessGroupNCCL.cpp:2074] [PG 0 (default_pg) Rank 3] ProcessGroupNCCL broadcast unique ID through store took 4341.57 ms
I0527 15:16:57.219751 58631 ProcessGroupNCCL.cpp:2074] [PG 0 (default_pg) Rank 7] ProcessGroupNCCL broadcast unique ID through store took 4103.04 ms
I0527 15:16:57.220209 58628 ProcessGroupNCCL.cpp:2074] [PG 0 (default_pg) Rank 4] ProcessGroupNCCL broadcast unique ID through store took 4130 ms
I0527 15:16:57.738664 58629 ProcessGroupNCCL.cpp:2183] [PG 0 (default_pg) Rank 5] ProcessGroupNCCL created ncclComm_ 0x555b8d27b370 on CUDA device: 
I0527 15:16:57.738689 58628 ProcessGroupNCCL.cpp:2183] [PG 0 (default_pg) Rank 4] ProcessGroupNCCL created ncclComm_ 0x55cec3d7ed80 on CUDA device:  
I0527 15:16:57.738708 58627 ProcessGroupNCCL.cpp:2183] [PG 0 (default_pg) Rank 3] ProcessGroupNCCL created ncclComm_ 0x55b616e29000 on CUDA device:  
I0527 15:16:57.738700 58631 ProcessGroupNCCL.cpp:2183] [PG 0 (default_pg) Rank 7] ProcessGroupNCCL created ncclComm_ 0x5646eabe8fc0 on CUDA device: 
I0527 15:16:57.738725 58624 ProcessGroupNCCL.cpp:2183] [PG 0 (default_pg) Rank 0] ProcessGroupNCCL created ncclComm_ 0x55efff5bd950 on CUDA device: 
I0527 15:16:57.738736 58629 ProcessGroupNCCL.cpp:2188] [PG 0 (default_pg) Rank 5] NCCL_DEBUG: N/A
I0527 15:16:57.738741 58628 ProcessGroupNCCL.cpp:2188] [PG 0 (default_pg) Rank 4] NCCL_DEBUG: N/A
I0527 15:16:57.738757 58627 ProcessGroupNCCL.cpp:2188] [PG 0 (default_pg) Rank 3] NCCL_DEBUG: N/A
I0527 15:16:57.738765 58631 ProcessGroupNCCL.cpp:2188] [PG 0 (default_pg) Rank 7] NCCL_DEBUG: N/A
I0527 15:16:57.738770 58624 ProcessGroupNCCL.cpp:2188] [PG 0 (default_pg) Rank 0] NCCL_DEBUG: N/A
I0527 15:16:57.738806 58630 ProcessGroupNCCL.cpp:2183] [PG 0 (default_pg) Rank 6] ProcessGroupNCCL created ncclComm_ 0x55879b0e7780 on CUDA device:  
I0527 15:16:57.738845 58630 ProcessGroupNCCL.cpp:2188] [PG 0 (default_pg) Rank 6] NCCL_DEBUG: N/A
I0527 15:16:57.738911 58626 ProcessGroupNCCL.cpp:2183] [PG 0 (default_pg) Rank 2] ProcessGroupNCCL created ncclComm_ 0x5557fde5af00 on CUDA device:  
I0527 15:16:57.738940 58626 ProcessGroupNCCL.cpp:2188] [PG 0 (default_pg) Rank 2] NCCL_DEBUG: N/A
I0527 15:16:57.738934 58625 ProcessGroupNCCL.cpp:2183] [PG 0 (default_pg) Rank 1] ProcessGroupNCCL created ncclComm_ 0x55a7cf86dfd0 on CUDA device:  
I0527 15:16:57.738967 58625 ProcessGroupNCCL.cpp:2188] [PG 0 (default_pg) Rank 1] NCCL_DEBUG: N/A
Running tokenizer on dataset (num_proc=16): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1091/1091 [00:03<00:00, 348.62 examples/s]
training example:
input_ids:
[128000, 128006, 882, 128007, 271, 6151, 128009, 128006, 78191, 128007, 271, 9906, 0, 358, 1097, 5991, 609, 39254, 459, 15592, 18328, 8040, 555, 5991, 3170, 3500, 13, 2650, 649, 358, 7945, 499, 3432, 30, 128009]
inputs:
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

hi<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hello! I am {{name}}, an AI assistant developed by {{author}}. How can I assist you today?<|eot_id|>
label_ids:
[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 9906, 0, 358, 1097, 5991, 609, 39254, 459, 15592, 18328, 8040, 555, 5991, 3170, 3500, 13, 2650, 649, 358, 7945, 499, 3432, 30, 128009]
labels:
Hello! I am {{name}}, an AI assistant developed by {{author}}. How can I assist you today?<|eot_id|>
[INFO|configuration_utils.py:670] 2025-05-27 15:17:01,523 >> loading configuration file /root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct/config.json
[INFO|configuration_utils.py:739] 2025-05-27 15:17:01,524 >> Model config LlamaConfig {
  "_name_or_path": "/root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128009,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.0",
  "use_cache": true,
  "vocab_size": 128256
}

[INFO|2025-05-27 15:17:01] llamafactory.model.model_utils.kv_cache:143 >> KV cache is disabled during training.
[INFO|modeling_utils.py:3723] 2025-05-27 15:17:01,797 >> loading weights file /root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct/model.safetensors.index.json
[INFO|modeling_utils.py:1622] 2025-05-27 15:17:01,797 >> Instantiating LlamaForCausalLM model under default dtype torch.bfloat16.
[WARNING|logging.py:328] 2025-05-27 15:17:01,798 >> Using the `SDPA` attention implementation on multi-gpu setup with ROCM may lead to performance issues due to the FA backend. Disabling it to use alternative backends.
[INFO|configuration_utils.py:1099] 2025-05-27 15:17:01,798 >> Generate config GenerationConfig {
  "bos_token_id": 128000,
  "eos_token_id": 128009,
  "use_cache": false
}

Loading checkpoint shards:   0%|                                                                                                                                                   | 0/4 [00:00<?, ?it/s]Using the `SDPA` attention implementation on multi-gpu setup with ROCM may lead to performance issues due to the FA backend. Disabling it to use alternative backends.
Using the `SDPA` attention implementation on multi-gpu setup with ROCM may lead to performance issues due to the FA backend. Disabling it to use alternative backends.
Using the `SDPA` attention implementation on multi-gpu setup with ROCM may lead to performance issues due to the FA backend. Disabling it to use alternative backends.
Using the `SDPA` attention implementation on multi-gpu setup with ROCM may lead to performance issues due to the FA backend. Disabling it to use alternative backends.
Using the `SDPA` attention implementation on multi-gpu setup with ROCM may lead to performance issues due to the FA backend. Disabling it to use alternative backends.
Loading checkpoint shards:   0%|                                                                                                                                                   | 0/4 [00:00<?, ?it/s]Using the `SDPA` attention implementation on multi-gpu setup with ROCM may lead to performance issues due to the FA backend. Disabling it to use alternative backends.
Using the `SDPA` attention implementation on multi-gpu setup with ROCM may lead to performance issues due to the FA backend. Disabling it to use alternative backends.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:14<00:00,  3.57s/it]
[INFO|modeling_utils.py:4568] 2025-05-27 15:17:16,169 >> All model checkpoint weights were used when initializing LlamaForCausalLM.

[INFO|modeling_utils.py:4576] 2025-05-27 15:17:16,169 >> All the weights of LlamaForCausalLM were initialized from the model checkpoint at /root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct.
If your task is similar to the task the model of the checkpoint was trained on, you can already use LlamaForCausalLM for predictions without further training.
[INFO|configuration_utils.py:1052] 2025-05-27 15:17:16,173 >> loading configuration file /root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct/generation_config.json
[INFO|configuration_utils.py:1099] 2025-05-27 15:17:16,173 >> Generate config GenerationConfig {
  "bos_token_id": 128000,
  "do_sample": true,
  "eos_token_id": [
    128001,
    128009
  ],
  "max_length": 4096,
  "temperature": 0.6,
  "top_p": 0.9
}

[INFO|2025-05-27 15:17:16] llamafactory.model.model_utils.checkpointing:143 >> Gradient checkpointing enabled.
[INFO|2025-05-27 15:17:16] llamafactory.model.model_utils.attention:143 >> Using torch SDPA for faster training and inference.
[INFO|2025-05-27 15:17:16] llamafactory.model.adapter:143 >> Upcasting trainable params to float32.
[INFO|2025-05-27 15:17:16] llamafactory.model.adapter:143 >> Fine-tuning method: LoRA
[INFO|2025-05-27 15:17:16] llamafactory.model.model_utils.misc:143 >> Found linear modules: v_proj,q_proj,k_proj,down_proj,o_proj,gate_proj,up_proj
Loading checkpoint shards:  75%|████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                  | 3/4 [00:14<00:04,  4.62s/it][INFO|2025-05-27 15:17:16] llamafactory.model.loader:143 >> trainable params: 20,971,520 || all params: 8,051,232,768 || trainable%: 0.2605
[INFO|trainer.py:667] 2025-05-27 15:17:16,639 >> Using auto half precision backend
[WARNING|2025-05-27 15:17:16] llamafactory.train.callbacks:154 >> Previous trainer log in this folder will be deleted.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:15<00:00,  3.78s/it]
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:15<00:00,  3.90s/it]
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:15<00:00,  3.98s/it]
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:16<00:00,  4.06s/it]
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:16<00:00,  4.08s/it]
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:16<00:00,  4.01s/it]
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:16<00:00,  4.09s/it]
[INFO|trainer.py:2243] 2025-05-27 15:17:20,382 >> ***** Running training *****
[INFO|trainer.py:2244] 2025-05-27 15:17:20,382 >>   Num examples = 1,091
[INFO|trainer.py:2245] 2025-05-27 15:17:20,382 >>   Num Epochs = 3
[INFO|trainer.py:2246] 2025-05-27 15:17:20,382 >>   Instantaneous batch size per device = 1
[INFO|trainer.py:2249] 2025-05-27 15:17:20,382 >>   Total train batch size (w. parallel, distributed & accumulation) = 64
[INFO|trainer.py:2250] 2025-05-27 15:17:20,383 >>   Gradient Accumulation steps = 8
[INFO|trainer.py:2251] 2025-05-27 15:17:20,383 >>   Total optimization steps = 51
[INFO|trainer.py:2252] 2025-05-27 15:17:20,387 >>   Number of trainable parameters = 20,971,520
  0%|                                                                                                                                                                             | 0/51 [00:00<?, ?it/s]/root/anaconda3/envs/dcu_llm_fine/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py:655: UserWarning: 1Torch was not compiled with memory efficient attention. (Triggered internally at /home/pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:627.)
  attn_output = torch.nn.functional.scaled_dot_product_attention(
/root/anaconda3/envs/dcu_llm_fine/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py:655: UserWarning: 1Torch was not compiled with memory efficient attention. (Triggered internally at /home/pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:627.)
  attn_output = torch.nn.functional.scaled_dot_product_attention(
/root/anaconda3/envs/dcu_llm_fine/lib/python3.10/site-packages/torch/utils/checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
/root/anaconda3/envs/dcu_llm_fine/lib/python3.10/site-packages/torch/utils/checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
/root/anaconda3/envs/dcu_llm_fine/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py:655: UserWarning: 1Torch was not compiled with memory efficient attention. (Triggered internally at /home/pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:627.)
  attn_output = torch.nn.functional.scaled_dot_product_attention(
/root/anaconda3/envs/dcu_llm_fine/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py:655: UserWarning: 1Torch was not compiled with memory efficient attention. (Triggered internally at /home/pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:627.)
  attn_output = torch.nn.functional.scaled_dot_product_attention(
/root/anaconda3/envs/dcu_llm_fine/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py:655: UserWarning: 1Torch was not compiled with memory efficient attention. (Triggered internally at /home/pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:627.)
  attn_output = torch.nn.functional.scaled_dot_product_attention(
/root/anaconda3/envs/dcu_llm_fine/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py:655: UserWarning: 1Torch was not compiled with memory efficient attention. (Triggered internally at /home/pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:627.)
  attn_output = torch.nn.functional.scaled_dot_product_attention(
/root/anaconda3/envs/dcu_llm_fine/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py:655: UserWarning: 1Torch was not compiled with memory efficient attention. (Triggered internally at /home/pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:627.)
  attn_output = torch.nn.functional.scaled_dot_product_attention(
/root/anaconda3/envs/dcu_llm_fine/lib/python3.10/site-packages/torch/utils/checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
/root/anaconda3/envs/dcu_llm_fine/lib/python3.10/site-packages/torch/utils/checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
/root/anaconda3/envs/dcu_llm_fine/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py:655: UserWarning: 1Torch was not compiled with memory efficient attention. (Triggered internally at /home/pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:627.)
  attn_output = torch.nn.functional.scaled_dot_product_attention(
/root/anaconda3/envs/dcu_llm_fine/lib/python3.10/site-packages/torch/utils/checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
/root/anaconda3/envs/dcu_llm_fine/lib/python3.10/site-packages/torch/utils/checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
/root/anaconda3/envs/dcu_llm_fine/lib/python3.10/site-packages/torch/utils/checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
/root/anaconda3/envs/dcu_llm_fine/lib/python3.10/site-packages/torch/utils/checkpoint.py:295: FutureWarning: `torch.cpu.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cpu', args...)` instead.
  with torch.enable_grad(), device_autocast_ctx, torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
{'loss': 1.4091, 'grad_norm': 1.0385138988494873, 'learning_rate': 9.806308479691595e-05, 'epoch': 0.58}                                                                                                 
{'loss': 1.0404, 'grad_norm': 0.6730291247367859, 'learning_rate': 7.795964517353735e-05, 'epoch': 1.17}                                                                                                 
{'loss': 0.9658, 'grad_norm': 0.41746750473976135, 'learning_rate': 4.477357683661734e-05, 'epoch': 1.75}                                                                                                
{'loss': 0.9389, 'grad_norm': 0.39423027634620667, 'learning_rate': 1.4033009983067452e-05, 'epoch': 2.34}                                                                                               
{'loss': 0.894, 'grad_norm': 0.4427163302898407, 'learning_rate': 1.2179748700879012e-07, 'epoch': 2.92}                                                                                                 
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [04:54<00:00,  4.38s/it][INFO|trainer.py:3705] 2025-05-27 15:22:15,098 >> Saving model checkpoint to saves/llama3-8b/lora/sft/checkpoint-51
[INFO|configuration_utils.py:670] 2025-05-27 15:22:15,118 >> loading configuration file /root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct/config.json
[INFO|configuration_utils.py:739] 2025-05-27 15:22:15,119 >> Model config LlamaConfig {
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128009,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.0",
  "use_cache": true,
  "vocab_size": 128256
}

[INFO|tokenization_utils_base.py:2649] 2025-05-27 15:22:15,589 >> tokenizer config file saved in saves/llama3-8b/lora/sft/checkpoint-51/tokenizer_config.json
[INFO|tokenization_utils_base.py:2658] 2025-05-27 15:22:15,589 >> Special tokens file saved in saves/llama3-8b/lora/sft/checkpoint-51/special_tokens_map.json
[INFO|trainer.py:2505] 2025-05-27 15:22:16,715 >> 

Training completed. Do not forget to share your model on huggingface.co/models =)


{'train_runtime': 296.3281, 'train_samples_per_second': 11.045, 'train_steps_per_second': 0.172, 'train_loss': 1.0480897531789892, 'epoch': 2.98}                                                        
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [04:56<00:00,  5.81s/it]
[INFO|trainer.py:3705] 2025-05-27 15:22:16,719 >> Saving model checkpoint to saves/llama3-8b/lora/sft
I0527 15:22:16.720973 58627 ProcessGroupNCCL.cpp:1166] [PG 0 (default_pg) Rank 3] Launching ProcessGroupNCCL abort asynchrounously.
I0527 15:22:16.721046 58625 ProcessGroupNCCL.cpp:1166] [PG 0 (default_pg) Rank 1] Launching ProcessGroupNCCL abort asynchrounously.
I0527 15:22:16.721163 58626 ProcessGroupNCCL.cpp:1166] [PG 0 (default_pg) Rank 2] Launching ProcessGroupNCCL abort asynchrounously.
I0527 15:22:16.721268 100243 ProcessGroupNCCL.cpp:1113] [PG 0 (default_pg) Rank 3] ProcessGroupNCCL destroying ncclComm_ 0x55b616e29000 on CUDA device: 3
I0527 15:22:16.721335 100243 NCCLUtils.hpp:364] Aborting ncclComm_ 0x55b616e29000 with reason: No abort reason provided.
I0527 15:22:16.721357 100244 ProcessGroupNCCL.cpp:1113] [PG 0 (default_pg) Rank 1] ProcessGroupNCCL destroying ncclComm_ 0x55a7cf86dfd0 on CUDA device: 1
I0527 15:22:16.721426 100244 NCCLUtils.hpp:364] Aborting ncclComm_ 0x55a7cf86dfd0 with reason: No abort reason provided.
I0527 15:22:16.721447 100245 ProcessGroupNCCL.cpp:1113] [PG 0 (default_pg) Rank 2] ProcessGroupNCCL destroying ncclComm_ 0x5557fde5af00 on CUDA device: 2
I0527 15:22:16.721467 100245 NCCLUtils.hpp:364] Aborting ncclComm_ 0x5557fde5af00 with reason: No abort reason provided.
I0527 15:22:16.721490 58630 ProcessGroupNCCL.cpp:1166] [PG 0 (default_pg) Rank 6] Launching ProcessGroupNCCL abort asynchrounously.
I0527 15:22:16.721518 58628 ProcessGroupNCCL.cpp:1166] [PG 0 (default_pg) Rank 4] Launching ProcessGroupNCCL abort asynchrounously.
I0527 15:22:16.721678 58629 ProcessGroupNCCL.cpp:1166] [PG 0 (default_pg) Rank 5] Launching ProcessGroupNCCL abort asynchrounously.
I0527 15:22:16.721833 100246 ProcessGroupNCCL.cpp:1113] [PG 0 (default_pg) Rank 6] ProcessGroupNCCL destroying ncclComm_ 0x55879b0e7780 on CUDA device: 6
I0527 15:22:16.721848 100247 ProcessGroupNCCL.cpp:1113] [PG 0 (default_pg) Rank 4] ProcessGroupNCCL destroying ncclComm_ 0x55cec3d7ed80 on CUDA device: 4
I0527 15:22:16.721904 100246 NCCLUtils.hpp:364] Aborting ncclComm_ 0x55879b0e7780 with reason: No abort reason provided.
I0527 15:22:16.721918 100247 NCCLUtils.hpp:364] Aborting ncclComm_ 0x55cec3d7ed80 with reason: No abort reason provided.
I0527 15:22:16.721957 100248 ProcessGroupNCCL.cpp:1113] [PG 0 (default_pg) Rank 5] ProcessGroupNCCL destroying ncclComm_ 0x555b8d27b370 on CUDA device: 5
I0527 15:22:16.722023 100248 NCCLUtils.hpp:364] Aborting ncclComm_ 0x555b8d27b370 with reason: No abort reason provided.
I0527 15:22:16.721987 58631 ProcessGroupNCCL.cpp:1166] [PG 0 (default_pg) Rank 7] Launching ProcessGroupNCCL abort asynchrounously.
I0527 15:22:16.722262 100249 ProcessGroupNCCL.cpp:1113] [PG 0 (default_pg) Rank 7] ProcessGroupNCCL destroying ncclComm_ 0x5646eabe8fc0 on CUDA device: 7
I0527 15:22:16.722324 100249 NCCLUtils.hpp:364] Aborting ncclComm_ 0x5646eabe8fc0 with reason: No abort reason provided.
[INFO|configuration_utils.py:670] 2025-05-27 15:22:16,738 >> loading configuration file /root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct/config.json
[INFO|configuration_utils.py:739] 2025-05-27 15:22:16,738 >> Model config LlamaConfig {
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128009,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.0",
  "use_cache": true,
  "vocab_size": 128256
}

[INFO|tokenization_utils_base.py:2649] 2025-05-27 15:22:17,076 >> tokenizer config file saved in saves/llama3-8b/lora/sft/tokenizer_config.json
[INFO|tokenization_utils_base.py:2658] 2025-05-27 15:22:17,076 >> Special tokens file saved in saves/llama3-8b/lora/sft/special_tokens_map.json
I0527 15:22:17.231048 100243 ProcessGroupNCCL.cpp:1132] [PG 0 (default_pg) Rank 3] ProcessGroupNCCL destroyed  communicator on CUDA device: 3 with stream: 3
I0527 15:22:17.231141 58627 ProcessGroupNCCL.cpp:1067] [PG 0 (default_pg) Rank 3] future is successfully executed for: ProcessGroup abort
I0527 15:22:17.231158 58627 ProcessGroupNCCL.cpp:1172] [PG 0 (default_pg) Rank 3] ProcessGroupNCCL aborts successfully.
I0527 15:22:17.236095 100249 ProcessGroupNCCL.cpp:1132] [PG 0 (default_pg) Rank 7] ProcessGroupNCCL destroyed  communicator on CUDA device: 7 with stream: 3
I0527 15:22:17.236198 58631 ProcessGroupNCCL.cpp:1067] [PG 0 (default_pg) Rank 7] future is successfully executed for: ProcessGroup abort
I0527 15:22:17.236215 58631 ProcessGroupNCCL.cpp:1172] [PG 0 (default_pg) Rank 7] ProcessGroupNCCL aborts successfully.
I0527 15:22:17.236606 100245 ProcessGroupNCCL.cpp:1132] [PG 0 (default_pg) Rank 2] ProcessGroupNCCL destroyed  communicator on CUDA device: 2 with stream: 3
I0527 15:22:17.236690 58626 ProcessGroupNCCL.cpp:1067] [PG 0 (default_pg) Rank 2] future is successfully executed for: ProcessGroup abort
I0527 15:22:17.236704 58626 ProcessGroupNCCL.cpp:1172] [PG 0 (default_pg) Rank 2] ProcessGroupNCCL aborts successfully.
I0527 15:22:17.237890 100244 ProcessGroupNCCL.cpp:1132] [PG 0 (default_pg) Rank 1] ProcessGroupNCCL destroyed  communicator on CUDA device: 1 with stream: 3
I0527 15:22:17.237986 58625 ProcessGroupNCCL.cpp:1067] [PG 0 (default_pg) Rank 1] future is successfully executed for: ProcessGroup abort
I0527 15:22:17.238003 58625 ProcessGroupNCCL.cpp:1172] [PG 0 (default_pg) Rank 1] ProcessGroupNCCL aborts successfully.
I0527 15:22:17.238159 100247 ProcessGroupNCCL.cpp:1132] [PG 0 (default_pg) Rank 4] ProcessGroupNCCL destroyed  communicator on CUDA device: 4 with stream: 3
I0527 15:22:17.238257 58628 ProcessGroupNCCL.cpp:1067] [PG 0 (default_pg) Rank 4] future is successfully executed for: ProcessGroup abort
I0527 15:22:17.238276 58628 ProcessGroupNCCL.cpp:1172] [PG 0 (default_pg) Rank 4] ProcessGroupNCCL aborts successfully.
I0527 15:22:17.238512 100248 ProcessGroupNCCL.cpp:1132] [PG 0 (default_pg) Rank 5] ProcessGroupNCCL destroyed  communicator on CUDA device: 5 with stream: 3
I0527 15:22:17.238644 58629 ProcessGroupNCCL.cpp:1067] [PG 0 (default_pg) Rank 5] future is successfully executed for: ProcessGroup abort
I0527 15:22:17.238694 58629 ProcessGroupNCCL.cpp:1172] [PG 0 (default_pg) Rank 5] ProcessGroupNCCL aborts successfully.
I0527 15:22:17.238704 100246 ProcessGroupNCCL.cpp:1132] [PG 0 (default_pg) Rank 6] ProcessGroupNCCL destroyed  communicator on CUDA device: 6 with stream: 3
I0527 15:22:17.238816 58630 ProcessGroupNCCL.cpp:1067] [PG 0 (default_pg) Rank 6] future is successfully executed for: ProcessGroup abort
I0527 15:22:17.238830 58630 ProcessGroupNCCL.cpp:1172] [PG 0 (default_pg) Rank 6] ProcessGroupNCCL aborts successfully.
***** train metrics *****
  epoch                    =     2.9781
  total_flos               = 22837125GF
  train_loss               =     1.0481
  train_runtime            = 0:04:56.32
  train_samples_per_second =     11.045
  train_steps_per_second   =      0.172
Figure saved at: saves/llama3-8b/lora/sft/training_loss.png
[WARNING|2025-05-27 15:22:17] llamafactory.extras.ploting:148 >> No metric eval_loss to plot.
[WARNING|2025-05-27 15:22:17] llamafactory.extras.ploting:148 >> No metric eval_accuracy to plot.
[INFO|modelcard.py:449] 2025-05-27 15:22:17,398 >> Dropping the following result as it does not have all the necessary fields:
{'task': {'name': 'Causal Language Modeling', 'type': 'text-generation'}}
I0527 15:22:17.408150 58624 ProcessGroupNCCL.cpp:1166] [PG 0 (default_pg) Rank 0] Launching ProcessGroupNCCL abort asynchrounously.
I0527 15:22:17.408450 100272 ProcessGroupNCCL.cpp:1113] [PG 0 (default_pg) Rank 0] ProcessGroupNCCL destroying ncclComm_ 0x55efff5bd950 on CUDA device: 0
I0527 15:22:17.408473 100272 NCCLUtils.hpp:364] Aborting ncclComm_ 0x55efff5bd950 with reason: No abort reason provided.
I0527 15:22:17.917073 100272 ProcessGroupNCCL.cpp:1132] [PG 0 (default_pg) Rank 0] ProcessGroupNCCL destroyed  communicator on CUDA device: 0 with stream: 3
I0527 15:22:17.917176 58624 ProcessGroupNCCL.cpp:1067] [PG 0 (default_pg) Rank 0] future is successfully executed for: ProcessGroup abort
I0527 15:22:17.917193 58624 ProcessGroupNCCL.cpp:1172] [PG 0 (default_pg) Rank 0] ProcessGroupNCCL aborts successfully.
```

正在使用 LlamaFactory 工具，在 8 个 GPU 上对 `Meta-Llama-3-8B-Instruct` 模型进行 LoRA (Low-Rank Adaptation) 方式的监督式微调 (SFT, Supervised Fine-Tuning)。训练数据包含 `identity` 和 `alpaca_en_demo` 两个数据集，训练过程持续 3 个 epoch。

------

按照微调流程的顺序，结合配置文件 (`llama3_lora_sft.yaml`) 和日志输出进行逐步解读。

**阶段一：命令执行与环境初始化 (日志时间: 15:16:40 - 15:16:51)**

1. **启动微调命令**:
   - **日志**: `(dcu_llm_fine) root@Ubuntu2204:~/AI-BOX/code/dcu/llama-factory# llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml`
   - **解释**: 你执行了 LlamaFactory 的命令行工具，指定了 `train` 动作，并传入了微调的配置文件 `examples/train_lora/llama3_lora_sft.yaml`。
2. **分布式环境探测与设置**:
   - 日志
     - `[2025-05-27 15:16:40,207] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)`
     - `[INFO|2025-05-27 15:16:43] llamafactory.cli:143 >> Initializing 8 distributed tasks at: 127.0.0.1:54447`
     - `W0527 15:16:45.699000 ... Setting OMP_NUM_THREADS environment variable for each process to be 1...`
   - **配置文件关联**: `ddp_timeout: 180000000` (虽然未直接在此处显示，但这是分布式训练（DDP）的一个参数配置，在此阶段会被 LlamaFactory/Accelerate 读取)。
   - **解释**: 系统自动检测并设置使用 CUDA 作为加速器。紧接着，LlamaFactory 初始化了8个分布式任务，这通常意味着你有8个GPU参与训练，并且它们都在本地 (`127.0.0.1`)。为了优化性能，OpenMP 的线程数被设置为1。
3. **NCCL (NVIDIA Collective Communications Library) 初始化**:
   - 日志
     - `I0527 15:16:51.665375 58626 ProcessGroupNCCL.cpp:869] [PG 0 Rank 2] ProcessGroupNCCL initialization options: size: 8, global rank: 2, TIMEOUT(ms): 180000000000...` (类似日志会为每个 rank/GPU 打印)
     - `[INFO|2025-05-27 15:16:51] llamafactory.hparams.parser:401 >> Process rank: 2, world size: 8, device: cuda:2, distributed training: True, compute dtype: torch.bfloat16` (类似日志会为每个 rank/GPU 打印)
   - **配置文件关联**: `bf16: true`
   - **解释**: 每个 **GPU 进程（rank 0 到 rank 7）开始初始化 NCCL，这是多 GPU 通信的基础**。日志显示了 `world size: 8` (总共8个GPU) 以及每个进程分配到的 `cuda`设备。重要的是，`compute dtype: torch.bfloat16` 反映了配置文件中 `bf16: true` 的设置，意味着后续计算将主要使用 bf16 半精度。
4. **DDP特定设置 (LoRA相关)**:
   - **日志**: `[INFO|2025-05-27 15:16:51] llamafactory.hparams.parser:143 >> Set ddp_find_unused_parameters to False in DDP training since LoRA is enabled.`
   - **配置文件关联**: `finetuning_type: lora`
   - **解释**: 由于你选择了 LoRA 微调 (`finetuning_type: lora`)，LlamaFactory 自动将 DDP 的 `find_unused_parameters` 设置为 `False`。这是一个针对 LoRA 训练的优化，因为 LoRA 只训练模型参数的一小部分，设置为 `False` 可以避免不必要的参数检查开销。

**阶段二：Tokenizer 与基础模型配置加载 (日志时间: 15:16:51 - 15:16:52)**

1. **加载 Tokenizer**:
   - 日志
     - `[INFO|tokenization_utils_base.py:2212] 2025-05-27 15:16:51,714 >> loading file tokenizer.json`
     - `[INFO|tokenization_utils_base.py:2212] 2025-05-27 15:16:51,714 >> loading file tokenizer.model`
     - ... (其他 tokenizer 相关文件)
   - **配置文件关联**: `model_name_or_path: /root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct`
   - **解释**: 系统从配置文件中 `model_name_or_path` 指定的路径加载 Llama 3 模型的 Tokenizer 文件。
2. **加载模型配置 (LlamaConfig)**:
   - 日志
     - `[INFO|configuration_utils.py:670] 2025-05-27 15:16:52,195 >> loading configuration file /root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct/config.json`
     - `[INFO|configuration_utils.py:739] 2025-05-27 15:16:52,196 >> Model config LlamaConfig { ... "torch_dtype": "bfloat16", ... }`
   - **配置文件关联**: `model_name_or_path`
   - **解释**: 加载基础模型的配置文件 (config.json)，其中包含了模型的架构信息如层数、隐藏单元数等。日志中打印的 `torch_dtype: "bfloat16"` 指的是该预训练模型本身推荐或保存时的精度。

**阶段三：数据集加载与预处理 (日志时间: 15:16:52 - 15:17:01)**

1. **模板与特殊 Token 处理**:
   - 日志
     - `[INFO|tokenization_utils_base.py:2478] 2025-05-27 15:16:52,194 >> Special tokens have been added in the vocabulary...`
     - `[INFO|2025-05-27 15:16:52] llamafactory.data.template:143 >> Add pad token: <|eot_id|>`
     - `[INFO|2025-05-27 15:16:52] llamafactory.data.template:143 >> Add <|eot_id|>,<|eom_id|> to stop words.`
   - **配置文件关联**: `template: llama3`
   - **解释**: 根据配置文件中的 `template: llama3`，LlamaFactory 为 Llama 3 模型配置了相应的特殊 token，如将 `<|eot_id|>` 同时用作 pad token，并将特定的 token 加入到停止词列表中，这对于后续的数据格式化和生成控制非常重要。
2. **加载数据集**:
   - 日志
     - `[INFO|2025-05-27 15:16:52] llamafactory.data.loader:143 >> Loading dataset identity.json...`
     - `[INFO|2025-05-27 15:16:56] llamafactory.data.loader:143 >> Loading dataset alpaca_en_demo.json...`
   - **配置文件关联**: `dataset: identity,alpaca_en_demo`
   - **解释**: 加载配置文件中 `dataset` 字段指定的两个数据集。
3. **数据格式化与 Tokenization**:
   - 日志
     - `Converting format of dataset (num_proc=16): 100%|...| 91/91 ...` (identity.json)
     - `Converting format of dataset (num_proc=16): 100%|...| 1000/1000 ...` (alpaca_en_demo.json)
     - `Running tokenizer on dataset (num_proc=16): 100%|...| 1091/1091 ...`
     - `training example: input_ids: ... inputs: <|begin_of_text|>... labels: ...`
   - 配置文件关联
     - `preprocessing_num_workers: 16`
     - `max_samples: 1000` (可能作用于 alpaca_en_demo，如果其原始大小超过1000)
     - `cutoff_len: 2048` (Tokenized 序列的最大长度)
     - `overwrite_cache: true` (如果存在旧的预处理缓存，则覆盖)
   - **解释**: 使用 `preprocessing_num_workers: 16` 指定的16个进程并行处理数据。首先将原始数据转换为 LlamaFactory 内部格式，然后根据 `template: llama3` 对数据进行 tokenize，生成 `input_ids` 和 `labels`。`max_samples` 限制了单个数据集的最大样本数，`cutoff_len` 保证了输入模型的序列不会过长。最后展示了一个处理完毕的训练样本。

**阶段四：基础模型权重加载与 LoRA 适配器设置 (日志时间: 15:17:01 - 15:17:16)**

1. **加载模型权重**:
   - 日志
     - `[INFO|2025-05-27 15:17:01] llamafactory.model.model_utils.kv_cache:143 >> KV cache is disabled during training.`
     - `[INFO|modeling_utils.py:3723] 2025-05-27 15:17:01,797 >> loading weights file /root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct/model.safetensors.index.json`
     - `[INFO|modeling_utils.py:1622] 2025-05-27 15:17:01,797 >> Instantiating LlamaForCausalLM model under default dtype torch.bfloat16.`
     - `Loading checkpoint shards: 100%|...| 4/4 ...`
     - `[INFO|modeling_utils.py:4568] ... All model checkpoint weights were used ...`
   - **配置文件关联**: `model_name_or_path`, `bf16: true`
   - **解释**: KV cache 在训练时被禁用。系统从 `model_name_or_path` 指定的路径加载模型的权重文件 (分片形式的 .safetensors)。模型实例化的默认数据类型是 `torch.bfloat16`，与 `bf16: true` 配置一致。
2. **SDPA (Scaled Dot Product Attention) 相关警告**:
   - **日志**: `WARNING|logging.py:328] ... Using the SDPA attention implementation on multi-gpu setup with ROCM may lead to performance issues ... Disabling it...`
   - **解释**: 这个重要的警告表明，在你的 AMD GPU (DCU，使用 ROCm) 环境下，PyTorch 的 SDPA 实现（尤其是 FlashAttention 后端）可能存在性能问题，因此被禁用或回退到其他实现。这与后面训练循环中反复出现的 `UserWarning: 1Torch was not compiled with memory efficient attention.` 直接相关。
3. **LoRA 配置与参数准备**:
   - 日志
     - `[INFO|2025-05-27 15:17:16] llamafactory.model.model_utils.checkpointing:143 >> Gradient checkpointing enabled.`
     - `[INFO|2025-05-27 15:17:16] llamafactory.model.model_utils.attention:143 >> Using torch SDPA for faster training and inference.` (尽管有警告，仍在尝试使用)
     - `[INFO|2025-05-27 15:17:16] llamafactory.model.adapter:143 >> Upcasting trainable params to float32.`
     - `[INFO|2025-05-27 15:17:16] llamafactory.model.adapter:143 >> Fine-tuning method: LoRA`
     - `[INFO|2025-05-27 15:17:16] llamafactory.model.model_utils.misc:143 >> Found linear modules: v_proj,q_proj,k_proj,down_proj,o_proj,gate_proj,up_proj`
     - `[INFO|2025-05-27 15:17:16] llamafactory.model.loader:143 >> trainable params: 20,971,520 || all params: 8,051,232,768 || trainable%: 0.2605`
   - **配置文件关联**: `finetuning_type: lora`, `lora_rank: 8`, `lora_target: all`
   - **解释**: 启用了梯度检查点以节省显存。可训练参数（主要是 LoRA 权重）被提升到 float32 以保证更新的稳定性。日志确认了微调方法是 LoRA，并根据 `lora_target: all` 找到了模型中所有符合条件的线性层以应用 LoRA 适配器。`lora_rank: 8` 决定了适配器的大小，最终的可训练参数量 (约2097万) 和占比 (0.2605%) 被计算并打印出来。

**阶段五：Trainer 初始化与训练开始 (日志时间: 15:17:16 - 15:17:20)**

1. **Trainer 设置**:
   - 日志
     - `[INFO|trainer.py:667] 2025-05-27 15:17:16,639 >> Using auto half precision backend`
     - `[WARNING|2025-05-27 15:17:16] llamafactory.train.callbacks:154 >> Previous trainer log in this folder will be deleted.`
   - **配置文件关联**: `overwrite_output_dir: true` (来自 `### output` 部分)
   - **解释**: Hugging Face Trainer 自动配置了半精度训练。由于 `overwrite_output_dir: true`，旧的训练日志（如果存在）会被删除。
2. **训练参数摘要**:
   - 日志
     - `[INFO|trainer.py:2243] 2025-05-27 15:17:20,382 >> ***** Running training *****`
     - `[INFO|trainer.py:2244] ... Num examples = 1,091`
     - `[INFO|trainer.py:2245] ... Num Epochs = 3`
     - `[INFO|trainer.py:2246] ... Instantaneous batch size per device = 1`
     - `[INFO|trainer.py:2249] ... Total train batch size (w. parallel, distributed & accumulation) = 64`
     - `[INFO|trainer.py:2250] ... Gradient Accumulation steps = 8`
     - `[INFO|trainer.py:2251] ... Total optimization steps = 51`
     - `[INFO|trainer.py:2252] ... Number of trainable parameters = 20,971,520`
   - 配置文件关联
     - `num_train_epochs: 3.0`
     - `per_device_train_batch_size: 1`
     - `gradient_accumulation_steps: 8`
   - **解释**: 正式开始训练前，Trainer 打印了关键的训练参数摘要。这些参数大部分直接来自你的配置文件，例如 epoch 数、每设备批大小、梯度累积步数。总训练批大小 (64) 和总优化步数 (51) 是基于这些配置和 GPU 数量 (8) 计算得出的。

**阶段六：训练循环 (日志时间: 15:17:20 - 15:22:15)**

1. **迭代训练与指标记录**:
   - 日志
     - `0%| | 0/51 [00:00<?, ?it/s]` (进度条开始)
     - `{'loss': 1.4091, 'grad_norm': 1.0385..., 'learning_rate': 9.8063...e-05, 'epoch': 0.58}` (周期性打印)
     - ... (后续的 loss, grad_norm, learning_rate, epoch 更新)
   - 配置文件关联: (来自 `### train`和 `### output`部分)
     - `logging_steps: 10`
     - `learning_rate: 1.0e-4`
     - `lr_scheduler_type: cosine`
     - `warmup_ratio: 0.1`
   - **解释**: 训练开始迭代。每 `logging_steps: 10` 个瞬时批次（注意，不是优化步）会打印一次训练指标，包括损失 (loss)、梯度范数 (grad_norm)、当前学习率 (learning_rate) 和所处轮数 (epoch)。学习率的变化遵循了配置文件中定义的初始值 (`1.0e-4`)、预热比例 (`warmup_ratio: 0.1`) 和余弦衰减调度器 (`lr_scheduler_type: cosine`)。
2. **内存高效注意力警告 (反复出现)**:
   - **日志**: `/root/anaconda3/envs/dcu_llm_fine/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py:655: UserWarning: 1Torch was not compiled with memory efficient attention. (Triggered internally at /home/pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:627.)`
   - **解释**: 此警告在训练过程中反复出现，再次确认了在你的 AMD DCU (HIP是AMD的CUDA等效接口) 环境下，PyTorch 未能使用最优的内存高效注意力机制。这可能会对训练速度和显存效率造成一定影响。

**阶段七：训练完成与模型保存 (日志时间: 15:22:15 - 15:22:17)**

1. **训练结束与最终检查点保存**:
   - 日志
     - `100%|...| 51/51 [04:54<00:00, 4.38s/it]` (训练完成)
     - `[INFO|trainer.py:3705] 2025-05-27 15:22:15,098 >> Saving model checkpoint to saves/llama3-8b/lora/sft/checkpoint-51`
     - `[INFO|tokenization_utils_base.py:2649] ... tokenizer config file saved ...`
   - **配置文件关联**: `output_dir: saves/llama3-8b/lora/sft` (来自 `### output`)
   - **解释**: 51个优化步全部完成。系统将最后一个检查点（包含LoRA权重和训练器状态，因为 `save_only_model: false`）保存到 `output_dir` 下的 `checkpoint-51` 子目录。Tokenizer 配置也被保存。
2. **最终训练指标与模型保存**:
   - 日志
     - `Training completed. ...`
     - `{'train_runtime': 296.3281, 'train_samples_per_second': 11.045, ..., 'train_loss': 1.0480..., 'epoch': 2.98}`
     - `[INFO|trainer.py:3705] 2025-05-27 15:22:16,719 >> Saving model checkpoint to saves/llama3-8b/lora/sft`
   - **配置文件关联**: `output_dir: saves/llama3-8b/lora/sft`
   - **解释**: 打印最终的训练统计数据，包括总运行时长、吞吐量、平均损失等。然后，模型（主要是LoRA适配器）被保存到配置文件中 `output_dir` 指定的顶层目录 `saves/llama3-8b/lora/sft`。
3. **NCCL 清理与绘图**:
   - 日志
     - `I0527 15:22:16.720973 ... Launching ProcessGroupNCCL abort asynchrounously.` (各 rank 开始清理 NCCL)
     - `***** train metrics ***** ...` (打印最终训练指标摘要)
     - `Figure saved at: saves/llama3-8b/lora/sft/training_loss.png`
     - `[WARNING|2025-05-27 15:22:17] llamafactory.extras.ploting:148 >> No metric eval_loss to plot.`
     - `[WARNING|2025-05-27 15:22:17] llamafactory.extras.ploting:148 >> No metric eval_accuracy to plot.`
   - 配置文件关联
     - `plot_loss: true` (来自 `### output`)
     - 评估相关配置 (`### eval`) 被注释掉。
   - **解释**: 分布式环境的 NCCL 通信器被正确关闭和清理。由于 `plot_loss: true`，训练损失曲线图被绘制并保存。因为评估相关的配置项在 `llama3_lora_sft.yaml` 中被注释掉了，所以日志提示没有评估损失和评估准确率可以绘制。

------

**核心微调流程总结:**

1. **环境设置**: 初始化分布式环境，每个 GPU (共8个) 作为一个独立的进程。
2. **资源加载**: 加载 Llama 3 8B Instruct 模型的配置、tokenizer 和预训练权重。
3. 数据准备
   - 加载 `identity` 和 `alpaca_en_demo` 数据集。
   - 按照 Llama 3 的对话模板对数据进行格式化和 tokenize。
   - 将输入序列截断或填充到 `cutoff_len: 2048`。
   - 准备 `input_ids` 和 `label_ids`，其中 `label_ids` 中对应 prompt 的部分被设置为 -100 以在损失计算中忽略。
4. LoRA适配器注入
   - 在模型的所有线性层 (`lora_target: all`) 上添加 LoRA 适配器，秩为 `lora_rank: 8`。
   - 冻结原始模型的大部分参数，只使 LoRA 相关的参数（约2097万）可训练。
5. 分布式训练
   - 使用 `bfloat16` 混合精度和梯度检查点以优化显存和速度。
   - 数据被分配到 8 个 GPU上。
   - 每个 GPU 以 `per_device_train_batch_size: 1` 进行前向和反向传播。
   - 梯度累积 `gradient_accumulation_steps: 8` 次后，进行一次模型参数更新。有效批次大小为 64。
   - 学习率按照余弦衰减策略调整，并有 10% 的预热。
   - 训练进行 3 个 epoch，总共 51 个优化步骤。
   - **关键瓶颈/警告**: 由于在 AMD DCU 上运行，PyTorch 未能使用内存高效的注意力实现 (FlashAttention)，这可能影响了训练效率。
6. 模型保存
   - 训练完成后，保存 LoRA 适配器权重和配置文件到指定的 `output_dir`。
   - 同时保存训练日志和损失曲线图。

------

**针对 AMD DCU 环境的建议 (如果适用且希望优化):**

- **ROCm 和 PyTorch 版本**: 确保你使用的 ROCm 版本与 PyTorch 版本兼容，并且 PyTorch 是针对你的特定 AMD GPU 架构编译或优化的。有时，特定组合才能最好地支持 FlashAttention 或类似的内存高效注意力机制。
- **Transformers 版本**: 保持 Transformers 库更新到最新，它可能包含对 AMD GPU 更好的支持。
- **FlashAttention for ROCm**: 社区有一些努力将 FlashAttention 移植或适配到 ROCm。可以关注相关项目（如 FlashAttention 官方仓库或 AMD 的 MIOpen/rocBLAS 等）的进展，看是否有可用的、针对你硬件的优化库。
- **LlamaFactory/Accelerate 配置**: 检查 LlamaFactory 或其依赖的 Accelerate 库是否有针对 AMD GPU 的特定配置选项，可能需要手动指定某些后端。

这个日志提供了非常丰富的信息，清晰地展示了 LlamaFactory 在多 GPU 环境下进行 LoRA 微调的完整过程。最值得注意的观察点是关于在 AMD GPU 上 SDPA 和内存高效注意力机制的警告。

### 模型微调后推理测试

```bash
(dcu_llm_fine) root@Ubuntu2204:~/AI-BOX/code/dcu/llama-factory# llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
[2025-05-27 17:30:16,582] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[INFO|tokenization_utils_base.py:2212] 2025-05-27 17:30:20,049 >> loading file tokenizer.json
[INFO|tokenization_utils_base.py:2212] 2025-05-27 17:30:20,049 >> loading file tokenizer.model
[INFO|tokenization_utils_base.py:2212] 2025-05-27 17:30:20,049 >> loading file added_tokens.json
[INFO|tokenization_utils_base.py:2212] 2025-05-27 17:30:20,049 >> loading file special_tokens_map.json
[INFO|tokenization_utils_base.py:2212] 2025-05-27 17:30:20,049 >> loading file tokenizer_config.json
[INFO|tokenization_utils_base.py:2478] 2025-05-27 17:30:20,484 >> Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
[INFO|configuration_utils.py:670] 2025-05-27 17:30:20,486 >> loading configuration file /root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct/config.json
[INFO|configuration_utils.py:739] 2025-05-27 17:30:20,487 >> Model config LlamaConfig {
  "_name_or_path": "/root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128009,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.0",
  "use_cache": true,
  "vocab_size": 128256
}

[INFO|tokenization_utils_base.py:2212] 2025-05-27 17:30:20,488 >> loading file tokenizer.json
[INFO|tokenization_utils_base.py:2212] 2025-05-27 17:30:20,488 >> loading file tokenizer.model
[INFO|tokenization_utils_base.py:2212] 2025-05-27 17:30:20,488 >> loading file added_tokens.json
[INFO|tokenization_utils_base.py:2212] 2025-05-27 17:30:20,488 >> loading file special_tokens_map.json
[INFO|tokenization_utils_base.py:2212] 2025-05-27 17:30:20,488 >> loading file tokenizer_config.json
[INFO|tokenization_utils_base.py:2478] 2025-05-27 17:30:20,887 >> Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
[INFO|2025-05-27 17:30:20] llamafactory.data.template:143 >> Add pad token: <|eot_id|>
[INFO|2025-05-27 17:30:20] llamafactory.data.template:143 >> Add <|eot_id|>,<|eom_id|> to stop words.
[WARNING|2025-05-27 17:30:20] llamafactory.data.template:148 >> New tokens have been added, make sure `resize_vocab` is True.
[INFO|configuration_utils.py:670] 2025-05-27 17:30:20,904 >> loading configuration file /root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct/config.json
[INFO|configuration_utils.py:739] 2025-05-27 17:30:20,905 >> Model config LlamaConfig {
  "_name_or_path": "/root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128009,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.0",
  "use_cache": true,
  "vocab_size": 128256
}

[INFO|2025-05-27 17:30:20] llamafactory.model.model_utils.kv_cache:143 >> KV cache is enabled for faster generation.
[INFO|modeling_utils.py:3723] 2025-05-27 17:30:20,938 >> loading weights file /root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct/model.safetensors.index.json
[INFO|modeling_utils.py:1622] 2025-05-27 17:30:20,938 >> Instantiating LlamaForCausalLM model under default dtype torch.bfloat16.
[WARNING|logging.py:328] 2025-05-27 17:30:20,938 >> Using the `SDPA` attention implementation on multi-gpu setup with ROCM may lead to performance issues due to the FA backend. Disabling it to use alternative backends.
[INFO|configuration_utils.py:1099] 2025-05-27 17:30:20,939 >> Generate config GenerationConfig {
  "bos_token_id": 128000,
  "eos_token_id": 128009
}

Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:06<00:00,  1.69s/it]
[INFO|modeling_utils.py:4568] 2025-05-27 17:30:28,047 >> All model checkpoint weights were used when initializing LlamaForCausalLM.

[INFO|modeling_utils.py:4576] 2025-05-27 17:30:28,047 >> All the weights of LlamaForCausalLM were initialized from the model checkpoint at /root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct.
If your task is similar to the task the model of the checkpoint was trained on, you can already use LlamaForCausalLM for predictions without further training.
[INFO|configuration_utils.py:1052] 2025-05-27 17:30:28,050 >> loading configuration file /root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3-8B-Instruct/generation_config.json
[INFO|configuration_utils.py:1099] 2025-05-27 17:30:28,051 >> Generate config GenerationConfig {
  "bos_token_id": 128000,
  "do_sample": true,
  "eos_token_id": [
    128001,
    128009
  ],
  "max_length": 4096,
  "temperature": 0.6,
  "top_p": 0.9
}

[INFO|2025-05-27 17:30:28] llamafactory.model.model_utils.attention:143 >> Using torch SDPA for faster training and inference.
[INFO|2025-05-27 17:31:18] llamafactory.model.adapter:143 >> Merged 1 adapter(s).
[INFO|2025-05-27 17:31:18] llamafactory.model.adapter:143 >> Loaded adapter(s): saves/llama3-8b/lora/sft
[INFO|2025-05-27 17:31:18] llamafactory.model.loader:143 >> all params: 8,030,261,248
Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.

User: 介绍中科曙光
Assistant: /root/anaconda3/envs/dcu_llm_fine/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py:655: UserWarning: 1Torch was not compiled with memory efficient attention. (Triggered internally at /home/pytorch/aten/src/ATen/native/transformers/hip/sdp_utils.cpp:627.)
  attn_output = torch.nn.functional.scaled_dot_product_attention(
[WARNING|logging.py:328] 2025-05-27 17:31:32,235 >> Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)
中科曙光是中国的光学仪器生产企业，位于中国江苏省苏州市。公司于1992年建立，主要从事光学仪器、激光技术、检测技术等领域的研发和生产。中科曙光的产品包括激光器、光学仪器、检测设备、光学材料等多种类型，广泛应用于科研、生产、教育和其他领域。

User: Who are you
Assistant: I am {{name}}, an AI assistant trained by {{author}}. I am here to assist you with any questions or tasks you may have.

User: 你是谁
Assistant: 我是 {{name}}，由 {{author}} 训练的 AI 助手。我旨在为您提供帮助，回答问题和完成任务。

User: 你有名字吗
Assistant: 是的，我是 {{name}}，由 {{author}} 训练的 AI 助手。

User: 
```



## 常见问题

------

### 1. 微调 Llama3-8B 启动异常：`NameError: name 'amdsmi' is not defined`

**问题 (Question):**

使用 LlamaFactory 微调 `Llama3-8B-Instruct` 模型时，遇到了 `NameError: name 'amdsmi' is not defined` 的错误。

我的环境信息如下：

- 模型：`Llama3-8B-Instruct`
- 框架：`llamafactory`
- 命令：`llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml`
- 操作系统：Ubuntu 22.04
- DCU：K100-AI
- DTK：25.04

错误发生在尝试初始化 `amdsmi` 时。这是什么原因导致的，以及你是如何解决的？

------

**错误信息 (Error Message):**

主要的错误栈信息如下：

```bash
Traceback (most recent call last):
  File "/root/anaconda3/envs/dcu_llm_fine/lib/python3.10/site-packages/torch/cuda/__init__.py", line 642, in _raw_device_count_amdsmi
    amdsmi.amdsmi_init()
NameError: name 'amdsmi' is not defined
```

该错误发生在 PyTorch 的 `torch/cuda/__init__.py` 文件中的 `_raw_device_count_amdsmi` 函数内部，当代码尝试调用 `amdsmi.amdsmi_init()` 时。

------

**原因分析 (Cause Analysis):**

这个 `NameError` 表明，当 Python 解释器执行到 `amdsmi.amdsmi_init()` 这一行时，它找不到名为 `amdsmi` 的变量或模块的定义。根本原因通常是 `amdsmi` 模块没有被成功导入到当前代码的执行作用域中。

具体到你的情况和提供的图片信息，可能的原因包括：

1. amdsmi Python 包缺失或未正确安装:

   在你的 Conda 环境 dcu_llm_fine 中，可能没有安装 amdsmi 相关的 Python 包。PyTorch 在其 ROCm (HIP) 后端会尝试使用 AMDSMI (AMD System Management Interface) 来获取 AMD GPU 的信息。如果导入 amdsmi 失败（即使这个导入操作被一个 try-except ModuleNotFoundError 块包裹且静默处理了），后续代码若直接尝试使用 amdsmi 变量，就会引发 NameError。

2. **PyTorch 内部逻辑缺陷**:

   > /root/anaconda3/envs/dcu_llm_fine/lib/python3.10/site-packages/torch/cuda/__init__.py

   ```python
   try:
       # 第57行
       try:
           # 第58行：尝试导入 pynvml 库
           # pynvml 是用于与 NVIDIA GPU 通信的 Python 库 (NVIDIA Management Library)
           import pynvml # type: ignore[import]  
           # 第59行：如果导入成功，则执行下面的代码
           # 第60行：设置一个名为 _HAS_PYNVML 的布尔型全局变量为 True，表示 pynvml 库可用
           _HAS_PYNVML = True
       # 第61行：如果导入 pynvml 失败，并且错误类型是 ModuleNotFoundError (即找不到该模块)
       except ModuleNotFoundError:
           # 第62行：则什么也不做 (pass)
           pass # pynvml 未找到
   
       # 第63行
       try:
           # 第64行：尝试导入 amdsmi 库
           # amdsmi 是用于与 AMD GPU 通信的 Python 库 (AMD System Management Interface)
           import amdsmi # type: ignore[import]
           # 第65行：如果导入成功，则执行下面的代码
           # 第66行：设置一个名为 _HAS_PYNVML 的布尔型全局变量为 True，表示 amdsmi 库可用
           # 注意：这里可能是一个笔误 (typo)。通常情况下，这里应该设置一个与 amdsmi 相关的标志，
           # 例如 _HAS_AMDSMI = True。将它也设置为 _HAS_PYNVML 可能会导致后续逻辑判断错误。
           _HAS_PYNVML = True  # <--- 潜在的笔误，可能应该是 _HAS_AMDSMI = True
       # 第67行：如果导入 amdsmi 失败，并且错误类型是 ModuleNotFoundError
       except ModuleNotFoundError:
           _PYNVML_ERR = err  # sometimes a lib is installed but the import fails for some other reason, so we log the error for later
   ```

   - 可以看到，PyTorch 在尝试导入 `amdsmi` 模块后，错误地将 `_HAS_PYNVML` 标志设置为 `True`（`_HAS_PYNVML = True`），而不是一个专门针对 `amdsmi` 可用性的标志（例如，本应是 `_HAS_AMDSMI = True`）。这是一个明显的笔误。

     

   - 在出错的 `_raw_device_count_amdsmi` 函数中，存在一个条件检查 `if not _HAS_PYNVML:`。这个检查本身可能就是不恰当的，因为它似乎是基于 PYNVML (NVIDIA 的管理库) 的可用性来决定是否执行 AMDSMI (AMD 的管理库) 相关的代码。

   - 如果 `_HAS_PYNVML` 标志由于某种原因（例如，系统中安装了 PYNVML，或者由于上述笔误在导入 `amdsmi` 后被错误地设为 `True`）为 `True`，那么 `if not _HAS_PYNVML:` 这个条件就会为假，代码会继续执行到 `amdsmi.amdsmi_init()`。但此时，如果 `amdsmi` 模块之前并未被成功导入并加载到全局命名空间（例如，因为 `amdsmi` 包未安装），那么 `NameError` 就会发生。

------

**解决方案 :**

通过修改 PyTorch 的源代码文件 `/root/anaconda3/envs/dcu_llm_fine/lib/python3.10/site-packages/torch/cuda/__init__.py` 中的 `_raw_device_count_amdsmi` 函数，解决了这个问题。

具体的修改是在 `try: amdsmi.amdsmi_init()` 这一行 **之前** 添加了一行 `return -1`。

修改后的相关代码片段如下：

```Python
def _raw_device_count_amdsmi() -> int:
    if not _HAS_PYNVML: # PyTorch 原始代码中的检查
        return -1
    # 你添加的代码行：
    return -1
    try:
        amdsmi.amdsmi_init()
    except amdsmi.AmdSmiException as e:
        warnings.warn(f"Can't initialize amdsmi - Error code: {e.err_code}")
        return -1
    socket_handles = amdsmi.amdsmi_get_processor_handles()
    return len(socket_handles)
```

这个改动使得 `_raw_device_count_amdsmi` 函数在尝试执行任何 `amdsmi` 初始化或调用之前就直接返回 `-1`。在 PyTorch 中，类似的设备计数函数返回 `-1` 通常表示无法通过该特定方法获取设备数量。

------

**进一步的建议与思考:**

虽然修改使得程序能够继续运行，避免了报错，但它实际上是绕过了通过 AMDSMI 获取 AMD GPU 信息的步骤。如果返回 `-1` 导致 PyTorch 或上层应用（如 LlamaFactory/VLLM）无法正确识别或利用你的 K100-AI DCU，可能会影响训练性能或导致其他潜在问题。

## 参考资料

- [README_zh]()
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)


# 附录



