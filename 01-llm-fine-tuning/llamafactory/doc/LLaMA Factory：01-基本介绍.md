# LLaMA Factory：01-基本介绍

## 简介

LLama-factory 是一款整合了主流的各种高效训练微调技术，适配市场主流开源模型，而形成的一个功能丰富、适配性好的训练框架。LLama-factory 提供了多个高层次抽象的调用接口，包含多阶段训练、推理测试、benchmark 评测、API Server 等，使开发者开箱即用。

同时提供了基于 gradio 的网页版工作台，方便初学者迅速上手操作，开发出自己的第一个模型。

## 项目特色

- **多种模型**：LLaMA、LLaVA、Mistral、Mixtral-MoE、Qwen、Qwen2-VL、DeepSeek、Yi、Gemma、ChatGLM、Phi 等等。
- **集成方法**：（增量）预训练、（多模态）指令监督微调、奖励模型训练、PPO 训练、DPO 训练、KTO 训练、ORPO 训练等等。
- **多种精度**：16 比特全参数微调、冻结微调、LoRA 微调和基于 AQLM/AWQ/GPTQ/LLM.int8/HQQ/EETQ 的 2/3/4/5/6/8 比特 QLoRA 微调。
- **先进算法**：[GaLore](https://github.com/jiaweizzhao/GaLore)、[BAdam](https://github.com/Ledzy/BAdam)、[APOLLO](https://github.com/zhuhanqing/APOLLO)、[Adam-mini](https://github.com/zyushun/Adam-mini)、[Muon](https://github.com/KellerJordan/Muon)、DoRA、LongLoRA、LLaMA Pro、Mixture-of-Depths、LoRA+、LoftQ 和 PiSSA。
- **实用技巧**：[FlashAttention-2](https://github.com/Dao-AILab/flash-attention)、[Unsloth](https://github.com/unslothai/unsloth)、[Liger Kernel](https://github.com/linkedin/Liger-Kernel)、RoPE scaling、NEFTune 和 rsLoRA。
- **广泛任务**：多轮对话、工具调用、图像理解、视觉定位、视频识别和语音理解等等。
- **实验监控**：LlamaBoard、TensorBoard、Wandb、MLflow、[SwanLab](https://github.com/SwanHubX/SwanLab) 等等。
- **极速推理**：基于 [vLLM](https://github.com/vllm-project/vllm) 或 [SGLang](https://github.com/sgl-project/sglang) 的 OpenAI 风格 API、浏览器界面和命令行接口。

## 模型

| 模型名                                                       | 参数量                           | Template            |
| ------------------------------------------------------------ | -------------------------------- | ------------------- |
| [Baichuan 2](https://huggingface.co/baichuan-inc)            | 7B/13B                           | baichuan2           |
| [BLOOM/BLOOMZ](https://huggingface.co/bigscience)            | 560M/1.1B/1.7B/3B/7.1B/176B      | -                   |
| [ChatGLM3](https://huggingface.co/THUDM)                     | 6B                               | chatglm3            |
| [Command R](https://huggingface.co/CohereForAI)              | 35B/104B                         | cohere              |
| [DeepSeek (Code/MoE)](https://huggingface.co/deepseek-ai)    | 7B/16B/67B/236B                  | deepseek            |
| [DeepSeek 2.5/3](https://huggingface.co/deepseek-ai)         | 236B/671B                        | deepseek3           |
| [DeepSeek R1 (Distill)](https://huggingface.co/deepseek-ai)  | 1.5B/7B/8B/14B/32B/70B/671B      | deepseekr1          |
| [Falcon](https://huggingface.co/tiiuae)                      | 7B/11B/40B/180B                  | falcon              |
| [Gemma/Gemma 2/CodeGemma](https://huggingface.co/google)     | 2B/7B/9B/27B                     | gemma               |
| [Gemma 3](https://huggingface.co/google)                     | 1B/4B/12B/27B                    | gemma3/gemma (1B)   |
| [GLM-4/GLM-4-0414/GLM-Z1](https://huggingface.co/THUDM)      | 9B/32B                           | glm4/glmz1          |
| [GPT-2](https://huggingface.co/openai-community)             | 0.1B/0.4B/0.8B/1.5B              | -                   |
| [Granite 3.0-3.3](https://huggingface.co/ibm-granite)        | 1B/2B/3B/8B                      | granite3            |
| [Hunyuan](https://huggingface.co/tencent/)                   | 7B                               | hunyuan             |
| [Index](https://huggingface.co/IndexTeam)                    | 1.9B                             | index               |
| [InternLM 2-3](https://huggingface.co/internlm)              | 7B/8B/20B                        | intern2             |
| [InternVL 2.5-3](https://huggingface.co/OpenGVLab)           | 1B/2B/8B/14B/38B/78B             | intern_vl           |
| [Kimi-VL](https://huggingface.co/moonshotai)                 | 16B                              | kimi_vl             |
| [Llama](https://github.com/facebookresearch/llama)           | 7B/13B/33B/65B                   | -                   |
| [Llama 2](https://huggingface.co/meta-llama)                 | 7B/13B/70B                       | llama2              |
| [Llama 3-3.3](https://huggingface.co/meta-llama)             | 1B/3B/8B/70B                     | llama3              |
| [Llama 4](https://huggingface.co/meta-llama)                 | 109B/402B                        | llama4              |
| [Llama 3.2 Vision](https://huggingface.co/meta-llama)        | 11B/90B                          | mllama              |
| [LLaVA-1.5](https://huggingface.co/llava-hf)                 | 7B/13B                           | llava               |
| [LLaVA-NeXT](https://huggingface.co/llava-hf)                | 7B/8B/13B/34B/72B/110B           | llava_next          |
| [LLaVA-NeXT-Video](https://huggingface.co/llava-hf)          | 7B/34B                           | llava_next_video    |
| [MiMo](https://huggingface.co/XiaomiMiMo)                    | 7B                               | mimo                |
| [MiniCPM](https://huggingface.co/openbmb)                    | 1B/2B/4B                         | cpm/cpm3            |
| [MiniCPM-o-2.6/MiniCPM-V-2.6](https://huggingface.co/openbmb) | 8B                               | minicpm_o/minicpm_v |
| [Ministral/Mistral-Nemo](https://huggingface.co/mistralai)   | 8B/12B                           | ministral           |
| [Mistral/Mixtral](https://huggingface.co/mistralai)          | 7B/8x7B/8x22B                    | mistral             |
| [Mistral Small](https://huggingface.co/mistralai)            | 24B                              | mistral_small       |
| [OLMo](https://huggingface.co/allenai)                       | 1B/7B                            | -                   |
| [PaliGemma/PaliGemma2](https://huggingface.co/google)        | 3B/10B/28B                       | paligemma           |
| [Phi-1.5/Phi-2](https://huggingface.co/microsoft)            | 1.3B/2.7B                        | -                   |
| [Phi-3/Phi-3.5](https://huggingface.co/microsoft)            | 4B/14B                           | phi                 |
| [Phi-3-small](https://huggingface.co/microsoft)              | 7B                               | phi_small           |
| [Phi-4](https://huggingface.co/microsoft)                    | 14B                              | phi4                |
| [Pixtral](https://huggingface.co/mistralai)                  | 12B                              | pixtral             |
| [Qwen (1-2.5) (Code/Math/MoE/QwQ)](https://huggingface.co/Qwen) | 0.5B/1.5B/3B/7B/14B/32B/72B/110B | qwen                |
| [Qwen3 (MoE)](https://huggingface.co/Qwen)                   | 0.6B/1.7B/4B/8B/14B/32B/235B     | qwen3               |
| [Qwen2-Audio](https://huggingface.co/Qwen)                   | 7B                               | qwen2_audio         |
| [Qwen2.5-Omni](https://huggingface.co/Qwen)                  | 3B/7B                            | qwen2_omni          |
| [Qwen2-VL/Qwen2.5-VL/QVQ](https://huggingface.co/Qwen)       | 2B/3B/7B/32B/72B                 | qwen2_vl            |
| [Seed Coder](https://huggingface.co/ByteDance-Seed)          | 8B                               | seed_coder          |
| [Skywork o1](https://huggingface.co/Skywork)                 | 8B                               | skywork_o1          |
| [StarCoder 2](https://huggingface.co/bigcode)                | 3B/7B/15B                        | -                   |
| [TeleChat2](https://huggingface.co/Tele-AI)                  | 3B/7B/35B/115B                   | telechat2           |
| [XVERSE](https://huggingface.co/xverse)                      | 7B/13B/65B                       | xverse              |
| [Yi/Yi-1.5 (Code)](https://huggingface.co/01-ai)             | 1.5B/6B/9B/34B                   | yi                  |
| [Yi-VL](https://huggingface.co/01-ai)                        | 6B/34B                           | yi_vl               |
| [Yuan 2](https://huggingface.co/IEITYuan)                    | 2B/51B/102B                      | yuan                |

> [!Note]
>
> 对于所有“基座”（Base）模型，`template` 参数可以是 `default`, `alpaca`, `vicuna` 等任意值。但“对话”（Instruct/Chat）模型请务必使用**对应的模板**。
>
> 请务必在训练和推理时采用**完全一致**的模板。
>
> *：您需要从 main 分支安装 `transformers` 并使用 `DISABLE_VERSION_CHECK=1` 来跳过版本检查。
>
> **：您需要安装特定版本的 `transformers` 以使用该模型。
>
> 项目所支持模型的完整列表请参阅 [constants.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/extras/constants.py)。
>
> 您也可以在 [template.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/template.py) 中添加自己的对话模板。

## 训练方法

| 方法         | 全参数训练 | 部分参数训练 | LoRA | QLoRA |
| ------------ | ---------- | ------------ | ---- | ----- |
| 预训练       | ✅          | ✅            | ✅    | ✅     |
| 指令监督微调 | ✅          | ✅            | ✅    | ✅     |
| 奖励模型训练 | ✅          | ✅            | ✅    | ✅     |
| PPO 训练     | ✅          | ✅            | ✅    | ✅     |
| DPO 训练     | ✅          | ✅            | ✅    | ✅     |
| KTO 训练     | ✅          | ✅            | ✅    | ✅     |
| ORPO 训练    | ✅          | ✅            | ✅    | ✅     |
| SimPO 训练   | ✅          | ✅            | ✅    | ✅     |

> [!Tip]
>
> 有关 PPO 的实现细节，请参考[此博客](https://newfacade.github.io/notes-on-reinforcement-learning/17-ppo-trl.html)。

## 数据集



部分数据集的使用需要确认，我们推荐使用下述命令登录您的 Hugging Face 账户。

```
pip install --upgrade huggingface_hub
huggingface-cli login
```

## 软硬件依赖

| 必需项       | 至少   | 推荐   |
| ------------ | ------ | ------ |
| python       | 3.9    | 3.10   |
| torch        | 2.0.0  | 2.6.0  |
| torchvision  | 0.15.0 | 0.21.0 |
| transformers | 4.45.0 | 4.50.0 |
| datasets     | 2.16.0 | 3.2.0  |
| accelerate   | 0.34.0 | 1.2.1  |
| peft         | 0.14.0 | 0.15.1 |
| trl          | 0.8.6  | 0.9.6  |

| 可选项       | 至少   | 推荐   |
| ------------ | ------ | ------ |
| CUDA         | 11.6   | 12.2   |
| deepspeed    | 0.10.0 | 0.16.4 |
| bitsandbytes | 0.39.0 | 0.43.1 |
| vllm         | 0.4.3  | 0.8.2  |
| flash-attn   | 2.5.6  | 2.7.2  |

### 硬件依赖

\* *估算值*

| 方法                            | 精度 | 7B    | 14B   | 30B   | 70B    | `x`B    |
| ------------------------------- | ---- | ----- | ----- | ----- | ------ | ------- |
| Full (`bf16` or `fp16`)         | 32   | 120GB | 240GB | 600GB | 1200GB | `18x`GB |
| Full (`pure_bf16`)              | 16   | 60GB  | 120GB | 300GB | 600GB  | `8x`GB  |
| Freeze/LoRA/GaLore/APOLLO/BAdam | 16   | 16GB  | 32GB  | 64GB  | 160GB  | `2x`GB  |
| QLoRA                           | 8    | 10GB  | 20GB  | 40GB  | 80GB   | `x`GB   |
| QLoRA                           | 4    | 6GB   | 12GB  | 24GB  | 48GB   | `x/2`GB |
| QLoRA                           | 2    | 4GB   | 8GB   | 16GB  | 24GB   | `x/4`GB |