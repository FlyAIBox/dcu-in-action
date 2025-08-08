# LLaMA Factory：微调DeepSeek-R1-Distill-Qwen-7B模型实现新闻标题分类器

# 使用LLaMAFactory微调DeepSeek-R1蒸馏模型

[LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) 是一款开源低代码大模型微调框架，集成了业界广泛使用的微调技术，支持通过 Web UI 界面零代码微调大模型，目前已经成为开源社区内最受欢迎的微调框架之一，GitHub 星标超过 4 万。本教程将基于深度求索公司开源的 [DeepSeek-R1-Distill-Qwen-7B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) 模型（以 Qwen2.5-Math-7B 为基模型从 DeepSeek-R1 蒸馏得到），介绍如何使用 PAI 平台及 LLaMA Factory 训练框架微调得到新闻标题分类器：给定新闻的类别范围，通过自然语言触发新闻标题分类的功能，并以特定的格式进行返回。具体的要求为：

- 给定新闻种类，根据新闻标题对新闻进行分类。
- 输入为新闻标题，输出为新闻所属的种类，不需要额外解释，但保留思维链的内容；在输入的新闻标题前加上“新闻分类：”等类似文本来标明询问目的。

## 运行环境要求

- GPU推荐使用24GB显存的A10（`ecs.gn7i-c8g1.2xlarge`）或更高配置
- 镜像选择DSW官方镜像 `modelscope:1.21.0-pytorch2.4.0-gpu-py310-cu124-ubuntu22.04`

## 1. 安装LLaMA Factory

首先，拉取LLaMA-Factory项目到DSW实例。

```
!git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
```

接着，我们安装LLaMA-Factory依赖环境。

```
%cd LLaMA-Factory
!pip install -e ".[torch,metrics]"
```

发现出现环境冲突，使用 pip install --no-deps -e . 解决

```
!pip install --no-deps -e .
```

运行如下命令，如果显示llamafactory-cli的版本，则表示安装成功。

```
!llamafactory-cli version
```

## 2. 下载数据集

LLaMA-Factory项目内置了丰富的数据集，放在了`data`目录下。您也可以准备自定义数据集，将数据处理为框架特定的格式，放在`data`下，并且修改`dataset_info.json`文件。

本教程准备了一份sharegpt格式的对话数据集，运行下述命令下载数据。

```
!wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/llama_factory/data_news_300.zip
!unzip data_news_300.zip
```

数据集分为训练集和验证集两个文件，其中训练集包含300组对话样本，是根据新闻分类器的要求准备的：

- 根据新闻标题对新闻进行分类，新闻种类包括：政治、经济、科技、娱乐、体育、教育、健康、国际、国内、社会。
- 输入为新闻标题，输出为新闻所属的种类，不需要额外解释，但保留思维链的内容；在输入的新闻标题前加上“新闻分类：”等类似标签来标明询问目的。

数据样例如下：

```
[
    {
        "conversations": [
            {
                "from": "human",
                "value": "新闻分类：给力！中国两次出手，美到手的订单黄了，该国从此只认中国制造！"
            },
            {
                "from": "gpt",
                "value": "<think>\n首先，我需要分析给出的新闻标题：“给力！中国两次出手，美到手的订单黄了，该国从此只认中国制造！”\n\n接下来，根据标题内容进行分类。标题中提到了中国两次行动导致美国订单出现问题，并且其他国家开始依赖中国制造，说明这涉及国家之间的经济合作和社会影响。\n\n结合新闻种类，考虑到涉及国际贸易和经济合作，最合适的分类是“经济”。所以，这条新闻应该归类到“经济”类别中。\n</think>\n\n经济"
            }
        ]
    },
    {
        "conversations": [
            {
                "from": "human",
                "value": "新闻分类：澳大利亚旅游"
            },
            {
                "from": "gpt",
                "value": "<think>\n好吧，我现在要分析这条新闻标题“澳大利亚旅游”属于哪个类别。首先，标题提到的是澳大利亚旅游，所以它和旅游相关。接下来，我需要看看新闻种类里有没有相关的类别。新闻种类包括政治、经济、科技、娱乐、体育、教育、健康、国际、国内、社会，以及旅游。\n\n根据知识储备，旅游本身就属于其中一个类别，可能属于社会或娱乐，但通常旅游更具体，可能单独作为一个类别。不过在这份分类中，并没有独自一个旅游的类别，可能需要看看有没有更合适的。比如，如果列举的是旅游，可能属于社会，因为旅游涉及到人们的生活方式和社会活动。\n\n再想一下，可能“旅游”不太明显属于哪一个大类，可能需要归类到社会中。或者，在给定的选项中，可能没有明确的旅游类别，所以可能需要根据其他因素来判断。假设在预定义的分类中，有旅游作为独立类别的话，那就是旅游；如果没有，就放到社会里。\n\n不过根据用户提供的分类，旅游并没有单独作为一个类，所以可能得归类到社会。或者，可能更接近于“国际”或“国内”，因为澳大利亚可能分为国际或国内旅游。但在这份列表里，国际和国内是两个类别，所以“澳大利亚旅游”作为国内的，可能更准确。\n\n再仔细分析，用户提供的新闻类型是国际、国内，所以如果是国内，应该是国内旅游；如果是跨国，比如其他国家的旅游，那就是国际。所以“澳大利亚旅游”属于国内，因为澳大利亚是中国的国内。所以最终归类到“国内”。\n</think>\n\n国内"
            }
        ]
    }
]
```

## 3. 原模型对话

### 3.1 启动Web UI

做好前序准备工作后，直接运行下述命令就可以启动Web UI。这里用到的环境变量解释如下：

- `USE_MODELSCOPE_HUB`设为1，表示模型来源是ModelScope。使用HuggingFace模型可能会有网络问题。

点击返回的URL地址，进入Web UI页面。

```
!export USE_MODELSCOPE_HUB=1 && \
llamafactory-cli webui
```

### 3.2 将模型下载到本地（推荐，可选）

启动 WebUI 选择模型后，在启动聊天或者训练时，会自动将模型下载到本地，且在 notebook 中展示下载进度，但是在下载速度慢的情况下，下载进度刷新频繁，不方便查看。所以我们推荐先手动将模型下载到本地。

打开 DSW 的 Terminal，如果出现以下显示，

![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/initial_terminal.png)

输入 bash，进入操作更便捷的 bash 命令行界面：

![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/bash_terminal.png)

输入以下命令，下载模型 DeepSeek-R1-Distill-Qwen-7B:

```
modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

可以看到，模型被下载到了 /mnt/workspace/.cache/modelscope/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 目录下，等待下载完成即可。 ![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/model_download.png)

### 3.3 加载原模型，进行对话

进入WebUI后，可以切换到中文（zh）。首先配置模型，本教程选择DeepSeek-R1-7B-DIstill模型，对话模板选择deepseek3。

如果将模型下载到了本地，更改模型路径为 /mnt/workspace/.cache/modelscope/hub/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/ ![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/model_params_local_path.png)

如果未将模型下载到本地，保留原配置(ModelScope 模型标识符)。如果已经将模型下载到本地，仍使用ModelScope 模型标识符的话，会重新下载模型对已下载的模型进行覆盖。 ![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/model_params.png)

选择「Chat」栏，确保适配器路径是空白的，表示没有加载 LoRA 部分，点击「加载模型」即可在Web UI中和原模型进行对话。 ![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/chat_params_origin.png)

在页面底部的对话框输入想要和模型对话的内容，点击「提交」即可发送消息。发送后模型会逐字生成回答，可以看出模型有时能够准确识别新闻分类的意图，但回复格式与要求不符，有时并不能够识别用户新闻分类的意图。

![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/chat_result_origin_1.png)

![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/chat_result_origin_2.png)

完成对话后，点击「卸载模型」对模型进行卸载，避免占用显存影响下面微调训练。模型卸载完成后，会显示“模型已卸载”。

![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/chat_unload_model.png)

现在我们运用前面提到的数据集对模型进行微调，看能否达到想要的效果。

## 4. 模型微调

### 4.1 配置参数

微调方法则保持默认值lora，数据集使用上述下载的`train`数据文件。 ![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/finetune_data.png)

可以点击「预览数据集」。点击关闭返回训练界面。 ![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/finetune_preview.png)

设置学习率为5e-6，梯度累积为2，有利于模型拟合。 ![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/finetune_params_A10.png)

点击LoRA参数设置展开参数列表，设置LoRA+学习率比例为16，LoRA+被证明是比LoRA学习效果更好的算法。在LoRA作用模块中填写all，即将LoRA层挂载到模型的所有线性层上，提高拟合效果。 ![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/finetune_lora.png)

### 4.2 启动微调

输出目录默认以时间作为后缀，记住这个时间戳，训练后的LoRA权重将会保存在此目录中。点击「预览命令」可展示所有已配置的参数，您如果想通过代码运行微调，可以复制这段命令，在命令行运行。

点击「开始」启动模型微调。 ![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/finetune_start.png)

启动微调后需要等待一段时间，待模型下载完毕后可在界面观察到训练进度和损失曲线。显示“训练完毕”代表微调成功。 ![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/finetune_result_A10.png)

## 5. 模型评估

微调完成后，点击页面上方的检查点路径，会弹出训练完成的LoRA权重，点击选择刚刚训练好的输出目录，在模型启动时即可加载微调结果。 ![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/evaluate_adaptor.png)

选择「Evaluate&Predict」栏，数据路径为 data_news_300, 在数据集下拉列表中选择「eval」（验证集）评估模型。可以更改输出目录名，模型评估结果将会保存在该目录中。最后点击开始按钮启动模型评估。 ![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/evaluate_start.png)

模型评估完成后会在界面上显示验证集的分数。其中ROUGE分数衡量了模型输出答案（predict）和验证集中标准答案（label）的相似度，ROUGE分数越高代表模型学习得更好。

## 6. 模型对话

选择「Chat」栏，如果已经有加载的模型，先点击「卸载模型」进行卸载，然后确保适配器路径是刚刚的训练输出路径，点击「加载模型」，模型加载完毕后，即可在Web UI中和微调模型进行对话。 ![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/chat_params.png)

在页面底部的对话框输入想要和模型对话的内容，点击「提交」即可发送消息。发送后模型会逐字生成回答，从回答中可以发现模型学习到了数据集中的内容，能够识别出用户是想对后面的标题进行新闻分类，且满足要求的输出格式，直接输出新闻分类，没有额外解释。 ![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/chat_result_finetune_1.png) ![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/chat_result_finetune_2.png)

把前缀“新闻分类”改成类似的表达去询问，发现仍能满足要求。 ![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/chat_result_finetune_3.png)

清空历史后（避免上下文的影响），问模型其他的问题，发现其仍能正常回答，没有过拟合。 ![image.png](https://dsw-js.data.aliyun.com/production/pai-dsw-examples/v0.6.180/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b/preview/_images/chat_result_finetune_4.png)

## 7. 总结

本次教程介绍了如何使用PAI和LLaMA Factory框架，基于轻量化LoRA方法微调DeepSeek-R1-Distill-Qwen-7B模型，使其能够识别新闻标题分类的询问并按照指定格式输出回答，同时通过验证集ROUGE分数和人工测试验证了微调的效果。在后续实践中，可以使用实际业务数据集，对模型进行微调，得到能够解决实际业务场景问题的本地领域大模型。