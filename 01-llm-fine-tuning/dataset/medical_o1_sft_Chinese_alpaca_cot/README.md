## 数据集简介

`medical_o1_sft_Chinese_alpaca_cot` 是一个用于中文医学领域大模型有监督微调（SFT）的示例数据集，采用 Alpaca 指令风格与链式推理（CoT）风格的回答。数据主要整理自以下公开来源（Apache-2.0 许可），并面向初学者提供易上手的预处理与训练建议：

- 数据来源：FreedomIntelligence 开源的医学推理 SFT 数据集（含中英文子集）
  - 数据站点：[`FreedomIntelligence/medical-o1-reasoning-SFT`](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
  - 官网说明与论文链接见数据站点页（含 2025/04/22 数据拆分说明、License、使用统计等）
- 参考项目：HuatuoGPT-o1（医学复杂推理方向）
  - 项目主页：[`FreedomIntelligence/HuatuoGPT-o1`](https://github.com/FreedomIntelligence/HuatuoGPT-o1)

重要提示（医疗安全与法律合规）
- 本数据集仅供科研与模型训练研究使用，不能替代专业医疗建议或临床决策。
- 训练得到的模型输出不得用于诊断、治疗等医疗行为；如需临床使用，需符合当地法律法规与伦理审查要求。
- 公开来源为 Apache-2.0 许可，但请同时遵循本仓库与上游项目的 License 与使用条款。

## 目录结构

- `medical_o1_sft_Chinese_alpaca_cot.jsonl`：JSON Lines 格式，UTF-8 编码，每行一条样本
- 本文件：`README.md`

## 数据格式

每行是一条 JSON 记录，核心字段如下：
- `prompt`：中文医疗场景指令或问题，通常要求“逐步推理并给出答案”，符合 Alpaca 指令风格；
- `response`：模型期望回答。样例中常见模式为“包含 <think> … </think> 的推理过程 + 最终答案”。

示例（节选并做少量缩减，仅演示字段结构）：

```json
{"prompt": "… 请逐步推理并给出答案。", "response": "<think>…推理过程…</think>\n\n…最终答案…"}
```

说明：上游医学推理数据中，部分样本包含显式的思维链标注（如 `<think>…</think>`）。实际训练时，建议谨慎处理思维链文本（见下文“预处理建议”）。
```json
{
	"prompt": "针对82岁男性，出现头摇不止、肢体震颤、面色淡白、表情淡漠、神疲乏力、动则气短、心悸健忘、眩晕、纳呆等症状，舌体胖大、舌质淡红、舌苔薄白滑、脉沉濡无力或沉细弱，治疗这种病证应首选哪种中药方剂？ 请逐步推理并给出答案。\n患者是一位82岁的老人，他的症状让我立即想到了一些常见的老年问题。头摇不止、肢体震颤这些表现非常明显，这可能跟老年性震颤有关，也让我猜测可能存在中医所说的“肝风内动”的现象。\n\n接着，我注意到患者面色淡白、神疲乏力，这些症状则更像是气血不足。结合动则气短、心悸健忘，典型的气血两虚啊。而且舌体胖大、舌质淡红、舌苔薄白滑也在提示脾虚湿困的情况，脾胃虚弱可能是个关键因素。\n\n针对这些症状，我开始考虑应对的方法。首先想到的是益气养血的经典方剂，比如八珍汤，它就很适合气血不足的症状，但这似乎不完全够，因为还有震颤的问题。\n\n继续思考，震颤问题让我想到天麻钩藤饮，因为它可以平肝息风，对付高血压引起的头痛眩晕效果不错。然而，我稍微犹豫了，因为这方剂可能在抗震颤方面不如一些更专注的方子。\n\n看来，需要重新分析，得找一个兼具益气补血又能息风止颤的方案。镇肝息风汤也许会有帮助，它平肝潜阳的效果不错，但同样也不是完全符合这个病例。\n\n细细梳理，患者的症状中气血两虚的问题特别明显，那么对应的益气补血的方剂可能更适合。哦，对了，还有人参养荣汤！它益气健脾，还有助于补血，简直太合适不过了!\n\n经过这样一番推敲，再一一对应症状，选择人参养荣汤是合理且合适的。这方剂可以帮助改善患者的气血两虚，缓解震颤等问题。",
	"response": "<think>\n针对患者表现出的综合症状，如头摇不止、肢体震颤、面色淡白、神疲乏力、动则气短等，中医考虑这是一种气血两虚并伴有肝风内动、脾虚的情况。结合整个病症的分析，人参养荣汤是较为合适的中药方剂选择。此方以益气补血为主，同时兼顾健脾，能够有效缓解气血两虚导致的体力不支、震颤等症状。因此，建议使用人参养荣汤进行治疗。\n</think>\n\n针对患者表现出的综合症状，如头摇不止、肢体震颤、面色淡白、神疲乏力、动则气短等，中医考虑这是一种气血两虚并伴有肝风内动、脾虚的情况。结合整个病症的分析，人参养荣汤是较为合适的中药方剂选择。此方以益气补血为主，同时兼顾健脾，能够有效缓解气血两虚导致的体力不支、震颤等症状。因此，建议使用人参养荣汤进行治疗。"
}

```
## 预处理建议（强烈推荐）

初学者常见做法是直接把 `prompt` 与 `response` 拼接训练。但医疗场景与思维链文本存在风险与合规考量，建议采用如下策略：

- 是否使用显式 CoT：
  - 保守做法：默认“去除或遮蔽”`<think>…</think>` 思维链，仅保留“最终可验证答案/结论”。
  - 如确需训练推理能力：可将思维链用于“输入上下文（不计入监督目标）”或采用“简短要点式 rationale”，避免长篇 free-form CoT 直接作为监督信号。
- 清洗建议：
  - 过滤超长样本、异常字符；
  - 移除或压缩冗长 `<think>` 内容；
  - 标注“医学免责声明”提示语，避免模型在回答时给出医疗建议性措辞。

### 快速清洗示例（Python）

```python
import json, re
from pathlib import Path

in_path = Path("medical_o1_sft_Chinese_alpaca_cot.jsonl")
out_path = Path("medical_o1_sft_Chinese_alpaca_cot.cleaned.jsonl")

think_pat = re.compile(r"<think>[\s\S]*?</think>", re.MULTILINE)

with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
    for line in fin:
        obj = json.loads(line)
        prompt = (obj.get("prompt") or "").strip()
        resp = (obj.get("response") or "").strip()
        # 去除显式思维链
        resp_no_think = think_pat.sub("", resp).strip()
        # 可选：附加安全提示（根据产品需求自定）
        # resp_no_think = resp_no_think + "\n\n【提示】本回答不构成医疗建议，请咨询专业医生。"
        json.dump({"instruction": prompt, "output": resp_no_think}, fout, ensure_ascii=False)
        fout.write("\n")
```

生成后的行格式示例：
```json
{"instruction": "…", "output": "…（已去除思维链的最终答案）"}
```

## 训练建议（面向初学者）

- 任务类型：指令微调（SFT）。
- 模型选择：建议优先选择已支持中文的开源基础模型（如 Qwen、Llama 中文增强版等）。
- 参数高效微调（PEFT）：优先 QLoRA / LoRA，降低显存需求。
- 典型超参起点（需按机型与样本长度微调）：
  - max_seq_len：1024–2048（结合实际样本长度与显存）；
  - batch_size（总）：64–256（含梯度累积）；
  - learning_rate：1e-5 ～ 2e-5（LoRA 可略高）；
  - epochs：2–3；warmup_ratio：0.03～0.06；
  - fp16/bf16：按硬件与框架支持选择；
  - lr_scheduler：cosine 或 linear；
  - weight_decay：0.01；
- 模板与分隔：保持一贯的“指令-输出”模板；若之后要与英文或其他来源混训，建议统一模板字段名与分隔符。
- 评估与早停：
  - 构造小规模验证集（从本 JSONL 随机切 5–10%），监控 val loss 与领域指标；
  - 人类评审关注：医学事实正确性、可验证性、禁忌用语过滤。
- 安全对齐：
  - 在 SFT 后继续做拒答/安全对齐（DPO/ORPO/SAFE-RLHF 等）时，确保加入医疗场景的安全指引数据。

## 快速查看与统计

- 查看前 2 行：
  ```bash
  head -n 2 medical_o1_sft_Chinese_alpaca_cot.jsonl | cat
  ```
- 样本总数（行数）：
  ```bash
  wc -l medical_o1_sft_Chinese_alpaca_cot.jsonl
  ```
- 随机抽样查看：
  ```bash
  shuf -n 3 medical_o1_sft_Chinese_alpaca_cot.jsonl | cat
  ```

## 医疗与合规注意事项

- 本数据或含医疗诊断/治疗相关内容，训练与使用时务必加入“非医疗建议”声明与使用边界说明；
- 不得用于直接诊疗或替代医师判断；
- 请遵守当地法律法规（含隐私、数据跨境、医疗器械合规等）。

## License 与来源引用

- 上游数据许可：Apache-2.0（详见数据站点与上游项目说明）。
- 使用本数据集请注明来源：
  - 数据：[`FreedomIntelligence/medical-o1-reasoning-SFT`](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
  - 参考项目：[`FreedomIntelligence/HuatuoGPT-o1`](https://github.com/FreedomIntelligence/HuatuoGPT-o1)
- 相关论文（示例引用，详见数据站点）：
  - Chen et al., 2024. HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs. arXiv:2412.18925.

## 版本与变更

- 2025-08-（v0.1）：新增 README，提供样例、清洗与训练建议，面向初学者友好说明。 