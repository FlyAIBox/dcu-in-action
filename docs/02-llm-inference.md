# 海光DCU大模型推理实战教程

## 📋 目录
- [环境准备](#环境准备)
- [推理框架选择](#推理框架选择)
- [模型部署](#模型部署)
- [性能优化](#性能优化)
- [实际应用](#实际应用)
- [故障排查](#故障排查)

---

## 🚀 环境准备

### 1. DCU环境检查

在开始推理之前，确保DCU环境正确配置：

```bash
# 检查DCU设备状态
hy-smi

# 查看DTK版本
dtk-config --version

# 验证PyTorch+ROCm环境
python -c "import torch; print(f'DCU可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'DCU数量: {torch.cuda.device_count()}')"
```

### 2. 依赖安装

```bash
# 安装推理相关依赖
pip install transformers>=4.35.0
pip install accelerate>=0.24.0
pip install bitsandbytes>=0.41.0
pip install optimum>=1.15.0

# 安装推理加速框架
pip install vllm  # 高性能推理引擎
pip install text-generation-webui  # Web界面
```

---

## 🛠️ 推理框架选择

### 1. Transformers原生推理

适用于小规模模型和实验场景：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载模型和分词器
model_name = "THUDM/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 推理
prompt = "介绍一下海光DCU加速卡的优势"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 2. vLLM高性能推理

适用于生产环境和高并发场景：

```python
from vllm import LLM, SamplingParams

# 初始化vLLM引擎
llm = LLM(
    model="Qwen/Qwen-7B-Chat",
    tensor_parallel_size=2,  # 多卡并行
    gpu_memory_utilization=0.9,
    dtype="float16"
)

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512
)

# 批量推理
prompts = [
    "解释什么是人工智能",
    "介绍Python编程语言",
    "描述机器学习的应用"
]

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"输入: {output.prompt}")
    print(f"输出: {output.outputs[0].text}")
    print("-" * 50)
```

### 3. 使用API服务

部署推理API服务：

```python
# FastAPI推理服务
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# 全局加载模型
model = None
tokenizer = None

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512
    temperature: float = 0.7

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    model_name = "baichuan-inc/Baichuan2-7B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

@app.post("/chat")
async def chat(request: ChatRequest):
    inputs = tokenizer(request.message, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

# 启动命令: uvicorn inference_api:app --host 0.0.0.0 --port 8000
```

---

## ⚡ 性能优化

### 1. 混合精度推理

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 使用FP16精度
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/DialoGPT-medium",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 使用BF16精度（推荐）
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/DialoGPT-medium",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

### 2. 量化推理

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 4bit量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
```

### 3. KV缓存优化

```python
from transformers import AutoModelForCausalLM

# 启用KV缓存优化
model = AutoModelForCausalLM.from_pretrained(
    "gpt2-medium",
    torch_dtype=torch.float16,
    device_map="auto",
    use_cache=True  # 启用KV缓存
)

# 对于长文本生成，使用past_key_values
past_key_values = None
for i in range(5):  # 多轮对话
    inputs = tokenizer(f"第{i+1}轮对话内容", return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            past_key_values=past_key_values,
            use_cache=True
        )
    past_key_values = outputs.past_key_values
```

### 4. 批处理优化

```python
def batch_inference(model, tokenizer, prompts, batch_size=4):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        
        # 批量编码
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        ).to("cuda")
        
        # 批量推理
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )
        
        # 解码结果
        batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(batch_results)
    
    return results
```

---

## 📱 实际应用场景

### 1. 聊天机器人

```python
class ChatBot:
    def __init__(self, model_name="THUDM/chatglm3-6b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.chat_history = []
    
    def chat(self, user_input):
        # 构建对话历史
        self.chat_history.append(f"用户: {user_input}")
        context = "\n".join(self.chat_history[-10:])  # 保留最近10轮对话
        
        # 生成回复
        inputs = self.tokenizer(context, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_reply = response.split("助手:")[-1].strip()
        
        self.chat_history.append(f"助手: {assistant_reply}")
        return assistant_reply

# 使用示例
bot = ChatBot()
while True:
    user_input = input("用户: ")
    if user_input.lower() in ['exit', 'quit', '退出']:
        break
    reply = bot.chat(user_input)
    print(f"助手: {reply}")
```

### 2. 文档问答系统

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class DocumentQA:
    def __init__(self, model_name="THUDM/chatglm3-6b"):
        # 加载问答模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载向量模型
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
        self.index = None
    
    def add_documents(self, docs):
        self.documents.extend(docs)
        # 计算文档向量
        doc_embeddings = self.embedding_model.encode(docs)
        
        if self.embeddings is None:
            self.embeddings = doc_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, doc_embeddings])
        
        # 构建FAISS索引
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings.astype('float32'))
    
    def search_documents(self, query, top_k=3):
        query_embedding = self.embedding_model.encode([query])
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        relevant_docs = [self.documents[i] for i in indices[0]]
        return relevant_docs
    
    def answer_question(self, question):
        # 检索相关文档
        relevant_docs = self.search_documents(question)
        context = "\n".join(relevant_docs)
        
        # 构建提示
        prompt = f"""基于以下文档内容回答问题：

文档内容：
{context}

问题：{question}

回答："""
        
        # 生成答案
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.split("回答：")[-1].strip()

# 使用示例
qa_system = DocumentQA()
documents = [
    "海光DCU是中国自主研发的GPU加速计算芯片...",
    "PyTorch是Facebook开发的深度学习框架...",
    "机器学习是人工智能的一个重要分支..."
]
qa_system.add_documents(documents)

question = "什么是海光DCU？"
answer = qa_system.answer_question(question)
print(f"问题: {question}")
print(f"答案: {answer}")
```

### 3. 代码生成助手

```python
class CodeGenerator:
    def __init__(self, model_name="microsoft/CodeGPT-small-py"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def generate_code(self, description, language="python"):
        prompt = f"""# 任务描述：{description}
# 编程语言：{language}
# 代码实现：

"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                stop_strings=["# 任务描述", "```"]
            )
        
        code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return code.split("# 代码实现：")[-1].strip()

# 使用示例
code_gen = CodeGenerator()
description = "实现一个快速排序算法"
code = code_gen.generate_code(description)
print(f"生成的代码：\n{code}")
```

---

## 🔧 故障排查

### 1. 常见错误及解决方案

#### 错误1：CUDA out of memory
```bash
# 错误信息
RuntimeError: CUDA out of memory. Tried to allocate XX GB

# 解决方案
1. 减少批次大小
2. 使用梯度累积
3. 启用混合精度
4. 使用模型量化
5. 增加虚拟内存
```

#### 错误2：模型加载失败
```bash
# 错误信息
OSError: Can't load the model

# 解决方案
1. 检查网络连接
2. 使用本地模型路径
3. 验证模型文件完整性
4. 清除缓存后重新下载
```

#### 错误3：DCU设备不可用
```bash
# 错误信息
AssertionError: DCU is not available

# 解决方案
1. 检查DCU驱动安装
2. 验证DTK环境变量
3. 重启DCU服务
4. 检查设备权限
```

### 2. 性能调优建议

```python
# 性能监控工具
import time
import psutil
import torch

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        torch.cuda.empty_cache()  # 清空缓存
        torch.cuda.synchronize()  # 同步
        self.start_time = time.time()
    
    def end(self):
        torch.cuda.synchronize()
        self.end_time = time.time()
    
    def get_stats(self):
        # 计算用时
        inference_time = self.end_time - self.start_time
        
        # GPU内存使用
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**3
        
        # CPU和系统内存
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        
        return {
            "inference_time": f"{inference_time:.3f}s",
            "gpu_memory_current": f"{gpu_memory:.2f}GB",
            "gpu_memory_peak": f"{gpu_memory_max:.2f}GB",
            "cpu_usage": f"{cpu_percent:.1f}%",
            "system_memory": f"{memory_info.percent:.1f}%"
        }

# 使用示例
monitor = PerformanceMonitor()
monitor.start()

# 执行推理任务
# your_inference_code_here()

monitor.end()
stats = monitor.get_stats()
print("性能统计:", stats)
```

### 3. 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 设置环境变量进行调试
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步执行
os.environ['TORCH_USE_CUDA_DSA'] = '1'    # 检测内存错误

# 使用torch.autograd.profiler分析性能
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    # 执行推理代码
    output = model.generate(**inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

## 📊 性能对比

| 推理框架 | 模型大小 | 推理速度 | 内存占用 | 并发支持 |
|----------|----------|----------|----------|----------|
| Transformers | 7B | 50 tokens/s | 14GB | 低 |
| vLLM | 7B | 150 tokens/s | 16GB | 高 |
| TensorRT-LLM | 7B | 200 tokens/s | 12GB | 中 |
| 量化版本 | 7B | 120 tokens/s | 8GB | 中 |

---

## 🎯 最佳实践

1. **模型选择**：根据任务需求选择合适大小的模型
2. **精度平衡**：在精度和性能之间找到平衡点
3. **批处理**：充分利用批处理提高吞吐量
4. **缓存策略**：合理使用KV缓存减少重复计算
5. **监控调优**：持续监控性能指标并优化

---

## 📚 参考资源

- [海光DCU开发者文档](https://developer.sourcefind.cn/)
- [Transformers官方文档](https://huggingface.co/docs/transformers)
- [vLLM项目文档](https://vllm.readthedocs.io/)
- [PyTorch性能调优指南](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

*本文档持续更新中，欢迎提交建议和改进！*
