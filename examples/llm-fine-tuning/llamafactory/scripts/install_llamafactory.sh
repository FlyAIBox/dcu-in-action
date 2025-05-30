#!/bin/bash
# LLaMA Factory 一键安装脚本
# 支持GPU/CPU环境自动检测和配置

set -e

echo "🚀 开始安装LLaMA Factory环境..."

# 检查GPU环境
if command -v nvidia-smi &> /dev/null; then
    echo "✓ 检测到NVIDIA GPU"
    nvidia-smi
    GPU_AVAILABLE=true
else
    echo "⚠️ 未检测到NVIDIA GPU，将使用CPU模式"
    GPU_AVAILABLE=false
fi

# 检查Python版本
PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "Python版本: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION < 3.8" | bc -l) -eq 1 ]]; then
    echo "❌ Python版本过低，需要Python 3.8+，当前版本: $PYTHON_VERSION"
    exit 1
fi

# 创建虚拟环境
echo "📦 创建Python虚拟环境..."
python3 -m venv llamafactory_env
source llamafactory_env/bin/activate

# 升级pip
echo "⬆️ 升级pip..."
pip install --upgrade pip setuptools wheel

# 安装PyTorch
echo "🔧 安装PyTorch..."
if [ "$GPU_AVAILABLE" = true ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 克隆LLaMA Factory仓库
echo "📥 克隆LLaMA Factory仓库..."
if [ ! -d "LLaMA-Factory" ]; then
    git clone https://github.com/hiyouga/LLaMA-Factory.git
fi
cd LLaMA-Factory

# 安装LLaMA Factory
echo "🛠️ 安装LLaMA Factory..."
pip install -e .

# 安装额外依赖
echo "📚 安装额外依赖..."
pip install datasets transformers accelerate peft trl
pip install bitsandbytes  # 用于量化训练
pip install wandb  # 用于实验跟踪
pip install gradio  # WebUI界面
pip install fastapi uvicorn  # API服务
pip install rouge-score nltk  # 评估工具

# 下载NLTK数据
echo "📊 下载NLTK数据..."
python3 -c "import nltk; nltk.download('punkt')"

# 验证安装
echo "✅ 验证安装..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers版本: {transformers.__version__}')"

if [ "$GPU_AVAILABLE" = true ]; then
    python3 -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
    python3 -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        echo "✅ GPU环境配置成功"
    else
        echo "⚠️ GPU环境配置可能有问题，但基础安装完成"
    fi
fi

# 创建示例配置文件
echo "📄 创建示例配置文件..."
mkdir -p examples/configs

cat > examples/configs/lora_config.yaml << 'EOF'
# LoRA微调配置示例
model_name_or_path: "THUDM/chatglm3-6b"
dataset: "alpaca_zh"
template: "chatglm3"
finetuning_type: "lora"
lora_target: "query_key_value"
output_dir: "./saves/ChatGLM3-6B/lora/train"

# LoRA配置
lora_rank: 64
lora_alpha: 16
lora_dropout: 0.1
modules_to_save: "embed_tokens,lm_head"

# 训练参数
stage: "sft"
do_train: true
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
lr_scheduler_type: "cosine"
logging_steps: 10
save_steps: 1000
learning_rate: 5.0e-5
num_train_epochs: 3.0
max_grad_norm: 1.0
quantization_bit: 4
use_unsloth: true
val_size: 0.1
evaluation_strategy: "steps"
eval_steps: 500
load_best_model_at_end: true
plot_loss: true
EOF

cat > examples/configs/qlora_config.yaml << 'EOF'
# QLoRA微调配置示例
model_name_or_path: "meta-llama/Llama-2-7b-chat-hf"
dataset: "alpaca_en"
template: "llama2"
finetuning_type: "lora"
lora_target: "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"
output_dir: "./saves/Llama2-7B/qlora/train"

# QLoRA配置
lora_rank: 64
lora_alpha: 16
lora_dropout: 0.1
quantization_bit: 4
use_double_quant: true
quant_type: "nf4"
compute_dtype: "bfloat16"

# 训练参数
stage: "sft"
do_train: true
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
lr_scheduler_type: "cosine"
logging_steps: 10
save_steps: 1000
learning_rate: 2.0e-5
num_train_epochs: 3.0
max_grad_norm: 1.0
val_size: 0.1
evaluation_strategy: "steps"
eval_steps: 500
load_best_model_at_end: true
plot_loss: true
EOF

# 创建启动脚本
cat > start_webui.sh << 'EOF'
#!/bin/bash
# 启动WebUI的便捷脚本
source llamafactory_env/bin/activate
cd LLaMA-Factory
python src/train_web.py
EOF

chmod +x start_webui.sh

cat > requirements.txt << 'EOF'
torch>=2.0.0
transformers>=4.35.0
datasets>=2.12.0
accelerate>=0.24.0
peft>=0.6.0
trl>=0.7.0
bitsandbytes>=0.41.0
wandb>=0.15.0
gradio>=3.50.0
fastapi>=0.104.0
uvicorn>=0.24.0
rouge-score>=0.1.2
nltk>=3.8.0
pyyaml>=6.0
pandas>=1.5.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
EOF

echo "🎉 LLaMA Factory安装完成！"
echo ""
echo "📝 后续步骤："
echo "1. 激活虚拟环境: source llamafactory_env/bin/activate"
echo "2. 启动WebUI: ./start_webui.sh"
echo "3. 或使用命令行: cd LLaMA-Factory && llamafactory-cli train examples/configs/lora_config.yaml"
echo ""
echo "📂 配置文件位置:"
echo "   - examples/configs/lora_config.yaml"
echo "   - examples/configs/qlora_config.yaml"
echo ""
echo "🌐 WebUI访问地址: http://localhost:7860"
echo ""

# 显示系统信息
echo "💻 系统信息:"
echo "   - Python: $PYTHON_VERSION"
if [ "$GPU_AVAILABLE" = true ]; then
    echo "   - GPU: 已检测到"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1 | awk '{print "   - GPU型号: " $0}'
else
    echo "   - GPU: 未检测到"
fi
echo "   - 安装目录: $(pwd)"
echo ""
echo "✨ 安装完成！开始您的大模型微调之旅吧！" 