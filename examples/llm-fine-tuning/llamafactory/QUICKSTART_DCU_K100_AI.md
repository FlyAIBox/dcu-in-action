# 🚀 DCU k100-AI LLaMA Factory 5分钟快速入门

> 专为海光DCU k100-AI加速卡设计的大模型微调快速上手指南

## ⚡ 环境准备（2分钟）

### 第一步：检查DCU环境

```bash
# 检查DCU驱动
dcu-smi -L

# 检查DTK版本
cat /opt/dtk/VERSION

# 应该看到类似输出：
# 0: k100-AI（64GB显存）
# DTK 25.04+
```

### 第二步：一键环境配置

```bash
# 克隆项目（如果还没有）
git clone https://github.com/your-org/dcu-in-action.git
cd dcu-in-action/examples/llm-fine-tuning/llamafactory

# 运行自动配置脚本
chmod +x scripts/dcu_k100_ai_setup.sh
./scripts/dcu_k100_ai_setup.sh

# 加载DCU环境
source ~/.bashrc
```

## 🎯 模型微调（3分钟）

### 第一步：启动Web UI

```bash
# 启动LLaMA Factory Web界面
~/dcu_configs/start_webui.sh
```

访问：http://localhost:7860

### 第二步：选择模型和数据

**Train页面配置**：

```
模型名称: qwen2.5-3b-instruct
微调方法: LoRA
数据集: 选择您的数据集文件

DCU k100-AI优化参数:
- 学习率: 2e-4
- 批处理大小: 8
- 梯度累积: 4
- LoRA rank: 32
- 精度: bf16
```

### 第三步：开始训练

1. 点击"Preview Command"检查配置
2. 点击"Start"开始训练
3. 在Console页面监控进度

**预期训练时间**：
- 3B模型：20-30分钟
- 7B模型：45-60分钟

## 🔍 实时监控

在另一个终端运行DCU监控脚本：

```bash
# 监控DCU使用情况
~/dcu_configs/monitor_dcu.sh
```

**正常状态指标**：
- DCU利用率：80-95%
- 显存使用：18-22GB/64GB
- 训练速度：~180 tokens/s

## ✅ 快速验证

### 训练完成后测试模型：

1. **切换到Chat页面**
2. **加载微调模型**：
   ```
   Checkpoint path: saves/qwen2.5-3b-dcu/checkpoint-xxx
   ```
3. **测试对话**：
   ```
   Q: 你学到了什么新知识？
   A: [模型回答体现训练数据特征]
   ```

## 🚀 性能测试（可选）

```bash
# 运行DCU性能测试
python scripts/test_dcu_performance.py --quick

# 快速LLM测试
python scripts/test_dcu_performance.py --llm-only
```

## 📊 预期性能指标

| 模型规模 | 训练速度 | 显存占用 | 训练时间(1000步) |
|----------|----------|----------|------------------|
| Qwen2.5-3B | ~180 tok/s | 18GB | 25分钟 |
| Qwen2.5-7B | ~120 tok/s | 24GB | 40分钟 |
| Qwen2.5-14B | ~80 tok/s | 30GB | 60分钟 |

## ⚠️ 常见问题

### 问题1：显存不足
```bash
# 解决方案：减小批处理大小
per_device_train_batch_size: 4 → 2
gradient_accumulation_steps: 4 → 8
```

### 问题2：训练速度慢
```bash
# 解决方案：检查环境配置
source ~/.dcurc
echo $HIP_VISIBLE_DEVICES  # 应该是0
dcu-smi  # 确认DCU被正确识别
```

### 问题3：DCU不可用
```bash
# 检查驱动状态
sudo systemctl status rock-dkms
sudo dmesg | grep -i dcu
```

## 📞 获取帮助

- **查看详细教程**：`cat doc/LLaMA\ Factory：03-Easy\ Dataset\ 让大模型高效学习领域知识.md`
- **查看完整README**：`cat README_DCU_K100_AI.md`
- **查看使用指南**：`cat ~/DCU_K100_AI_GUIDE.md`
- **官方文档**：https://developer.hygon.cn

## 🎉 下一步

1. **尝试更大模型**：Qwen2.5-7B, Qwen2.5-14B
2. **探索不同数据集**：代码、数学、多语言
3. **部署推理服务**：参考inference示例
4. **多卡分布式训练**：支持多DCU并行

---

**�� 小贴士**：DCU k100-AI的64GB大显存是其最大优势，充分利用可以获得比同级别GPU更好的训练体验！

**🚀 开始您的DCU k100-AI AI之旅吧！** 