# vLLM推理服务压测工具 - 项目结构

## 优化后的目录结构

```
vllm_benchmark_serving/
├── README.md                           # 项目说明文档
├── config.yaml                         # 压测配置文件
├── requirements.txt                    # Python依赖包
├── benchmark.py                        # 统一入口脚本（新增）
├── run.py                             # 批量压测脚本
├── aggregate_results.py               # 结果聚合便捷脚本（新增）
│
├── src/                               # 源代码目录（新增）
│   ├── __init__.py
│   ├── core/                          # 核心模块
│   │   ├── __init__.py
│   │   └── benchmark_serving.py       # 主压测引擎
│   ├── datasets/                      # 数据集处理模块
│   │   ├── __init__.py
│   │   └── benchmark_dataset.py       # 数据集处理框架
│   ├── backends/                      # 后端请求处理模块
│   │   ├── __init__.py
│   │   └── backend_request_func.py    # 后端请求函数
│   ├── utils/                         # 工具函数模块
│   │   ├── __init__.py
│   │   └── benchmark_utils.py         # 通用工具函数
│   └── aggregation/                   # 结果聚合模块
│       ├── __init__.py
│       └── aggregate_result.py        # 结果聚合处理
│
├── docs/                              # 文档目录（新增）
│   ├── architecture.md                # 系统架构图
│   ├── data_flow.md                   # 数据流程图
│   └── project_structure.md           # 项目结构说明
│
├── results/                           # 压测结果目录
│   ├── bench_io256x256_mc1_np10.json
│   ├── bench_io256x256_mc4_np40.json
│   └── aggregate_results_20250727.csv
│
└── .kiro/                             # Kiro IDE配置（可选）
    └── steering/
        └── project_guidelines.md
```

## 模块职责划分

### 1. 入口脚本层
- **benchmark.py**: 统一的命令行入口，支持批量压测、单次压测、结果聚合
- **run.py**: 批量压测控制器，根据配置文件执行多组压测
- **aggregate_results.py**: 结果聚合便捷脚本

### 2. 核心模块层 (src/core/)
- **benchmark_serving.py**: 主压测引擎
  - 异步请求调度
  - 性能指标收集
  - 结果统计分析

### 3. 数据集处理层 (src/datasets/)
- **benchmark_dataset.py**: 数据集处理框架
  - 支持多种数据集格式
  - 统一的数据采样接口
  - 多模态数据处理

### 4. 后端处理层 (src/backends/)
- **backend_request_func.py**: 后端请求处理
  - OpenAI兼容API支持
  - 多种推理后端适配
  - 详细延迟指标收集

### 5. 工具函数层 (src/utils/)
- **benchmark_utils.py**: 通用工具函数
  - 数据格式转换
  - 统计计算辅助
  - 配置处理工具

### 6. 结果处理层 (src/aggregation/)
- **aggregate_result.py**: 结果聚合处理
  - JSON结果文件合并
  - CSV报告生成
  - 双语输出支持

## 使用方式对比

### 原有使用方式（仍然支持）
```bash
# 批量压测
python3 run.py

# 单次压测
python3 src/core/benchmark_serving.py --backend vllm --model xxx --base-url xxx

# 聚合结果
python3 aggregate_results.py
```

### 新的统一入口方式
```bash
# 批量压测
python benchmark.py batch

# 单次压测
python benchmark.py single --model xxx --base-url xxx --num-prompts 100

# 聚合结果
python benchmark.py aggregate
```

## 优化特点

### 1. 模块化设计
- 按功能职责清晰分离
- 便于维护和扩展
- 降低模块间耦合

### 2. 向后兼容
- 保持原有脚本可用
- 最小化代码改动
- 渐进式迁移支持

### 3. 统一入口
- 提供便捷的命令行接口
- 减少用户学习成本
- 标准化操作流程

### 4. 文档完善
- 架构图和流程图
- 详细的使用说明
- 代码结构文档

### 5. IDE友好
- 支持Kiro IDE配置
- 便于代码导航和调试
- 模块化导入路径

## 迁移指南

### 对于现有用户
1. 现有的使用方式完全不变
2. 可以逐步迁移到新的统一入口
3. 配置文件格式保持不变

### 对于新用户
1. 推荐使用 `python benchmark.py` 统一入口
2. 参考 `docs/` 目录下的文档
3. 使用 `benchmark.py --help` 查看帮助

### 对于开发者
1. 新功能开发在对应的 `src/` 子模块中
2. 遵循模块化设计原则
3. 更新相应的文档和测试