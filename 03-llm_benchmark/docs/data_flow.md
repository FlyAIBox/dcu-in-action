# vLLM推理服务压测工具 - 数据流程图

## 整体数据流程

```mermaid
flowchart TD
    START([开始压测]) --> LOAD_CONFIG[加载config.yaml配置]
    LOAD_CONFIG --> GEN_PARAMS[生成参数组合<br/>输入输出长度 × 并发请求数]
    
    GEN_PARAMS --> LOOP_START{遍历参数组合}
    LOOP_START --> INIT_DATASET[初始化数据集<br/>Random/ShareGPT/Sonnet等]
    
    INIT_DATASET --> LOAD_TOKENIZER[加载分词器<br/>计算token长度]
    LOAD_TOKENIZER --> GEN_REQUESTS[生成测试请求<br/>根据输入输出长度]
    
    GEN_REQUESTS --> ASYNC_DISPATCH[异步请求调度<br/>控制并发数]
    
    subgraph "并发请求处理"
        ASYNC_DISPATCH --> REQ1[请求1]
        ASYNC_DISPATCH --> REQ2[请求2]
        ASYNC_DISPATCH --> REQN[请求N]
        
        REQ1 --> SEND_HTTP1[发送HTTP请求<br/>到vLLM服务器]
        REQ2 --> SEND_HTTP2[发送HTTP请求<br/>到vLLM服务器]
        REQN --> SEND_HTTPN[发送HTTP请求<br/>到vLLM服务器]
        
        SEND_HTTP1 --> STREAM1[处理流式响应<br/>记录时间戳]
        SEND_HTTP2 --> STREAM2[处理流式响应<br/>记录时间戳]
        SEND_HTTPN --> STREAMN[处理流式响应<br/>记录时间戳]
    end
    
    STREAM1 --> COLLECT_METRICS[收集性能指标<br/>TTFT/TPOT/ITL/E2EL]
    STREAM2 --> COLLECT_METRICS
    STREAMN --> COLLECT_METRICS
    
    COLLECT_METRICS --> CALC_STATS[计算统计指标<br/>平均值/中位数/百分位数]
    CALC_STATS --> SAVE_JSON[保存JSON结果文件<br/>bench_io{input}x{output}_mc{concurrency}_np{prompts}.json]
    
    SAVE_JSON --> LOOP_END{是否还有参数组合?}
    LOOP_END -->|是| LOOP_START
    LOOP_END -->|否| AGGREGATE[聚合所有结果<br/>aggregate_result.py]
    
    AGGREGATE --> MERGE_JSON[合并所有JSON文件]
    MERGE_JSON --> GEN_CSV[生成CSV报告<br/>aggregate_results_{date}.csv]
    GEN_CSV --> END([压测完成])
    
    style LOAD_CONFIG fill:#e3f2fd
    style ASYNC_DISPATCH fill:#fff3e0
    style COLLECT_METRICS fill:#e8f5e8
    style SAVE_JSON fill:#fce4ec
```

## 性能指标收集流程

```mermaid
sequenceDiagram
    participant Client as 压测客户端
    participant vLLM as vLLM服务器
    participant Metrics as 指标收集器
    
    Note over Client: 记录请求开始时间
    Client->>vLLM: 发送HTTP POST请求
    Note over vLLM: 模型推理处理
    
    vLLM-->>Client: 返回第一个token
    Note over Metrics: 计算TTFT<br/>(Time to First Token)
    
    loop 流式响应
        vLLM-->>Client: 返回后续token
        Note over Metrics: 记录token间时间<br/>计算ITL (Inter-token Latency)
    end
    
    vLLM-->>Client: 响应结束
    Note over Metrics: 计算最终指标:<br/>• TPOT (Time per Output Token)<br/>• E2EL (End-to-End Latency)<br/>• 总吞吐量
    
    Metrics->>Client: 返回完整性能数据
```

## 数据结构流转

```mermaid
graph LR
    subgraph "输入数据"
        CONFIG_YAML[config.yaml<br/>配置参数]
        DATASET_RAW[原始数据集<br/>ShareGPT/Random等]
    end
    
    subgraph "中间数据结构"
        SAMPLE_REQ[SampleRequest<br/>采样请求对象]
        REQ_INPUT[RequestFuncInput<br/>请求输入结构]
        REQ_OUTPUT[RequestFuncOutput<br/>请求输出结构]
    end
    
    subgraph "输出数据"
        JSON_RESULT[JSON结果文件<br/>单次压测结果]
        CSV_REPORT[CSV聚合报告<br/>所有结果汇总]
    end
    
    CONFIG_YAML --> SAMPLE_REQ
    DATASET_RAW --> SAMPLE_REQ
    SAMPLE_REQ --> REQ_INPUT
    REQ_INPUT --> REQ_OUTPUT
    REQ_OUTPUT --> JSON_RESULT
    JSON_RESULT --> CSV_REPORT
    
    style SAMPLE_REQ fill:#e1f5fe
    style REQ_INPUT fill:#f3e5f5
    style REQ_OUTPUT fill:#e8f5e8
```

## 关键性能指标定义

| 指标 | 英文全称 | 中文含义 | 计算方法 |
|------|----------|----------|----------|
| TTFT | Time to First Token | 首token时间 | 从请求发送到收到第一个token的时间 |
| TPOT | Time per Output Token | 每token输出时间 | 总输出时间 / 输出token数量 |
| ITL | Inter-token Latency | token间延迟 | 相邻两个token之间的时间间隔 |
| E2EL | End-to-End Latency | 端到端延迟 | 从请求发送到响应完成的总时间 |
| Throughput | Request/Token Throughput | 吞吐量 | 每秒处理的请求数/token数 |