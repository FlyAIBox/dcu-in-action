export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_P2P_LEVEL=SYS
export VLLM_NUMA_BIND=0
export VLLM_RANK0_NUMA=0
export VLLM_RANK1_NUMA=0
export VLLM_RANK2_NUMA=0
export VLLM_RANK3_NUMA=0
export VLLM_RANK4_NUMA=0
export VLLM_RANK5_NUMA=0
export VLLM_RANK6_NUMA=0
export VLLM_RANK7_NUMA=0


vllm serve /模型地址  --trust-remote-code  --dtype float16 --max-model-len 32768 --max-seq-len-to-capture 32768 -tp 4 --gpu-memory-utilization 0.9  --disable-log-requests --port 8888