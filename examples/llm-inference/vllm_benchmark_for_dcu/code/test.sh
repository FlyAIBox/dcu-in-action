export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "tp,data_type,batch,prompt_tokens,completion_tokens,TOTAL_THROUGHPUT(toks/s),generate_throughput(toks/s), TTFT(ms),TPOT(ms),ITL(ms)" > r1-awq-0705.csv
pairs=( "512 512" "1024 1024" )
model_path="/data/model/cognitivecomputations/DeepSeek-R1-awq/"
tp=8
data_type="float16"
mkdir -p ./log/
for batch in 1 2 4 6 8 10 16 20 24 32 64; do
    for pair in "${pairs[@]}"; do
        prompt_tokens=${pair%% *}
        completion_tokens=${pair#* }
        echo "data_type: $data_type,batch: $batch, prompt_tokens: $prompt_tokens, completion_tokens: $completion_tokens, tp: ${tp}"
        log_path="log/vllm_${model}_batch_${batch}_prompt_tokens_${prompt_tokens}_completion_tokens_${completion_tokens}_tp_${tp}.log"
        touch $log_path
        # benchmark_throughput.py
        python benchmark_serving.py \
                --backend openai \
                --port 8000 \
                --model ${model_path} \
                --trust-remote-code \
                --dataset-name random \
                --ignore-eos \
                --random-input-len ${prompt_tokens} \
                --random-output-len ${completion_tokens} \
                --num-prompts ${batch}  \
                2>&1 | tee  $log_path
        #metric
        E2E_TIME=`grep "^Benchmark duration" $log_path | awk -F ' ' '{print $4}'`
        REQ_THROUGHPUT=`grep "^Request throughput"  $log_path| awk -F ' ' '{print $4}'`
        GEN_THROUGHPUT=`grep "^Output token"  $log_path| awk -F ' ' '{print $5}'`
        TOTAL_THROUGHPUT=`grep "^Total Token" $log_path| awk -F ' ' '{print $5}'`
        TTFT=`grep "^Mean TTFT"  $log_path| awk -F ' ' '{print $4}'`
        TPOT=`grep "^Mean TPOT"  $log_path| awk -F ' ' '{print $4}'`
        ITL=`grep "^Mean ITL"  $log_path| awk -F ' ' '{print $4}'`
        P99_ITL=`grep "^P99 ITL"  $log_path| awk -F ' ' '{print $4}'`
        P99_TTFT=`grep "^P99 TTFT"  $log_path| awk -F ' ' '{print $4}'`
        P99_TPOT=`grep "^P99 TPOT"  $log_path| awk -F ' ' '{print $4}'`
        echo "$tp,$data_type,$batch,$prompt_tokens,$completion_tokens,$TOTAL_THROUGHPUT,$GEN_THROUGHPUT,$TTFT,$TPOT, $ITL" >> r1-awq-0705.csv
    done
done
