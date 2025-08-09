 curl -k -X POST "http://172.16.1.3:8010/v1/chat/completions" \
 -H "Content-Type: application/json" \
 -d '{
  "model": "/data/model/cognitivecomputations/DeepSeek-R1-awq",
  "messages": [
   {"role": "system", "content": "你是一位专业的医疗AI助手，请根据提供的临床症状和检查结果，进行严谨的医学诊断推理。"},
   {"role": "user", "content": "一名40岁男性突发上腹剧痛伴恶心半小时，查体显示上腹部深压痛，但Murphy征阴性，血常规显示白细胞计数略增，尿便常规正常，考虑其最可能的诊断是什么？ 请逐步推理并给出答案。"}
  ],
  "max_tokens": 3000,
  "temperature": 0.5,
  "top_p": 0.95
 }'