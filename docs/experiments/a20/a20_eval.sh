#!/bin/bash
# MATH-500 Level 3 with the A20 decode kernel (fast path on).
set -u
pkill -9 -f "vllm serve"; pkill -9 -f "VLLM::EngineCore"; sleep 3
cd /workspace
VLLM_PLUGINS=fl nohup vllm serve /workspace/MiniCPM5-2B --port 9031 \
  --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  > /workspace/experiments/serve_a20_eval.log 2>&1 &
for i in $(seq 1 300); do
  grep -q "Application startup complete" /workspace/experiments/serve_a20_eval.log && break
  sleep 2
done
grep -q "Application startup complete" /workspace/experiments/serve_a20_eval.log || { echo SERVE_FAILED; exit 1; }
evalscope eval --model minicpm --api-url http://127.0.0.1:9031/v1/chat/completions \
  --api-key EMPTY --eval-type openai_api --datasets math_500 \
  --dataset-args '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}' \
  --eval-batch-size 8 --timeout 3600 \
  --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
  --work-dir /workspace/evalscope-datasets/level3_a20 --ignore-errors
echo "EVAL_DONE"
