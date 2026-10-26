#!/bin/bash
set -u
source /workspace/bi-env.sh
unset VLLM_FL_DISABLE_DECODE_FAST
ROOT=/workspace/audit-results/bi-final
/workspace/audit-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' > "$ROOT/eval-serve.log" 2>&1 &
PID=$!
READY=0
for i in $(seq 1 1200); do
  if curl -sf http://127.0.0.1:9031/v1/models > "$ROOT/eval-models.json" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "eval serve exited"; tail -60 "$ROOT/eval-serve.log"; exit 1; fi
  sleep 1
done
if [ "$READY" != 1 ]; then echo "eval serve not ready"; kill -TERM "$PID" 2>/dev/null || true; exit 1; fi
for RUN in 1 2; do
  mkdir -p "$ROOT/eval$RUN"
  evalscope eval --model minicpm \
    --api-url http://127.0.0.1:9031/v1/chat/completions --api-key EMPTY \
    --eval-type openai_api --datasets math_500 \
    --dataset-args '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}' \
    --eval-batch-size 8 --timeout 3600 \
    --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
    --work-dir "$ROOT/eval$RUN" --ignore-errors > "$ROOT/eval$RUN.log" 2>&1
  echo $? > "$ROOT/eval$RUN.rc"
  JSON=$(ls "$ROOT/eval$RUN"/*/reports/minicpm/math_500.json 2>/dev/null | head -1)
  if [ -n "$JSON" ]; then cp "$JSON" "$ROOT/eval$RUN-math500.json"; fi
done
kill -TERM "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
