#!/bin/bash
set -u
ROOT=/workspace/experiments/runs/a36-fullpage-loop
while kill -0 24329 2>/dev/null; do
  sleep 10
done
if [ ! -f /workspace/experiments/runs/a36-4k-b2/summary.csv ]; then
  echo "A/B did not complete; eval not started" >&2
  exit 1
fi
source /workspace/bi-env.sh
export PYTHONPATH=/workspace/audit-plugin:/workspace/a36-official-gems/src
/workspace/audit-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  > "$ROOT/eval-serve.log" 2>&1 &
PID=$!
cleanup() {
  RC=$?
  kill -TERM "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
  ixsmi > "$ROOT/eval-gpu-after.txt" 2>&1 || true
  exit "$RC"
}
trap cleanup EXIT
READY=0
for _ in $(seq 1 1200); do
  if curl -sf http://127.0.0.1:9031/v1/models > "$ROOT/eval-models.json" 2>/dev/null; then
    READY=1
    break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "eval serve exited" >&2
    tail -60 "$ROOT/eval-serve.log" >&2
    exit 1
  fi
  sleep 1
done
if [ "$READY" != 1 ]; then
  echo "eval serve not ready" >&2
  exit 1
fi
mkdir -p "$ROOT/eval1"
evalscope eval --model minicpm \
  --api-url http://127.0.0.1:9031/v1/chat/completions --api-key EMPTY \
  --eval-type openai_api --datasets math_500 \
  --dataset-args '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}' \
  --eval-batch-size 8 --timeout 3600 \
  --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
  --work-dir "$ROOT/eval1" --ignore-errors > "$ROOT/eval1.log" 2>&1
RC=$?
echo "$RC" > "$ROOT/eval1.rc"
JSON=$(ls "$ROOT/eval1"/*/reports/minicpm/math_500.json 2>/dev/null | head -1 || true)
if [ -n "$JSON" ]; then
  cp "$JSON" "$ROOT/eval1-math500.json"
fi
exit "$RC"
