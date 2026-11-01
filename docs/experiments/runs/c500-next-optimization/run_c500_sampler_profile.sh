#!/bin/bash
set -u
source /data/c500-env.sh
ROOT=/data/experiments/runs/c500-current-sampler-profile
mkdir -p "$ROOT"
export HOST_PROFILE="$ROOT/host"
python /data/host_profile_sampler.py \
  --model /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  > "$ROOT/serve.log" 2>&1 &
PID=$!
echo "$PID" > "$ROOT/serve.pid"
cleanup() {
  kill -TERM "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
}
trap cleanup EXIT
READY=0
for _ in $(seq 1 1200); do
  if curl -sf http://127.0.0.1:9031/v1/models > "$ROOT/models.json" 2>/dev/null; then
    READY=1
    break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    tail -60 "$ROOT/serve.log" >&2
    exit 1
  fi
  sleep 1
done
if [ "$READY" != 1 ]; then
  exit 1
fi
vllm bench serve --backend vllm --model minicpm \
  --tokenizer /workspace/MiniCPM5-2B --dataset-name random \
  --random-input-len 4096 --random-output-len 512 --num-prompts 64 \
  --max-concurrency 64 --ignore-eos --port 9031 \
  > "$ROOT/bench.log" 2>&1
RC=$?
echo "$RC" > "$ROOT/benchmark.rc"
exit "$RC"
