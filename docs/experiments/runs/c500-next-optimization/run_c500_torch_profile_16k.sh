#!/bin/bash
set -u
source /data/c500-env.sh
ROOT=/data/experiments/runs/c500-current-torch-profile-16k
mkdir -p "$ROOT/trace"
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$ROOT/trace\"}" \
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
curl -sf -X POST http://127.0.0.1:9031/start_profile > "$ROOT/start.json"
vllm bench serve --backend vllm --model minicpm \
  --tokenizer /workspace/MiniCPM5-2B --dataset-name random \
  --random-input-len 16384 --random-output-len 512 --num-prompts 64 \
  --max-concurrency 64 --ignore-eos --port 9031 \
  > "$ROOT/bench.log" 2>&1
RC=$?
curl -sf -X POST http://127.0.0.1:9031/stop_profile > "$ROOT/stop.json"
echo "$RC" > "$ROOT/benchmark.rc"
exit "$RC"
