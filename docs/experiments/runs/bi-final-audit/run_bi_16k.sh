#!/bin/bash
set -u
MODE=$1
RUN=$2
source /workspace/bi-env.sh
ROOT=/workspace/audit-results/bi-final/$RUN-$MODE
mkdir -p "$ROOT"
START=$(date +%s)
if [ "$MODE" = a20 ]; then
  unset VLLM_FL_DISABLE_DECODE_FAST
else
  export VLLM_FL_DISABLE_DECODE_FAST=1
fi
/workspace/audit-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' > "$ROOT/16k-serve.log" 2>&1 &
PID=$!
READY=0
for i in $(seq 1 1200); do
  if curl -sf http://127.0.0.1:9031/v1/models > "$ROOT/16k-models.json" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "serve exited"; tail -60 "$ROOT/16k-serve.log"; exit 1; fi
  sleep 1
done
if [ "$READY" != 1 ]; then echo "serve not ready"; kill -TERM "$PID" 2>/dev/null || true; exit 1; fi
echo "startup_seconds=$(( $(date +%s) - START ))" | tee "$ROOT/16k-startup.txt"
cd "$ROOT"
timeout 7200 /workspace/audit-venv/bin/python /workspace/audit-plugin/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[16384,1024,64,128]]' > "$ROOT/16k-bench.log" 2>&1
RC=$?
echo "$RC" > "$ROOT/16k-benchmark.rc"
kill -TERM "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
exit "$RC"
