#!/bin/bash
set -u
source /workspace/env.sh
ROOT=/workspace/audit-results/c500-baseline
mkdir -p "$ROOT"
START=$(date +%s)
/workspace/baseline-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 > "$ROOT/serve.log" 2>&1 &
PID=$!
READY=0
for i in $(seq 1 1200); do
  if curl -sf http://127.0.0.1:9031/v1/models > "$ROOT/models.json" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "serve exited"; tail -60 "$ROOT/serve.log"; exit 1; fi
  sleep 1
done
if [ "$READY" != 1 ]; then echo "baseline serve not ready"; kill -TERM "$PID" 2>/dev/null || true; exit 1; fi
echo "startup_seconds=$(( $(date +%s) - START ))" | tee "$ROOT/startup.txt"
cd "$ROOT"
timeout 3600 /workspace/baseline-venv/bin/python /workspace/baseline-plugin/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]' > "$ROOT/bench.log" 2>&1
RC=$?
echo "$RC" > "$ROOT/benchmark.rc"
kill -TERM "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
exit "$RC"
