#!/bin/bash
set -u
MODE=$1
RUN=$2
ROOT=/workspace/a28-results
mkdir -p "$ROOT"
VENV=/workspace/${MODE}-venv
PLUGIN=/workspace/${MODE}-plugin
source /workspace/env.sh
LOG=$ROOT/${MODE}-${RUN}-serve.log
pkill -TERM -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 5
for i in $(seq 1 30); do
  if ! curl -sf http://127.0.0.1:9031/v1/models >/dev/null 2>&1; then break; fi
  sleep 1
done
if curl -sf http://127.0.0.1:9031/v1/models >/dev/null 2>&1; then
  echo "port 9031 still occupied"; exit 1
fi
START=$(date +%s)
"$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 > "$LOG" 2>&1 &
PID=$!
READY=0
for i in $(seq 1 240); do
  if curl -sf http://127.0.0.1:9031/v1/models >/dev/null 2>&1; then READY=1; break; fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "serve exited"; tail -40 "$LOG"; exit 1; fi
  sleep 1
done
if [ "$READY" != 1 ]; then echo "serve did not become ready"; tail -40 "$LOG"; kill -TERM "$PID" 2>/dev/null || true; exit 1; fi
READY_AT=$(date +%s)
echo "startup_seconds=$((READY_AT-START))"
cd "$ROOT"
timeout 1800 "$VENV/bin/python" "$PLUGIN/benchmarks/benchmark_throughput_serve.py" \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[16384,1024,64,128]]' > "$ROOT/${MODE}-${RUN}-bench.log" 2>&1
RC=$?
kill -TERM "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
echo "benchmark_rc=$RC"
tail -35 "$ROOT/${MODE}-${RUN}-bench.log"
exit "$RC"
