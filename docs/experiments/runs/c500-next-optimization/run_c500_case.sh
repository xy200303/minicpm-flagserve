#!/bin/bash
set -u
EXP_ID=$1
CASE=$2
source "${ENV_FILE:-/data/c500-env.sh}"
ROOT=/data/experiments/runs/$EXP_ID
PLUGIN_ROOT=${PLUGIN_ROOT:-/data/src/vllm-plugin-FL}
GEMS_ROOT=${GEMS_ROOT:-/data/src/FlagGems}
mkdir -p "$ROOT"
{
  date -Is
  uname -a
  git -C "$PLUGIN_ROOT" rev-parse HEAD
  git -C "$PLUGIN_ROOT" status --short
  git -C "$GEMS_ROOT" rev-parse HEAD
  git -C "$GEMS_ROOT" status --short
  python -m pip show vllm vllm-plugin-fl flag_gems torch
} > "$ROOT/versions.txt" 2>&1
mx-smi > "$ROOT/gpu-before.txt" 2>&1
free -h > "$ROOT/memory-before.txt" 2>&1
printf 'exp_id=%s\ncase=%s\nplugin_root=%s\ngems_root=%s\n' \
  "$EXP_ID" "$CASE" "$PLUGIN_ROOT" "$GEMS_ROOT" > "$ROOT/config.txt"
START=$(date +%s)
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  > "$ROOT/serve.log" 2>&1 &
PID=$!
echo "$PID" > "$ROOT/serve.pid"
cleanup() {
  kill -TERM "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
  mx-smi > "$ROOT/gpu-after.txt" 2>&1 || true
}
trap cleanup EXIT
READY=0
for _ in $(seq 1 1200); do
  if curl -sf http://127.0.0.1:9031/v1/models > "$ROOT/models.json" 2>/dev/null; then
    READY=1
    break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "serve exited" >&2
    tail -60 "$ROOT/serve.log" >&2
    exit 1
  fi
  sleep 1
done
if [ "$READY" != 1 ]; then
  echo "serve not ready" >&2
  exit 1
fi
echo "startup_seconds=$(( $(date +%s) - START ))" > "$ROOT/startup.txt"
cd "$ROOT"
timeout 3600 python /data/src/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases "$CASE" > "$ROOT/bench.log" 2>&1
RC=$?
echo "$RC" > "$ROOT/benchmark.rc"
SUMMARY=$(ls -t "$ROOT"/benchmark_results/summary_*.csv 2>/dev/null | head -1 || true)
if [ -n "$SUMMARY" ]; then
  cp "$SUMMARY" "$ROOT/summary.csv"
fi
exit "$RC"
