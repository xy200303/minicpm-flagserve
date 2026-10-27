#!/bin/bash
set -u

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <exp_id> <test_case_json>" >&2
  exit 2
fi

EXP=$1
CASE=$2
RESULT_ROOT=${RESULT_ROOT:-/workspace/experiments/runs}
PLUGIN_ROOT=${PLUGIN_ROOT:-/workspace/audit-plugin}
GEMS_ROOT=${GEMS_ROOT:-/workspace/audit-gems}
VENV=${VENV:-/workspace/audit-venv}
MODEL=${MODEL:-/workspace/MiniCPM5-2B}
BI_ENV=${BI_ENV:-/workspace/bi-env.sh}
IXSYS=${IXSYS:-0}
IXSYS_DELAY=${IXSYS_DELAY:-300}
IXSYS_LIMIT=${IXSYS_LIMIT:-120}
PROFILE_ONLY=${PROFILE_ONLY:-0}
PROFILE_BENCH_TIMEOUT=${PROFILE_BENCH_TIMEOUT:-180}
ROOT=$RESULT_ROOT/$EXP
PID=

cleanup() {
  RC=$?
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    kill -TERM "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
  fi
  ixsmi > "$ROOT/gpu_after.txt" 2>&1 || true
  exit "$RC"
}
trap cleanup EXIT

if [ -f "$BI_ENV" ]; then
  source "$BI_ENV"
fi
if curl -sf http://127.0.0.1:9031/v1/models >/dev/null 2>&1; then
  echo "port 9031 already serves a model; stop it before starting $EXP" >&2
  exit 1
fi
mkdir -p "$ROOT"
START=$(date +%s)
{
  echo "exp_id=$EXP"
  echo "date=$(date -Iseconds)"
  echo "case=$CASE"
  echo "plugin_root=$PLUGIN_ROOT"
  echo "plugin_commit=$(git -C "$PLUGIN_ROOT" rev-parse HEAD)"
  echo "plugin_branch=$(git -C "$PLUGIN_ROOT" branch --show-current)"
  echo "plugin_diff=$(git -C "$PLUGIN_ROOT" diff --shortstat)"
  echo "gems_root=$GEMS_ROOT"
  echo "gems_commit=$(git -C "$GEMS_ROOT" rev-parse HEAD)"
  echo "gems_branch=$(git -C "$GEMS_ROOT" branch --show-current)"
  echo "gems_diff=$(git -C "$GEMS_ROOT" diff --shortstat)"
  echo "ixsys=$IXSYS delay=$IXSYS_DELAY limit=$IXSYS_LIMIT"
  echo "profile_only=$PROFILE_ONLY bench_timeout=$PROFILE_BENCH_TIMEOUT"
  echo "uname=$(uname -a)"
  env | grep -E '^VLLM_FL_|^VLLM_PLUGINS|^PYTHONPATH|^MACA_|^LD_LIBRARY_PATH' | sort
} > "$ROOT/config.txt"
free -h > "$ROOT/memory_before.txt" 2>&1
ixsmi > "$ROOT/gpu_before.txt" 2>&1
"$VENV/bin/python" -m pip show vllm vllm-plugin-fl flag_gems torch evalscope \
  > "$ROOT/versions.txt" 2>&1

export VLLM_PLUGINS=fl
export PYTHONPATH=$PLUGIN_ROOT:$GEMS_ROOT/src${PYTHONPATH:+:$PYTHONPATH}
{
  echo "VLLM_PLUGINS=$VLLM_PLUGINS"
  echo "PYTHONPATH=$PYTHONPATH"
} >> "$ROOT/config.txt"
SERVE_CMD=(
  "$VENV/bin/python" -m vllm.entrypoints.openai.api_server --model "$MODEL"
  --port 9031 --served-model-name minicpm
  --gpu-memory-utilization 0.85 --max-model-len 131072
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
)
if [ "$IXSYS" = 1 ]; then
  /usr/local/corex/bin/ixsys \
    -d "$IXSYS_DELAY" -l "$IXSYS_LIMIT" \
    -t cuda_api,cuda_device -o "$ROOT/serve.ixsys" \
    --print-gpu-summary --print-api-summary \
    "${SERVE_CMD[@]}" > "$ROOT/serve.log" 2>&1 &
else
  "${SERVE_CMD[@]}" > "$ROOT/serve.log" 2>&1 &
fi
PID=$!
echo "$PID" > "$ROOT/serve.pid"
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
  echo "serve not ready after 1200 seconds" >&2
  exit 1
fi
echo "startup_seconds=$(( $(date +%s) - START ))" > "$ROOT/startup.txt"

cd "$ROOT"
BENCH_TIMEOUT=3600
if [ "$PROFILE_ONLY" = 1 ]; then
  BENCH_TIMEOUT=$PROFILE_BENCH_TIMEOUT
fi
timeout "$BENCH_TIMEOUT" "$VENV/bin/python" \
  "$PLUGIN_ROOT/benchmarks/benchmark_throughput_serve.py" \
  --model "$MODEL" --served-model-name minicpm --port 9031 \
  --test-cases "$CASE" > "$ROOT/bench.log" 2>&1
RC=$?
if [ "$PROFILE_ONLY" = 1 ] && [ "$RC" = 124 ]; then
  RC=0
fi
echo "$RC" > "$ROOT/benchmark.rc"
SUMMARY=$(ls -1t benchmark_results/summary_*.csv 2>/dev/null | head -1 || true)
if [ -n "$SUMMARY" ]; then
  cp "$SUMMARY" "$ROOT/summary.csv"
fi
exit "$RC"
