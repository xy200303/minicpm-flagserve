#!/bin/bash
# Experiment harness: serve -> benchmark -> archive everything, one run per call.
# usage: exp_run.sh <exp_id> <test_case_json> [runs]
# Env vars in the calling shell are recorded and passed to the serve.
set -u
EXP=$1; CASE=$2; RUNS=${3:-1}
DIR=/workspace/experiments/runs/$EXP
mkdir -p "$DIR"
cd /workspace/vllm-plugin-FL
REV=$(git rev-parse --short HEAD)
BR=$(git branch --show-current)
GL=$(git status --porcelain | head -5 | tr '\n' ';')
{
  echo "exp_id=$EXP"
  echo "date=$(date -Iseconds)"
  echo "branch=$BR rev=$REV dirty=$GL"
  echo "case=$CASE runs=$RUNS"
  echo "env:"; env | grep -E "^VLLM_FL_|^VLLM_PLUGINS" | sort
} > "$DIR/config.txt"
ixsmi --query-gpu=name,memory.used,clocks.sm --format=csv > "$DIR/gpu_before.txt" 2>&1
pkill -9 -f "vllm serve"; pkill -9 -f "VLLM::EngineCore"; sleep 3
cd /workspace
VLLM_PLUGINS=fl nohup vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' > "$DIR/serve.log" 2>&1 &
for i in $(seq 1 180); do
  grep -q "Application startup complete" "$DIR/serve.log" && break
  sleep 2
done
if ! grep -q "Application startup complete" "$DIR/serve.log"; then
  echo "SERVE_FAILED" | tee -a "$DIR/meta.txt"; exit 1
fi
RUNS_OVERRIDE=$RUNS python3 /tmp/bench_smoke.py --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm --port 9031 --test-cases "$CASE" > "$DIR/bench.log" 2>&1
CSV=$(grep -o "benchmark_results/summary_[0-9_]*\.csv" "$DIR/bench.log" | tail -1)
[ -n "$CSV" ] && cp "/workspace/$CSV" "$DIR/summary.csv"
ixsmi --query-gpu=name,memory.used,clocks.sm --format=csv > "$DIR/gpu_after.txt" 2>&1
LINE=$(tail -1 "$DIR/summary.csv" 2>/dev/null)
echo -e "$EXP\t$(date +%F) $BR@$REV\t$CASE\truns=$RUNS\t$LINE" >> /workspace/experiments/tracking.tsv
pkill -9 -f "vllm serve"; pkill -9 -f "VLLM::EngineCore"
echo "done: $EXP -> $DIR"
