#!/bin/bash
# Mixed-step parity: stock vs fast A20 decode-slice path.
set -e
run_one() {
  local tag=$1 extra=$2
  pkill -9 -f "vllm serve" || true; pkill -9 -f "VLLM::EngineCore" || true; sleep 3
  cd /workspace
  env $extra VLLM_PLUGINS=fl nohup vllm serve /workspace/MiniCPM5-2B --port 9031 \
    --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    > /workspace/experiments/runs/parityMix_${tag}.log 2>&1 &
  for i in $(seq 1 300); do
    grep -q "Application startup complete" /workspace/experiments/runs/parityMix_${tag}.log && break
    sleep 2
  done
  python3 /workspace/parity_mixed.py /workspace/experiments/parityMix_${tag}.txt
}
run_one stock "VLLM_FL_DISABLE_DECODE_FAST=1"
run_one fast "VLLM_FL_DECODE_FAST_DEBUG=1"
pkill -9 -f "vllm serve" || true; pkill -9 -f "VLLM::EngineCore" || true
if diff -q /workspace/experiments/parityMix_stock.txt /workspace/experiments/parityMix_fast.txt; then
  echo "MIXED_PARITY_OK"
else
  echo "MIXED_PARITY_DIFF"
  diff /workspace/experiments/parityMix_stock.txt /workspace/experiments/parityMix_fast.txt | head -20
fi
