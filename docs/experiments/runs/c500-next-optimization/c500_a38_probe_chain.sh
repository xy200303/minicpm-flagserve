#!/bin/bash
set -u
source /data/c500-env.sh
for shape in prefill_gateup prefill_qkv prefill_o prefill_down decode64_qkv decode64_o decode64_down; do
  python /data/c500_mcblas_algo_probe.py --shape "$shape" \
    --output "/data/experiments/runs/c500-a38-mcblas-algo/$shape.json" \
    > "/data/experiments/runs/c500-a38-mcblas-algo/$shape.log" 2>&1 || exit 1
done
