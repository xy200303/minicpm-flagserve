#!/bin/bash
set -u
source /data/c500-env.sh
python /data/c500_mcblas_algo_probe.py --shape decode64_qkv --algos -1 106 -1 106 -1 106 --reps 30 \
  --output /data/experiments/runs/c500-a38-mcblas-algo/decode64_qkv-interleaved.json \
  > /data/experiments/runs/c500-a38-mcblas-algo/decode64_qkv-interleaved.log 2>&1 && \
python /data/c500_mcblas_algo_probe.py --shape decode64_o --algos -1 111 -1 111 -1 111 --reps 30 \
  --output /data/experiments/runs/c500-a38-mcblas-algo/decode64_o-interleaved.json \
  > /data/experiments/runs/c500-a38-mcblas-algo/decode64_o-interleaved.log 2>&1
