#!/bin/bash
set -o pipefail
export PYTHONPATH=/workspace/vllm-plugin-image-align:/workspace/flag-image-align/src
export VLLM_PLUGINS=fl
cd /workspace/image-align-runs
python3 /workspace/vllm-plugin-image-align/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256]]'
echo $? > /workspace/image-align-runs/bench4k2.rc
