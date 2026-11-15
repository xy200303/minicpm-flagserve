#!/bin/bash
export PATH=/opt/conda/bin:$PATH
export MACA_PATH=/opt/maca MACA_HOME=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/lib64
export VLLM_PLUGINS=fl
export PYTHONPATH=/workspace/vllm-plugin-FL:/workspace/FlagGems/src
unset VLLM_FL_MM_WARMUP
export VLLM_FL_NN_MIN_M=0
export VLLM_FL_FREE_NT_WEIGHTS=1
cd /workspace/evidence/ab-single-layout
pkill -f "vllm serve" 2>/dev/null; sleep 6; pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 2
T0=$(date +%s)
nohup vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 > serve.log 2>&1 &
for i in $(seq 1 1800); do
  grep -q "Application startup complete" serve.log 2>/dev/null && { T1=$(date +%s); echo "startup_seconds=$((T1-T0))" > timing.txt; break; }
  sleep 2
done
grep -E "GPU KV cache size|Maximum concurrency" serve.log >> timing.txt
cd /workspace/vllm-plugin-FL
python3 benchmarks/benchmark_throughput_serve.py --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm --port 9031 --test-cases '[[16384,1024,64,128]]' > /workspace/evidence/ab-single-layout/bench-16k.log 2>&1
echo "bench16k_exit=$?" >> /workspace/evidence/ab-single-layout/timing.txt
python3 benchmarks/benchmark_throughput_serve.py --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm --port 9031 --test-cases '[[4096,1024,64,256]]' > /workspace/evidence/ab-single-layout/bench-4k.log 2>&1
echo "bench4k_exit=$?" >> /workspace/evidence/ab-single-layout/timing.txt
echo done >> /workspace/evidence/ab-single-layout/timing.txt
