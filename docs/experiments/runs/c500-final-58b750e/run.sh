#!/bin/bash
export PATH=/opt/conda/bin:$PATH
export MACA_PATH=/opt/maca MACA_HOME=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/lib64
export VLLM_PLUGINS=fl
export PYTHONPATH=/workspace/vllm-plugin-FL:/workspace/FlagGems/src
unset VLLM_FL_MM_WARMUP
cd /workspace/evidence/final-58b750e

{ echo "date=$(date -Iseconds)"; echo "commit=vllm-plugin-FL 58b750e"; echo "gems=$(git -C /workspace/FlagGems rev-parse HEAD)";
  echo "plugin_patch_md5=$(md5sum /workspace/vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/metax/patches/mm_autotune_warmup.py | cut -d' ' -f1)";
  echo "serve_command=vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072";
  echo "env_VLLM_FL_MM_WARMUP=<unset>"; mx-smi | head -20; free -g; } > manifest.txt 2>&1

pkill -f "vllm serve" 2>/dev/null; sleep 5; pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 2
rm -f /root/.flaggems/config_cache/*.db
rm -rf /root/.triton

T0=$(date +%s)
nohup vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 > serve.log 2>&1 &
echo $! > serve.pid
for i in $(seq 1 1800); do
  grep -q "Application startup complete" serve.log 2>/dev/null && { T1=$(date +%s); echo "startup_seconds=$((T1-T0))" > timing.txt; break; }
  sleep 2
done

cd /workspace/vllm-plugin-FL
python3 benchmarks/benchmark_throughput_serve.py --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm --port 9031 --test-cases '[[4096,1024,64,256]]' > /workspace/evidence/final-58b750e/bench-4k.log 2>&1
echo "bench4k_exit=$?" >> /workspace/evidence/final-58b750e/timing.txt

python3 benchmarks/benchmark_throughput_serve.py --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm --port 9031 --test-cases '[[16384,1024,64,128]]' > /workspace/evidence/final-58b750e/bench-16k.log 2>&1
echo "bench16k_exit=$?" >> /workspace/evidence/final-58b750e/timing.txt

cd /workspace/evidence/final-58b750e
evalscope eval --model minicpm --api-url http://127.0.0.1:9031/v1/chat/completions --api-key EMPTY \
  --eval-type openai_api --datasets math_500 \
  --dataset-args '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}' \
  --eval-batch-size 8 --timeout 3600 \
  --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
  --work-dir /workspace/evalscope-datasets/level3-final-58b750e --ignore-errors > eval-level3.log 2>&1
echo "eval_exit=$?" >> timing.txt
cp -a /root/.flaggems/config_cache/*.db /workspace/evidence/final-58b750e/db-after-all.db 2>/dev/null
echo done >> timing.txt
