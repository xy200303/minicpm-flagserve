# MetaX C500 64GB 基线复现（2026-09-22）

评测机：模力方舟 `jiajia-mxc500`（曦云 C500，104 SMs，warp_size 64，cc 8.0，显存 63.6 GiB）。
FlagOS 2.1 镜像：torch 2.8.0+metax3.7.0.7 / vllm 0.24.0+empty / flag_gems 5.3.5 / vllm-plugin-fl 0.3.0rc1+g13eb9be69 / evalscope 1.11.1 / triton 3.0.0+metax3.7.0.7。

## 环境激活（镜像不会自动设置，每条 SSH 命令都要带）

```bash
export PATH=/opt/conda/bin:$PATH
export MACA_PATH=/opt/maca          # triton metax driver 读这个，不是 MACA_HOME
export MACA_HOME=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/lib64
```

## 启动命令（与官方基线一致，一字未改）

```bash
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072
```

## 评测命令（官方原样）

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

## 复现结果 vs 官网基线

| 场景 | 官网基线 total tok/s | 复现（4 跑） | 官网 TTFT ms | 复现 TTFT ms |
|---|---|---|---|---|
| 4k  [4096,1024,64,256]  | 5089.645 | 4440 / 4746 / 5079 / 5104（均值 4763，summary 口径 4763.32） | 3199.435 | 3156~3380 |
| 16k [16384,1024,64,128] | 7029.675 | 7024 / 7049 / 7038 / 7043（均值 7038.6，summary 口径 7043.35） | 27197.135 | 27040~27201 |

- 16k 场景极其稳定（±0.3%），与官网基线 +0.13% 完美对齐——prefill 主导，GPU-bound。
- 4k 场景波动 ±7%：decode 占比高，采样器走 eager PyTorch（vLLM Triton top-k/top-p 在 metax 被补丁禁用），
  弱 CPU（i7-8550U 12 核）抖动直接传导到 TPOT（Mean TPOT 59~69ms 间摆动，Median 恒 ~61.5ms）。
  官网 5089.645 应为其较好一次的成绩。
- 原始数据：`raw_runs_baseline.csv`（脚本每场景连跑 4 次）、`summary_baseline.csv`（脚本聚合口径）。

## 已确认的实现路径（metax.yaml 调度）

RMSNorm / RoPE / SiLU-mul → FlagGems Triton；attention → MetaX vendor flash_attn；
sampler → eager PyTorch（被补丁禁用 Triton 版）；lm_head GEMM → torch native。
精度基线待 evalscope math_500 Level 3 结果（官方基线 0.962，要求 ≥0.95）。
