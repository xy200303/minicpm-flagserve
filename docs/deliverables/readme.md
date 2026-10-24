# XLANG2026 ｜ FlagOS 赛道二提交说明

本包对应 `vllm-plugin-FL/` 与 `FlagGems/` 两个目录。基线为
`flagos-ai/vllm-plugin-FL` 分支 `flagos-2026-s2` 与 `FlagGems` tag `v5.3.5`。
**所有改动均在框架/算子内完成，`vllm serve` 启动参数与官方基线完全一致。**

## 0. 环境要求与启动耗时

| 项目 | 天数 BI-V150 | 沐曦 C500 |
|---|---|---|
| 显存 | 32 GB（本队开发机） | 64 GB |
| **CPU 内存** | ≥ 64 GB | **≥ 128 GB**（见下） |
| 启动命令中的 `--compilation-config` | `{"cudagraph_mode": "FULL_DECODE_ONLY"}` | **不要加**（与官方一致） |

**C500 内存说明（实测，非官方约束）**：在模力方舟 **64 GB CPU 内存**规格实例上，
**纯镜像原版** `vllm serve` 也会在 inductor 编译阶段被内核 OOM killer 杀掉
（`memory.events` 的 `oom_kill` 逐次 +1，匿名内存峰值 63.4 GB，
被杀时 GPU 显存仅 858 MiB）。128 GB 规格实例全程稳定。
**这不是代码问题**，与负载、vllm 版本、我们的改动均无关。

**启动耗时**：C500 首次启动需要执行 GEMM autotune 预热
（5 个权重形状 × 120 次 `torch.mm` 调用）。实测两台实例分别为
**367 秒与 973 秒**。第二次启动（triton cache 已存在）约 1–3 分钟。
**请耐心等待，不要误判为卡死**。如不需要预热可设 `VLLM_FL_MM_WARMUP=0`，
但 4k 吞吐会下降约 6%、TTFT 变差。

## 1. 性能结果

官方用例 `[input_len, output_len, concurrency, num_prompts]`，
`vllm bench serve` + 官方聚合口径，无计时埋点、冷缓存。

| 平台 | 场景 | 官网基线 | 本包实测 | 提升 |
|---|---|---|---|---|
| C500 | 4k `[4096,1024,64,256]` | 5089.645 | **11868.79 – 12403.21** | **2.33x – 2.44x** |
| C500 | 16k `[16384,1024,64,128]` | 7029.675 | **11098.02 – 11167.87** | **1.58x – 1.59x** |
| BI-V150 | 4k | 2028.01 | **2606.56** | **+28.5%** |
| BI-V150 | 16k | 915.15 | **3541.34** | **+287%** |

C500 数字给的是**两台不同实例**的实测区间（实例间存在真实差异：
4k 吞吐差 4%、TTFT 差 66%，16k 几乎一致）。
天数一栏为 **A20 交付栈**实测；其同日交错 A/B 为
**2239.43 → 2606.56（+16.4%）** 与 **2956.59 → 3541.34（+19.8%）**，
即 A20 在 A7 栈之上再带来的增量（A7 栈同日最佳为 3778.65 / 3567.33，
与 A20 不同日，不直接可比）。

**精度（MATH-500 Level 3，evalscope 官方原命令，105 题）**：
C500 **97.14% – 98.1%**；BI-V150 **96.2% – 98.1%**。均 ≥ 0.95。

**TTFT**：C500 4k 2965 ms（基线 3199 ms）、16k 17902 ms（基线 27197 ms）；
天数 A20 4k 113.6 → 95.5 ms（TPOT）、16k TTFT 155.5 → 130.1 s。

## 2. 天数 BI-V150 改动

### A1/A6：prefill attention 路由到 vendor ixAttnBkd

`vllm_fl/dispatch/backends/vendor/iluvatar/impl/vendor_flash_attn.py`

builder 一次性算出每步 decode/prefill 划分（避免每层 2 次 D2H）；
prefill 走 `ixformer.flash_attn_varlen_func`，KV 用 Triton `_dual_gather_kernel`
一次取出页内连续 K/V（比两次 `index_select` GPU 106→62 µs、host 68→37 µs）。
开关 `VLLM_FL_DISABLE_ILUVATAR_VENDOR_FA=1`。

### A7：decode unified-attention 发射配置 pin

`impl/decode_attn_pin.py`：仅对 3D 路径 pin `num_warps=1 / num_stages=1`，
输出与 stock **bit-exact**。开关 `VLLM_FL_DISABLE_DECODE_PIN=1`。

### A20：自研 paged decode attention 内核（本轮最大单项收益）

算子本体在 FlagGems：`FlagGems/src/flag_gems/fused/paged_decode_attention.py`
（Triton 实现，已在 `fused/__init__.py` 注册）。插件侧为薄封装与派发：
`impl/decode_attn_fast.py` + `vendor_flash_attn.py` / `iluvatar.py` 两处 hook。

- 一个 CTA = 1 warp = 一个 `(seq, kv_head, 段)`：`num_warps=1` 让 PV 的
  MMA 累加器→操作数布局转换留在 warp 内；
- `BLOCK_N=16`；base-2 在线 softmax（`scale` 预折 `log2(e)`）；
- 主循环无 mask，仅 ragged 尾部单独带 mask 的 block；
- **段数固定、段长由 device 端 `seq_lens` 现算** → CUDA-graph 安全；
  段缓存 slab 在 `apply()`（捕获前）一次性预分配；
- 段边界按整页对齐（`block_n == block_size` 时 page 变标量，
  去掉逐 token 除模与向量页 gather，再快 7–10%）；
- 混合步（prefill+decode 混批）的 decode 行也走本内核（`try_fast_decode`）。

开关：`VLLM_FL_DISABLE_DECODE_FAST=1` 整体回退；
`VLLM_FL_DECODE_FAST_SEGMENTS/_BLOCK/_WARPS/_STAGES` 调形状；
`VLLM_FL_DECODE_FAST_DEBUG=1` 打印不合格原因。

### 其它

- A3：builder 的 host 隐式同步修复；
- A4：GEMM pinned-config（libtuner 误选）；
- A5：采样链融合（跨平台移植；本 benchmark 的 top_k 路径不适用，自动退回）。

## 3. 沐曦 C500 改动（opt1–opt11）

代码位于 `vllm_fl/dispatch/backends/vendor/metax/` 与
`FlagGems/src/flag_gems/`（`fused/`、`runtime/backend/_metax/`）。

| 项 | 内容 |
|---|---|
| opt1/opt3 | **sort-free 采样链**：bf16 直读 + 温度折叠的 Gumbel-Max `argmax(x/T+g)`，<br>替换 eager PyTorch 的 fp32 全词表采样。`FlagGems/fused/sampler_fused.py` |
| opt7/opt11 | **top-p 直方图两轮缩放**：256→1024 bin 质量直方图求 top-p 阈值，<br>与 mask 采样一趟完成；`fused/top_p_sample.py`。<br>消除每步约 1.6 ms 的 `_top_k_top_p_kernel` 与 lm_head 的冗余 cast |
| opt2/opt4/opt6 | **nn 布局权重 + double-buffered skinny-M GEMM**：<br>权重一次性转置为 `[K,N]`，decode 细瘦 M 走 `nn_db`，<br>大 M prefill 走 vendor；`linear_nn_repack.py` |
| opt5 | **autotune 预热**：模型加载后按 (M,K,N) 触发一次，<br>避免运行时 autotune 抖动（4k +5.9%，见第 0 节启动耗时） |
| opt8 | **prefill GEMM 直调 vendor mcBLAS**：`patches/mcblas_mm.py` 用 ctypes<br>直绑 `libmcblas` 的 `mcblasGemmEx`（cuBLAS 兼容 ABI），<br>注册为 custom op 保证 compiled 区域安全 |
| opt10 | **消除 prefill 每层 DtoH 同步**：attention metadata 改由 CPU 侧<br>一次性构建，消除每层两次隐式同步（4k +10.9%、16k +9.9%） |
| opt11 | lm_head nn 布局路由 + top-p zoom 单轮 1024-bin 定稿 |

## 4. 复现

### 4.1 安装

```bash
cd vllm-plugin-FL && pip install -e . --no-build-isolation
cd ../FlagGems && pip install -e . --no-build-isolation
```

### 4.2 环境变量

天数（Iluvatar）无需额外变量。C500 需要：

```bash
export PATH=/opt/conda/bin:$PATH
export MACA_PATH=/opt/maca
export MACA_HOME=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/lib64
export VLLM_PLUGINS=fl
```

### 4.3 启动（与官方一致，一字未改）

```bash
# 天数
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'

# 沐曦 C500（不要加 --compilation-config）
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072
```

### 4.4 性能评测

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

**注意**：本 benchmark 使用 `seed=0`，**同一 serve 上连跑两次相同
`input_len` 的负载会命中 prefix cache 导致吞吐虚高**。每个数据点都应
在新启动的 serve 上获取，或使用不同 `input_len`。

### 4.5 精度评测

```bash
evalscope eval --model minicpm \
  --api-url http://127.0.0.1:9031/v1/chat/completions --api-key EMPTY \
  --eval-type openai_api --datasets math_500 \
  --dataset-args '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}' \
  --eval-batch-size 8 --timeout 3600 \
  --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
  --work-dir /workspace/evalscope-datasets/level3 --ignore-errors
```

结果在 `reports/minicpm/math_500.json` 的 `score` 字段。

## 5. 已评估未采纳（负面结果，均留档）

以下均在同一实例上做当日 A/B 后被否决，**没有进入提交代码**：

- **W8 权重量化（A22/A27）**：修正实现后隔离测量确有 1.82x，
  但接进引擎端到端 −19%（4k 2080 vs 2571），TTFT 恶化 4.4 倍，已回退。
- **RMSNorm IR 级融合（A21）**：−5.9%，TTFT 翻倍，已回退。
- **shape-aware NSEG 分派（A26）**：官方 c64 下当前 NSEG=32 已最优，
  且 kv=16k 已跑到卡带宽 88%，收益上限 1.4%，不实现。
- **n-gram 投机采样（A10–A13、A23/A24）**：draft 质量完美（接受长度 6.00），
  但官方 c64 下混合批膨胀导致 prefill 饥饿，5 分钟仅完成 3/256 请求。
- **`num_splits` 调优（C500）**：微基准 3 倍差异，真实服务冷缓存无收益。
- **图内 CUDA event 计时**：`invalid resource handle`，不可用于本栈。

逐项实验记录（配置、serve/bench 日志、summary、机器状态、结论）见
`xy200303/minicpm-flagserve` 仓库的 `docs/experiments/` 与
`docs/baselines/`。
