# XLANG2026 ｜ FlagOS 赛道二提交说明（vllm-plugin-FL 改动）

本文档对应提交包里的 `vllm-plugin-FL/` 目录。基线是 `flagos-ai/vllm-plugin-FL`
分支 `flagos-2026-s2`；本目录在其基础上叠加了以下改动，**全部在插件内**，
`vllm serve` 参数与官方基线完全一致。

## 1. 天数 BI-V150（Iluvatar）

### A1/A6：prefill attention 路由到 vendor ixAttnBkd

`vllm_fl/dispatch/backends/vendor/iluvatar/impl/vendor_flash_attn.py`

- builder 里一次性算出每步的 decode/prefill 请求划分（避免每层 2 次 D2H）；
- prefill 行走 `ixformer.flash_attn_varlen_func`，KV 用 Triton `_dual_gather_kernel`
  一次性把页内 K/V 连续取出（比两次 `index_select` GPU 106→62 µs、host 68→37 µs）；
- 开关：`VLLM_FL_DISABLE_ILUVATAR_VENDOR_FA=1` 回到 stock。
- 官方 4k 贡献 **+38%**（关掉后 2341→1694 tok/s，TTFT 5.5→14.0 s）。

### A7：decode unified-attention 发射配置 pin

`vllm_fl/dispatch/backends/vendor/iluvatar/impl/decode_attn_pin.py`

只对 3D（flash-decoding）路径 pin `num_warps=1 / num_stages=1`：Triton 的
`w4s3` 默认在这张卡上系统性失配。输出与 stock **bit-exact**。
开关：`VLLM_FL_DISABLE_DECODE_PIN=1`、`VLLM_FL_DECODE_NUM_WARPS/_STAGES`。

### A20：自研 paged decode attention 内核（本轮新增，收益最大）

`vllm_fl/dispatch/backends/vendor/iluvatar/impl/decode_attn_fast.py`（新增文件）
`vendor_flash_attn.py` / `iluvatar.py`（两处 hook）

- 一个 CTA = 1 warp = 一个 (seq, kv_head, 段)：`num_warps=1` 让 `p` 的
  MMA 累加器→操作数布局转换留在 warp 内；
- `BLOCK_N=16`；base-2 在线 softmax（`scale` 预先折算 `log2(e)`）；
- 主循环无 mask，仅 ragged 尾部单独一个带 mask 的 block；
- **段数固定、段长在 device 端由 `seq_lens` 现算** → 可进 CUDA graph；
  段缓存 slab 在 `apply()`（捕获之前）一次性预分配；
- 混合步（prefill+decode 混批）里的 decode 行也走本内核（`try_fast_decode`）。

| | stock | A20 | 变化 |
|---|---|---|---|
| 4k `[4096,1024,64,256]` | 2329.1 | 2538.2 | **+9.0%**，TPOT 110.6→98.8 ms |
| 16k `[16384,1024,64,128]` | 2986.0 | 3445.3 | **+15.4%**，TTFT 154.6→133.8 s |
| MATH-500 Level 3 | 96.2% | **97.1%** | 达标（≥95%） |

（以上为同日交错 A/B，A×2 / B×3–4，取中位数。）

开关：`VLLM_FL_DISABLE_DECODE_FAST=1` 整体回退；`VLLM_FL_DECODE_FAST_SEGMENTS`
/`_BLOCK`/`_WARPS`/`_STAGES` 调形状；`VLLM_FL_DECODE_FAST_DEBUG=1` 打印
不合格原因；`VLLM_FL_DECODE_FAST_SKIP=1` 仅用于步内耗时归因（输出为垃圾值）。

### 其它（A3/A4/A5）

- A3：builder 的 host 隐式同步修复；
- A4：GEMM pinned-config（libtuner 误选）；
- A5：采样链融合（跨平台移植；本 benchmark 的 top_k 路径不适用，自动退回）。

## 2. 沐曦 C500

见 `vllm_fl/dispatch/backends/vendor/metax/` 下的 opt1–opt11（sort-free 采样链、
GEMM 路由、host 同步消除等），详见技术报告第 3 节。

## 3. 复现

```bash
cd vllm-plugin-FL && pip install -e .
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

每个优化的 A/B 开关见上；逐条实验记录（配置、serve/bench 日志、summary、
机器状态、结论）在配套仓库 `xy200303/minicpm-flagserve` 的
`docs/experiments/` 下。
