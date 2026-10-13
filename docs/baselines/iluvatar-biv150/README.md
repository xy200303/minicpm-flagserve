# 天数 BI-V150（Iluvatar，OnlineLab 实例 32GB 版）

赛道二第二张卡的工作区。注意：本实例是 **32GB** 显存版（赛题评测机为 64GB），
KV 池约 52 万 token，16k×64 并发会排队，绝对值低于 64GB 口径；
优化方向与相对收益不受影响（同 silicon）。

## 基线（stock flagos-2026-s2，官方 serve 命令 + FULL_DECODE_ONLY）

| 场景 | total tok/s | TTFT |
|---|---|---|
| 4k [4096,1024,64,256] | 1902.15 | 12.3s |
| 16k [16384,1024,64,128] | 861.98 | 636.5s |

根因：iluvatar 后端没有 flash_attn 包，attention 落到 vLLM 通用 Triton
参考实现 `kernel_unified_attention`——16k prefill 每层每 2048-chunk
36.8ms，占 GPU busy 91.8%。

## A1.5（最终版）：拆掉 kv≥8192 阈值门，全 kv 走 vendor（2026-10-12）

A1 曾因 4k pace A/B 误判加过 kv≥8192 门控；内核级实测 vendor 在
所有 kv 长度赢 12-20 倍，门拆掉后 4k 大涨。

| 场景 | 基线 | A1.5 | Δ |
|---|---|---|---|
| 4k | 1902.15 | **2635.85** | **+38.6%** |
| 16k | 861.98 | **2466.23** | **+186.1%** |

TTFT：4k 12.3s→4.8s；16k 636.5s→193.7s。
精度 MATH-500 L3 = **97.1%**（≥0.95 达标；A1 版测过 98.1%，同一路径
数值语义未变，波动在评测噪声内）。

详见 `a1-iluvatar-vendor-fa.md`（探针矩阵、cmt 语义、三个连环坑：
per-layer tolist 同步 / 分配器风暴 / 阈值门控误判与拆除）。

decode 侧（A2 调研结论，见 `a2-decode-attention-investigation.md`）：
自研 Triton flash-decoding 打平 stock、vendor decode 无 flash-decoding
分段、GEMM vendor≈gems——decode 在 Triton 层面已无可挖，收官。

## A3：host 侧修复（2026-10-13，见 `a3-host-sync-fix.md`）

补抓 A1.5 的 16k 端到端 profile 发现 builder 里 `aten::index` 每步
2 次隐式阻塞式 HtoD+全流同步（占混合步 wall 57%）。修复后混合步
wall 677→353ms、GPU busy 41%→76%；32GB 卡端到端持平（16k 受
decode/尾部/KV 排队支配），64GB 评测机上应有体现。
最终官方 benchmark：4k 2582.04 / 16k 2488.58。
