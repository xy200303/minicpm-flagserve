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

## A1：prefill 走 vendor ixAttnBkd flash（2026-10-10）

| 场景 | 基线 | A1 | Δ |
|---|---|---|---|
| 4k | 1902.15 | 1866.06 | -1.9%（kv<8192 门控走原路，噪声内） |
| 16k | 861.98 | **2461.59** | **+185.6%** |

16k TTFT 636.5s→194.5s；精度 MATH-500 L3 = **98.1%**（无损）。
详见 `a1-iluvatar-vendor-fa.md`（探针矩阵、cmt 语义、三个连环坑：
per-layer tolist 同步 / 分配器风暴 / 4k 净亏损与阈值门控）。

下一步：decode 侧仍是 stock Triton attention（16k decode TPOT ~150ms/步
量级），参照 A1 的 vendor op 路线继续。
