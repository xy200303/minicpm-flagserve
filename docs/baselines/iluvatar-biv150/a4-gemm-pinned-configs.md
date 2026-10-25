# 天数 BI-V150 A4：GEMM pinned-config（杀掉 libtuner 误选）

日期：2026-10-13 ｜ 实例：FlagOS OnlineLab BI-V150（32GB 显存版）

## 起因

A3 的附带发现：gems `mm_kernel` 在 serve 里跑得比离线最优慢
（qkv 303µs vs 最优 265µs）。排查确认根因不是内核本身，而是
FlagGems libtuner 的运行时 autotune 在 event-timing 降级模式下
选错配置。

## 方法：离线全空间扫描 → 派发层 pin 配置

对 MiniCPM5-2B serve 的全部 8 个 GEMM 形状做离线全空间扫描
（脚本 tmp/mm_sweep.py，结果 /tmp/mm_sweep_result.json），
对比 vendor `gemm_tcu` 与 gems `mm_kernel` 各配置：

| 形状 (M×K×N) | vendor µs | gems 最优 µs | pinned 配置 (BM,BN,BK,SK,warps,stages) |
|---|---|---|---|
| prefill qkv 2048×2048×2560 | 285.0 | **265.2** | 128,128,32,1,8,2 |
| prefill o 2048×2048×2048 | 193.0 | 194.2 | 256,256,32,1,16,2 |
| prefill gateup 2048×2048×12288 | **1127.5** | 1285.4 | 256,256,32,1,8,2 |
| prefill down 2048×6144×2048 | 542.2 | 548.1 | 256,256,32,1,16,2 |
| decode64 qkv | 38.1 | **21.0（1.81x）** | 64,128,64,4,8,1 |
| decode64 o | 29.2 | **16.6（1.75x）** | 64,128,64,2,8,2 |
| decode64 gateup | 113.5 | **92.8（1.22x）** | 64,256,64,1,8,1 |
| decode64 down | 64.8 | **50.5（1.28x）** | 64,64,32,4,4,2 |

## 实现（FlagGems `_iluvatar/ops/mm.py`，commit 5eac6b39）

`mm()` 内加 `_pinned_cfg(M,K,N)` 派发：M≤128 走 `_PINNED_DECODE`，
M≥1024 走 `_PINNED_PREFILL`，其余维持 libtuner。prefill 也 pin，
因为 libtuner 运行时误选比 vendor 差距更大（gateup 1416→1285、
down 719→548、o 240→194 µs/层）。

- 绕过 libtuner 的方式：`_raw_mm_kernel = mm_kernel.fn.fn`
  （LibEntry→LibTuner→Heuristics），保留 EVEN_K / UPGRADE*
  heuristics
- SPLIT_K>1 的 decode 配置先 `c.zero_()`（atomic_add 需零初始化；
  M=64 时仅 ~6.5µs）
- 包装在 eager 下 host 开销 ~117µs/call，但 decode 走 cudagraph
  replay（图内只有内核 GPU 时间 ~27.5µs vs vendor 38.1µs），
  不受影响；mixed 步 prefill pin 全是 SK=1，无此问题

## 正确性

单测 8 形状 + M=32/2112 边界全过（rel err ≤ 6.6e-3），重复调用
验证 zero_ 路径不积累垃圾。

## 端到端结果（官方 benchmark，4 跑弃首取后 3 均值）

| 场景 | A3 | A4 | 变化 |
|---|---|---|---|
| 4k total tok/s | 2582.04 | 2579.14 | -0.1%（噪声内） |
| 16k total tok/s | 2488.58 | 2485.52 | -0.1%（噪声内） |

**32GB 上端到端持平**：与 A3 相同，端到端被 decode 步、
尾部 drain 和 KV 池排队支配，GEMM 微基准收益未转化为整网收益。
不引用未实测的其他显存规格推断收益。

精度实测 **97.1%**（MATH-500 L3，105 题，evalscope 同口径），
与 A3 完全一致，≥0.95 达标。SPLIT_K 求和顺序变化对精度无影响。
