# A20：BI-V150 自研 paged decode attention kernel

> 目标：把官方 4k/16k 用例里 decode 步的 attention 从"stock Triton unified attention"
> 换成面向 BI-V150（16 SM、128 KB smem、warp size 64）手写的分块 KV 扫描内核。
> 交付：`vllm_fl/dispatch/backends/vendor/iluvatar/impl/decode_attn_fast.py`
> （挂在 `TritonAttentionImpl.forward` 上，`VLLM_FL_DISABLE_DECODE_FAST=1` 可整体回退）。

## 1. 为什么要重写

微基准（隔离、`B=64`、`kv=4096`、16 q / 2 kv、`d=128`、`block=16`、bf16，20 次取均值）：

| 路径 | ms/层 | 有效带宽 |
|---|---|---|
| stock 2D（Triton 默认 launch） | 2.30 | 117 GB/s |
| stock 3D（默认 launch） | 1.75 | 153 GB/s |
| stock 3D + A7 pin（w1s1，交付态） | 1.04 | 258 GB/s |
| 硬件 copy 上限（1 GB copy） | — | **587 GB/s** |
| 自研 kernel（本文件） | **0.72** | **372 GB/s** |

把 stock 3D kernel 逐段拆开看（同一 harness，只改被测 kernel 版本）：

| 版本 | ms/层 | 说明 |
|---|---|---|
| 只加载 K/V | 0.52 | 509 GB/s，已接近带宽上限 |
| + QK `tl.dot` | 0.53 | 几乎免费 |
| + 在线 softmax | 0.54 | 几乎免费 |
| + PV `tl.dot` | **0.90** | **+0.36 ms，唯一的大头** |

→ 结论：瓶颈不是带宽、不是 softmax，而是 **PV 那个 `tl.dot` 的
"MMA 累加器布局 → A 操作数布局"转换**（每个 KV block 都要过一次共享内存往返）。

## 2. kernel 设计

- **一个 CTA = 1 warp = 一个 (seq, kv_head, segment)**。`num_warps=1` 让 p 的布局转换
  留在 warp 内（实测 w2 慢 16%，w4/w8 更差）。
- **BLOCK_N=16**（KV tile 只取 16 个 token），让 p tile 不超过一个 MMA 形状；
  32/64 反而慢 12–30%。
- **base-2 在线 softmax**：`scale *= log2(e)` 预先折算，`exp2` 走 MUFU；
  用 `exp` 慢约 10%。
- **主循环不加 mask、只有 ragged 尾部单独一个带 mask 的 block**：
  每块都 mask 会从 0.72 掉到 0.98 ms。
- **段数固定、段长在 device 端由 `seq_lens` 现算**（`seg_len = ceil(kvlen/NSEG)`），
  与 stock 的 `num_par_softmax_segments` 同构。见 §4 的踩坑。
- 两段式：`_segment_kernel` 写每段 fp32 的 (m, l, acc)，`_combine_kernel` 在 base-2 空间合并。

正确性（对 stock kernel，同一批随机 q/k + 随机 block table）：

| 用例 | 自研 | stock | max&#124;diff&#124; |
|---|---|---|---|
| kv=4096 等长 | 0.716 ms | 1.039 ms | 2.4e-4 |
| ragged（4096±1500） | 0.663 ms | 0.900 ms | 9.8e-4 |
| kv=77（短） | 0.120 ms | 0.183 ms | 3.9e-3 |

## 3. 已否决的方案（不要再试）

| 方案 | 结果 |
|---|---|
| 双 KV head 打包进一次 `M=16` dot（列 = token×head，-inf 掩掉跨头块） | 1.18 ms，比 v1 慢；计算量翻倍吃掉收益 |
| PV 走 `tl.sum(p[:,:,None] * v[None])`（不用 MMA） | 1.69 ms，寄存器压力爆炸 |
| PV 转置成 `dot(vtT[D,BN], pT[BN,GQA])`（M=128） | 0.85 ms，`tl.trans(p)` 抵消收益 |
| `tl.dot(..., acc=...)` 融合累加 | 0.82 ms，无差别 |
| `num_stages=2/3` | 一致更慢（smem 压力） |
| 更大 BLOCK_N（32/64/128） | 一致更慢 |

## 4. CUDA graph 踩坑（两次）

1. **段数不能依赖 host 侧 kv 长度**。第一版 `nseg = ceil(max_seq_len/256)`：
   图捕获时 vLLM 的 `max_seq_len` 是 `max_model_len=131072` → nseg=512，超过上限被拒绝，
   **35 张图里全是 stock kernel**，运行时重放当然没效果（A/B 显示 2336 vs 2339，完全无变化）。
   改成固定 `NSEG=32`、段长由 `seq_lens` 在 device 端现算后，捕获就正常了。
2. **捕获期间不能分配显存**。第一版在每次调用里 `torch.empty` 分配段缓存，
   图捕获直接卡死（第一个 graph 停 2.5 min 不动）。改成在 `apply()`（捕获之前）
   一次性分配 BMAX×HKV×NSEG 的 slab（146 MiB）复用。

排查手段：`VLLM_FL_DECODE_FAST_DEBUG=1` 会按"首次原因"打印一次不合格理由
（`not_decode` / `kv_too_long` / `batch` / `causal` …），这次就是靠它定位到
捕获期被 `kv_too_long` 拒掉的。

## 5. 官方 A/B

见 `docs/experiments/tracking.md` 的 A20 小节（当日交错 A/B，4k 与 16k 各 A/B 两次）。

## 6. 引擎内结果（官方用例，同日交错）

### 4k `[[4096,1024,64,256]]`

| | 总吞吐 tok/s | 输出 tok/s | TTFT ms | TPOT ms |
|---|---|---|---|---|
| stock（A1/A2） | 2331.37 / 2326.89 | 466.27 / 465.38 | 5549 / 5576 | 110.58 / 110.70 |
| fast（B1/B2） | 2538.21 / 2563.13 | 507.64 / 512.63 | 5587 / 5547 | 98.92 / 98.37 |
| 中位数 | 2329.13 → **2550.67（+9.5%）** | +10.0% | 噪声内 | 110.64 → **98.65（−10.8%）** |

### 16k `[[16384,1024,64,128]]`

| | 总吞吐 tok/s | TTFT ms | TPOT ms |
|---|---|---|---|
| stock（A1/A2） | 2992.51 / 2979.55 | 154469 / 154749 | 151.55 / 152.15 |
| fast（B1/B2） | 2923.56 / 2933.99 | 158083 / 158160 | 156.03 / 154.94 |
| 中位数 | 2986.03 → 2928.78（**−1.9%**） | +2.3% | +2.1% |

但纯 decode 窗口（同为 31 个在跑请求）：**359.5 → 505.3 tok/s（+40.6%）**，decode 步 86 → 61 ms；
隔离微基准 kv=16384 也是 stock 3.50 → 自研 2.41 ms/层。16k 端到端为负是因为它是
prefill/容量受限（每波 TTFT 154 s，KV 池只够 32 路），decode 只占整轮 ~20% 时间。

### NSEG 扫描（隔离，B=64）

| kv | nseg=16 | 32 | 64 | 128 | stock |
|---|---|---|---|---|---|
| 4096 | **0.704** | 0.765 | — | — | 1.039 |
| 16384 | 2.457 | **2.414** | 2.544 | 2.940 | 3.497 |

`tl.arange` 要求 NSEG 为 2 的幂（48 会编译失败）。取 32 是两档的折中。

## 7. 第三个坑：scratch slab 会吃掉 KV 池

16k 用例的 KV 池只够 32 路并发，第一版 slab 按 `BMAX=256` 分配（146 MiB），
相当于少了约 1% 的 KV 槽位。改成按官方并发 `BMAX=64`（36 MiB，解码批量超过 64 时回退 stock）
后 16k 从 −2.3% 回到 −1.9%——说明容量影响存在但不是主因，主因是 16k 的 prefill 占比。

## 8. 精度

MATH-500 Level 3（同基线参数，105 题）：**97.1%**（基线 96.2%），达标 ≥95%。
greedy 对拍在长 filler prompt 上会分叉（数值噪声翻转 argmax），但 MATH-500 上没有体现损失。
