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

## 9. 混合步（prefill+decode 混批）里的 decode 行

`vendor_flash_attn` 会把混合步拆成 prefill 行（vendor FA）与 decode 行；**decode 行原先仍走 stock
unified_attention**。16k 用例绝大多数步都是混合步（每波 prefill 154 s），所以我们的 kernel 只覆盖了
少数纯 decode 步——这正是 16k 端到端 −1.9% 的主因（decode 变快、但调度与 TTFT 反而劣化）。

改法：在 `_vendor_forward` 的 decode 分支里先调 `decode_attn_fast.try_fast_decode(...)`，
形状不合格再退回 stock。`dec_meta` 里的 `seq_lens` / `block_table` / `max_query_len=1`
正好就是自研 kernel 需要的入参，无需额外同步。

效果（16k，同日）：

| 版本 | 总吞吐 tok/s | TTFT s | TPOT ms |
|---|---|---|---|
| stock（A1/A2） | 2992.51 / 2979.55 | 154.5 / 154.7 | 151.6 / 152.2 |
| fast，无混合步路径（B1/B2） | 2923.56 / 2933.99 | 158.1 / 158.2 | 156.0 / 154.9 |
| fast，含混合步路径（B3/B4） | 2973.27 / **3454.11** | 154.2 / 133.5 | 153.3 / 129.2 |

4k 不受影响：B3（最终代码）2533.86 / TPOT 98.84，与 B1/B2（2538.21 / 2563.13）同档。

## 10. 否决的两条路线（有测量，勿重试）

### 10.1 int8 KV cache（per-token-head scale）

动机：4k decode attention 已压到 KV 读带宽下限（11.3 GB/步），KV 减半等于直接再省 ~8 ms/步；
16k 还能把 KV 池从 32 路扩到 64 路。

实测（`dattn_q8.py`，独立 kernel，同形状）：

| 版本（kv=4096） | ms/层 | 有效带宽 |
|---|---|---|
| bf16 KV（现交付） | **0.72** | 372 GB/s |
| int8 KV（load + fp32 反量化 + bf16） | 1.11 | 121 GB/s |
| int8 KV（只 load + cast，**无反量化**） | 1.06 | 126 GB/s |
| int8 KV（bf16 乘法反量化） | 1.06 | 126 GB/s |

kv=16384 同向（2.41 → 3.63 最好）。**只加载就已经比 bf16 慢**，说明瓶颈不是反量化 ALU，
而是这套 Triton 后端对 1 字节 gather 的向量化/吞吐只有 2 字节的 1/3（每列仅 128 B 连续）。
→ int8（以及同样 1 字节的 fp8）KV 在 BI-V150 上不成立；除非把 int8 成对当作 int16 加载再拆包
（需要沿 D 轴 interleave，代价未测）。

### 10.2 vendor 的 paged prefill（`ixformer.batch_paged_attention`）

如果它能直接吃 paged KV，16k prefill 就能省掉每层每步的反分页 gather（约 22 GB 读 + 22 GB 写/波）。
实测该 API 的三条硬约束：

1. kv cache 必须是 `(K, V)` 两个 **3 维** `[total_entries, kv_heads, head_size]`；
2. 第一维必须**恰好等于** `max_page_num * page_size`——vLLM 的全局池（32603 页）与
   `block_tables` 的宽度（按 max_model_len 的 8192）对不上；
3. 走 tensor-desc 路径后直接 `CUINFER_STATUS_BAD_PARAM`（无文档，无法定位参数）。

→ 不可用。16k prefill 的下一步只能靠"减少 KV 重读"（chunk 数固定，不可调）或换布局，
本轮到此为止。

## 11. 硬件事实：sm_71 没有 bf16 tensor core

`torch.cuda.get_device_properties(0)` 报 **compute capability 7.1**（Volta 世代，
toolchain 是 CoreX CUDA 10.2）。sm_71 只有 fp16 的 HMMA（m8n8k4 / WMMA
m16n16k16），**没有 bf16 MMA**。所以：

- Triton 的 `tl.dot(bf16, bf16)` 在这张卡上必然要经过 bf16→fp16 的转换与
  fp32 累加器→操作数的布局搬运——这正是 PV 那 0.19 ms 的来源（见 §1/§10）；
- 手写 CUDA 也不会凭空获得 bf16 MMA，只能改用 fp16 HMMA + 显式的
  double-buffer，收益上限是"把 PV 的布局搬运省掉"，不是换一种算力。

顺带否掉一条便宜的猜测：**fp16 KV cache 不会更快**。同一 kernel、同形状，
bf16 K/V 0.698 ms/层、fp16 K/V 0.699 ms/层（对 stock 的 maxdiff 均为 0），
说明瓶颈不在 cache 的 dtype 转换上，换 fp16 cache 属于无收益的框架改造。

## 12. 混合步 engage 率（可审计）

`try_fast_decode` 现在每 4000 次调用打一行 INFO：
`decode_attn_fast slice: used=X skipped=Y`。16k 实测：

| run | 结果 | engage |
|---|---|---|
| A20_16k_B11 | 3430.62 tok/s | used=44000, skipped=0 |
| A20_16k_B12 | 3435.33 tok/s | used=44000, skipped=0 |

即混合步的 decode 切片 **100% 走自研 kernel**，不存在"偶尔退回 stock"的情况。
早前两次落在 stock 档的 16k B 跑（B3 2973 / B7 2929）在没有任何
fallback/not-eligible 日志的情况下与 stock 完全同档（连 TTFT 都是 154–159 s），
只能归因于这台共享实例的外部干扰——与台账里记录的 38–60% 跨时段漂移一致。

## 13. 手写 CUDA 的路被镜像堵死

`/usr/local/corex/bin/nvcc` 是**空壳**：打印 banner、返回 0、不产出任何文件
（`-ptx` / `-c` / `-shared` 都一样，见 `docs/experiments/a20/cuda_spike_run.py`）。
因此"把 PV 从 Triton 里拿出来手写 CUDA"在本实验室镜像里无法开发与验证。
即便能编，BI-V150 是 sm_71（无 bf16 tensor core），收益上限也只是把
0.19 ms/层的布局搬运省掉，不是换算力。
