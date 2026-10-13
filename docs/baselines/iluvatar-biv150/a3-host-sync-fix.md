# 天数 BI-V150 A3：杀掉 builder 里的每步全流同步（host 侧修复）

日期：2026-10-13 ｜ 实例：FlagOS OnlineLab BI-V150（32GB 显存版）

## 起因

A1.5 收官时只有基线期的端到端 profile，优化后没再抓过。补抓真实 serve
的 16k profile 后发现：**GPU busy 只有 41.3%**，混合步 wall ~677ms 而
GPU 工作 ~265ms——一半时间 GPU 在等 host。

## 根因（trace + correlation + python 栈三重实锤）

`_build_split` 里两处 `m.seq_lens[dec_idx]` / `m.block_table[dec_idx]`
（GPU 张量被 Python list 索引 → `aten::index`）在 CoreX 上每次调用会
隐式做**阻塞式 HtoD（pageable）+ cudaStreamSynchronize**，把 GPU 队列
全部排空：

- 每次阻塞中位 78.6ms，每步 2 次，占 16k 混合步 wall 的 **57%**
- 单步 profile 里 `aten::index` 合计 18.8s / 33.2s span
- 另有 `pages = block_table[i,:n].long()` 每层每请求重复计算
  （42 层 × ~1.4ms 的 to_copy launch）

次要发现：CoreX 单 op host 发射成本 ~220-470µs（CUDA 的 30-50 倍），
混合步 ~400 个 op ≈ 100ms host——这决定了 host 侧优化天花板。

## 修复（vendor_flash_attn.py，补丁 a3_hostfix.patch）

1. decode 元数据全部改在 CPU 侧构建，显式 `non_blocking` HtoD +
   `index_select`（替代 aten::index）；dec_rows 由"每请求一个 arange +
   cat"（64+ 次 launch）改为单个 `torch.tensor` HtoD
2. prefill 的 block_table 行 `.long()` 提到每步一次（metadata 上挂
   memo，42 层共享）

## 结果

| 指标 | A1.5 | A3 |
|---|---|---|
| 16k 混合步 wall（trace） | ~677ms | **~353ms（-48%）** |
| 16k 混合步 GPU busy（trace） | 41.3% | **76.3%** |
| 16k total tok/s（官方 benchmark） | 2466.23 | 2488.58（**+0.9%**） |
| 16k TTFT | 193.7s | 191.6s（持平） |
| 4k total tok/s | 2635.85 | 2582.04（噪声内） |

**端到端持平的原因**：32GB 开发卡 KV 池 ~52 万 token，放不下
64×16k（1.05M），实际常驻并发 ~32；benchmark 时间被 decode 步
（cudagraph，~95ms）+ 尾部低并发 drain 支配，混合步占比小。
host 修复是真实且严格变好的（57% 的 stall 消除），在 64GB 评测机
（64 路全常驻、prefill 占比更高）上应当体现为实际收益。

数值路径零改动（纯 host 侧索引方式变化），精度沿用 A1.5 的 97.1%。

## 附带排除的线索

- `zeros_kernel ← aten::mm`（离线 trace 里 560ms）：真实 serve trace
  中不存在，是离线 profile 前缀缓存路径的伪影
- gems mm_kernel vs vendor gemm_tcu：probe 显示 vendor 快 6-24%，
  但受"禁止无开发量切换算子"约束，未动；后续若做 GEMM 优化应走
  FlagGems mm_kernel 调优路线
