# 天数 BI-V150 A1：prefill attention 切到 vendor ixAttnBkd flash

日期：2026-10-10 ｜ 实例：FlagOS OnlineLab BI-V150（32GB 显存版）

## 基线（stock flagos-2026-s2，官方命令+FULL_DECODE_ONLY）

| 场景 | 基线 total tok/s | 备注 |
|---|---|---|
| 4k [4096,1024,64,256] | 1902.15 | TTFT 12.3s |
| 16k [16384,1024,64,128] | 861.98 | TTFT 636s（32GB KV 池排队放大） |

runs：4k 1565.30(弃)/1876.12/1907.64/1922.70；16k 843.61(弃)/862.02/861.84/862.07。
注意：本实例是 32GB 版本（赛题评测机为 64GB），KV 池 ~52 万 token，
16k×64 并发放不下，绝对值偏低；优化方向不受影响。

## 根因（profiler trace 实锤）

16k prefill trace：`kernel_unified_attention`（vLLM 通用 Triton 参考实现，
iluvatar 后端因无 flash_attn 包只能用它）占 GPU busy 的 **91.8%**，
每层每 2048-chunk 36.8ms——比 C500 vendor flash 慢 30 倍。
GPU busy 99.1%，纯粹是内核慢，不是调度问题。

## vendor FA 探针（关键实验）

CoreX 自带 `libixattnbkd.so`（ixAttnBkdFlashAttnForward，varlen+GQA），
torch 的 `aten::_efficient_attention_forward` 已接线到它。逐个语义/性能试出来：

| 调用方式 | 2048×16384 GQA causal | 语义 |
|---|---|---|
| F.sdpa is_causal（BHSD） | 404µs | top-left，chunked prefill 下**错误** |
| 直接 C API ctypes/包装 | segfault | ABI/布局不明，放弃 |
| `_efficient_attention_forward` cmt=1 | 404µs（快路径） | top-left，chunk 下错 |
| `_efficient_attention_forward` cmt=2 | 3492µs（慢路径） | **bottom-right，正确** |
| `_efficient_attention_forward` cmt=0（无 mask） | 3102µs | 非因果 |
| cmt=2 @ kv≤2048 | 407µs | 小 kv 走快路径 |
| `_flash_attention_forward`（BSHD raw） | 3492µs | bottom-right 正确 |

结论：cmt=2 是唯一正确且可用的选择；慢路径随 kv 线性（~0.22µs/token/层），
kv=2048 以下反而走快路径。拆分历史+当前块无法用上 cmt=1 的快路径
（历史块全连接 attention 也走慢路径），放弃。

## 补丁（vllm_fl iluvatar 后端 vendor_flash_attn.py）

两个 hook（apply() 在 iluvatar.py 模块级调用）：
1. `TritonAttentionMetadataBuilder.build`：每步一次从 CPU 侧元数据
   （query_start_loc_cpu + seq_lens_cpu_upper_bound）算 decode/prefill 请求
   切分，挂在 metadata 对象上（42 层共享）。
2. `TritonAttentionImpl.forward`：纯 decode 步走原路不动；混合步 decode 行
   走原版 unified_attention（切片 metadata），prefill 行逐请求走 vendor op
   （分页 KV 先 gather 到常驻池 buffer）。
   alibi/真滑窗/softcap/sinks/fp8KV/cascade/encoder/异常 → 全部回退原版。

## 踩过的坑（都有实测证据）

1. **每层 tolist() 同步**：第一版在 forward 里 per-layer 做
   `cu.tolist()/seq_lens.tolist()`，每层 2 次 DtoH 全队列同步 ×42 层 →
   split 段累计 268s/83 步。修法：挪到 builder 用 CPU 侧数据。
2. **分配器风暴**：32GB 卡 0.85 占用下，每层 index_select 新分配触发
   torch allocator 的 sync-free 循环（43ms/层 host）。修法：常驻池
   `_pool` + `index_select(out=)`。
3. vendor op 本身全异步（host 发射 0.04ms），无 per-shape 重调优。
4. iluvatar 的 `sliding_window` 恒为 (-1,-1) 不是 None——eligibility 要排除。
5. dbg 日志要在 mql>1 时才打，否则被 decode cudagraph 捕获阶段吃掉。

## 实测（官方 benchmark，4 跑弃首取均值；32GB 实例口径）

| 场景 | 基线 | A1 | Δ |
|---|---|---|---|
| 4k `[4096,1024,64,256]` | 1902.15 | 1866.06 | -1.9%（阈值门控走原路，噪声内） |
| 16k `[16384,1024,64,128]` | 861.98 | **2461.59** | **+185.6%（2.86x）** |

- 4k runs：1243.20（弃）/ 1863.11 / 1826.71 / 1908.36
- 16k runs：2389.33（弃）/ 2461.87 / 2461.95 / 2460.94
- 16k TTFT：636.5s → **194.5s**（3.3x）；16k 混合步速 ~1500ms → ~380ms（4x）
- 16k decode 仍走 stock Triton unified attention（后续可继续挖）
- 精度：MATH-500 Level 3 = **98.1%**（105 题，与基线/C500 完全一致，
  vendor FA 数值无损）。
- CSV：`benchmark_results/summary_20261010_144102.csv`
