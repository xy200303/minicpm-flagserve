# Attention 调研：C500 prefill flash-attn 内核选择已到顶（负面结果）

日期：2026-10-07 ｜ 机器：C500 64GB（140.207.205.81）

## 背景

16k prefill trace 里 vendor flash attention 占 GPU 时间 42%（6.55s/15.6s），
是 GEMM 之后的第二大头。16k 场景在 opt8（GEMM 走 vendor mcBLAS）之后，
attention 的相对占比进一步上升，因此调研其优化空间。

## 运行时路径（实锤）

- metax 后端用 MetaX 移植版 **FA2**（`flash_attn 2.6.3+metax3.7.0.7`，
  `flash_attn_2_cuda.so` 984MB，mctlass 预编译全部候选内核）。
- prefill 调用：`flash_attn_varlen_func`，paged KV（block_table），causal；
  每个 2048-token chunk 一次调用（`max_num_batched_tokens=2048`，
  每步 1 条 prompt 的 1 个 chunk）。
- FA3 的 `get_scheduler_metadata` 在 maca 上是 **None 桩**（未实现）。
- trace 里两种 traits：(128,128,64,4) n=4284 avg 1205µs +
  (128,64,64,4) n=672 avg 1900µs —— 默认启发式按形状二选一。

## vendor 自带调优框架（flash_attn.tuning）

- `kernel_traits_candidates.yaml`：fwd 70 组 / fwd_split 若干组候选
  （hdim128 bf16 的 splitkv 有 4 组：blockm×blockn = 128×64 / 64×64 /
  64×32 / 32×32）。
- `flash_attn_2_cuda.ks_set_solution(kernel_id, kernel_type, num_splits, alg)`
  可在运行时强制指定内核；`ks_load_solution()` + 环境变量
  `MHA_SOLUTION_PATH` 加载离线调优表（flatbuffers bin）。
- **solution 表按 problem 精确 hash 匹配**（hash 含 batch、max_seqlen、
  seqlen 向量——实测逐一敏感），对动态 serving 场景静态表不可用；
  正确用法是自建分桶 dispatcher + ks_set_solution。

## 扫描实验（sweep_fa_prefill.py）

单步 prefill 形状：batch=1，q=2048×16头×128，paged kv，causal，bf16。
对 5 个 kv 长度桶 × 4 个 splitkv 内核 × 6 种 alg × num_splits∈{1,2,4,8,16}
（另加 plain fwd 内核若干）全扫描：

| kv_len | 默认调度 | 全空间最优 | 提升 |
|---|---|---|---|
| 2048 | 248.6µs | 245.3µs | 1.01x |
| 4096 | 576.2µs | 569.9µs | 1.01x |
| 8192 | 1260.3µs | 1250.1µs | 1.01x |
| 12288 | 1946.3µs | 1938.0µs | 1.00x |
| 16384 | 2629.3µs | 2622.3µs | 1.00x |

**默认启发式在所有桶上均已选择最优内核**（128×64 splitkv），
内核选择层面无收益（<1%，噪声级）。

## 结论

C500 的 prefill attention 在 vendor 内核库存内已到顶。剩余理论空间只有
自研 Triton FA2 内核去挑战 mctlass——参照 GEMM 的经验（Triton 最佳仅为
vendor 的 60-74%），自研打 vendor 高度优化的 attention 管线胜率很低，
不投入。16k 场景后续空间主要在调度/主机侧（GPU idle 19%）而非内核。

（SM count = 104，供后续参考。）

## 补：decode 侧扫描（sweep_fa_decode.py，2026-10-07）

decode 走 `flash_attn_with_kvcache`（q=[64,1,16,128]，paged KV，causal）。
两个维度都扫：python API 的 num_splits（0=启发式 .. 32），以及
ks_set_solution 强制全部 5 组 splitkv traits × alg × splits。

| kv_len | 默认 | 全空间最优 | 提升 |
|---|---|---|---|
| 2048 | 147.8µs | 125.1µs（blockm_32 内核） | 1.18x |
| 4096 | 210.8µs | 210.8µs | 1.00x |
| 8192 | 394.0µs | 394.0µs | 1.00x |
| 16384 | 757.6µs | 750.5µs | 1.01x |

- 唯一有收益的是 kv=2048 桶（+18%），但**两个计分场景都不会落到这个桶**
  （4k 场景 decode 时 kv≈4096+，16k 场景 kv≈16384+）→ 无计分价值。
- num_splits 手动指定全面输启发式（如 kv16384 最好手动值 774.7µs > 757.6µs）。
- decode attention 在 kv=16384 时单步每层读 1.07GB KV，757.6µs ≈ 1.42TB/s，
  已在 HBM 带宽的较高分位，剩余空间需要新内核而非调度。

**总结论：attention（prefill + decode）在内核选择层面全部到顶。**
