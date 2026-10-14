# 天数 BI-V150 A7：decode attention 发射配置 pin（w1 s1）

日期：2026-10-14 ｜ 实例：FlagOS OnlineLab BI-V150（32GB 显存版）

## 起因

赛题组官方口径（陈卓，赛道二答疑）：允许"针对同一算子自行实现并
优化多个等价 Triton Kernel，根据输入 Tensor Shape 动态选择不同
kernel 实现或执行配置"，即 **shape-aware autotuning / dispatch 明确
合规**。A2 曾证明 decode attention 在算法层面（自研 flash-decoding
vs stock）打平，但当时只测了默认发射配置。A7 补上配置维度。

## 配置链摸底

Iluvatar decode（max_seqlen_q=1，batch≤64）走
`triton_unified_attention` 的 **3D 路径**（flash-decoding 分段）：

- `seq_threshold_3D = 128 // num_kv_heads = 64`（batch 64 恰好 ≤64
  进 3D）；`NUM_PAR_SOFTMAX_SEGMENTS = 16`
- `BLOCK_M=16 / BLOCK_Q=2`（GQA 8 q/kv）；`TILE_SIZE_DECODE=16`
  （bf16 默认）；`num_warps/num_stages = None` → Triton 默认 w4 s3

约束：非 TD 路径 TILE_SIZE 可大于 BLOCK_SIZE（内核有跨块分支），
TILE 只需 2 的幂。

## 离线扫描（batch=64，GQA 16q/2kv，d=128，block=16，bf16，
含 3D 主内核 + reduce_segments 全链路，50 次迭代均值）

**num_warps × num_stages**（kv=16896，TILE=16，SEG=16）：

| 配置 | µs | vs stock |
|---|---|---|
| stock（=w4 s3） | 6464 | 1.00x |
| **w1 s1** | **3535** | **1.83x** |
| w2 s1 | 3882 | 1.67x |
| w4 s1 | 4579 | 1.41x |
| w1 s2 | 7494 | 0.86x |
| w8 s3 | 9480 | 0.68x |

规律：`num_stages` 是最大负收益项（每个 warp 档上 s1→s3 单调
变差，软件流水线在这颗芯片上纯属开销）；warps 越少越好。
kv=4608 同样 w1 s1 最优：1099µs vs stock 1938µs（1.76x）。

**TILE_SIZE**（w1 s1，SEG=16）：16 → 3523µs 最优；32/64/128 =
4572/7664/19846µs（大 tile 寄存器压力爆炸）。**SEG**：4-64 扫过，
16 最优（曲线平，8-32 都在 ±4% 内）。

双场景同一配置获胜，无需分档派发，单一 pin 即可。

**正确性**：w1 s1 与 stock 输出 **max_diff = 0.0**（bit-exact，
调度参数不改数学）；fp32 eager 参考误差两者同为 0.0002。

## 实现（vllm-plugin-FL commit 84abed0）

新模块 `iluvatar/impl/decode_attn_pin.py`：包装
`kernel_unified_attention` 的 `__getitem__`，仅当 `IS_3D=True`
时注入 `num_warps=1, num_stages=1`（2D 路径不动），在
`iluvatar.py` 与 A1/A5 同点挂载。开关：
`VLLM_FL_DISABLE_DECODE_PIN=1` 回退 stock；
`VLLM_FL_DECODE_NUM_WARPS/NUM_STAGES` 覆盖 pin 值。

## 端到端（官方 benchmark，4 跑弃首取后 3 均值）

| 场景 | A6 | A7 | Δ |
|---|---|---|---|
| 4k total tok/s | 2738.65 | **3778.65** | **+38.0%** |
| 16k total tok/s | 2558.37 | **3567.33** | **+39.4%** |

16k 三次有效跑 3566.79/3567.92/3567.29，组内极差 0.03%。
TTFT 大降：4k mean 4403→4284ms、p99 26150→24532ms；
16k mean **186.4s→137.0s**、p99 396.2s→285.1s。
收益量级说明：decode 步 GPU 时间此前被 stock Triton decode
attention 主导（4k 场景单步 ~81ms/42 层 ≈ 步长的 ~2/3），
A2 的"打平"结论只覆盖了算法维度，配置维度才是大头。

精度实测 **98.1%**（MATH-500 L3，105 题，evalscope 同口径），
≥0.95 达标。bit-exact 改动，精度不变属预期。

## 备注

- 微基准脚本 `/tmp/a7_decode_sweep.py`（形状构造 + 分段扫描 +
  fp32 参考校验），bit-exact 验证 `/tmp/a7_bitexact.py`
- 4k 计分跑 3319.59/3898.19/4118.16 单调上升（JIT/分配器持续
  预热），取均值 3778.65；若舍首跑取后二均值口径更高，为保守
  起見按官方 4 跑弃首口径记
