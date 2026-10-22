# A22（否决）：W8 权重量化 + 厂商 int8 GEMM

## 动机

decode 步非 attention 的 11.7 ms **完全由权重读取解释**（5.3 GB/步 ÷ 450 GB/s）。
厂商栈提供 `torch._int_mm`（int8×int8→int32），实测在大形状上比 bf16 快得多：

| 形状（M=64） | bf16 | int8 | 备注 |
|---|---|---|---|
| K=2048, N=6144 | 56.8 µs | 26.1 µs | 2.2x（纯 GEMM） |
| gate_up + SwiGLU（融合 epilogue） | 137.5 µs | **73.8 µs** | 1.86x，**但激活是预算化好的** |

## 实测（真实集成路径）

把激活量化、反量化都算进去（插件的 `w8_mlp.py`，只改 gate_up）：

| 路径 | 耗时 |
|---|---|
| bf16 GEMM（`x @ w.t()`） | 120.9 µs |
| W8：`x.abs().max()` + 量化核 + `_int_mm` + 反量化核 | **157.5 µs（0.77x）** |

即：**两个额外的小核（per-call 激活量化 + 反量化）合计约 80 µs，而 gate_up 的 bf16 GEMM 只有 115–121 µs**，
收益被完全吃掉。

## 数值侧是干净的（不是失败原因）

- 纯 torch 参考路径：per-tensor 激活 + per-channel 权重量化，`y_dq` 与 bf16 matmul 的 maxdiff **0.054**
  （y 的 absmax 4.0）；
- `torch._int_mm` 在 M=64、K=2048、N=12288 上**逐位精确**（maxdiff 0）；
- 反量化 Triton 核与公式逐位一致（maxdiff 0.0）。

## 为什么救不回来

要把 80 µs 的量化开销藏起来，只有一个办法：**融合**（量化 + int8 GEMM + 反量化 + SwiGLU 一个核）。
但本卡上 Triton matmul 比厂商 BLAS 慢 ~6 倍（台账里的 GEMM 校准），自写融合 GEMM 不可能赢；
厂商又没有"带 epilogue 的 int8 GEMM"接口（`bnb_qgemm` 是 6 GB/s 的参考实现，直接排除）。

## 结论

在 M=64 这种"GEMM 本身只有 ~115 µs"的 decode 形态下，W8 的固定开销（两次小核）与收益同量级，
**净亏**。若批再大一个量级（如 prefill 的 M=4096），W8 才有意义，但那时 decode 的权重带宽已不是瓶颈。

脚本：`bench_w8_gateup.py`（融合 silu 的上界）、`test_w8_mlp.py`（真实集成）、
`debug_w8*.py` / `debug_intmm_*.py`（数值定位）、`w8_mlp.py`（插件原型，未启用）。
