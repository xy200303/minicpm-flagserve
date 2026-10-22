# A21（否决）：把 RMSNorm 从 IR "native" 换成融合 Triton 核

## 动机

vLLM 0.24 把层归一化做成 IR op，并在引擎配置里写死

```
ir_op_priority = IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])
```

"native" 是纯 torch 链（eager 下每次 8–10 个核）：cast→add→cast→pow→mean→rsqrt→mul→cast→mul。
MiniCPM5-2B 每层 2 次、42 层，即每步 ~84 次。

## 做了什么

`fused_add_rms_norm_exact`（Triton，fp32 方差 + **先转 bf16 再乘 weight**，与 native 的数值路径完全对齐），
通过 `PlatformFL.import_ir_kernels()` 注册为 IR op provider，
`PlatformFL.get_default_ir_op_priority()` 把 `fused_add_rms_norm` 的优先级改成
`['flag_gems_iluvatar', 'native']`；A/B 开关 `VLLM_FL_DISABLE_IR_RMSNORM=1`。

## 结果

隔离（同输入、逐位对照）：

| 形状 | native | 融合核 | 加速 |
|---|---|---|---|
| N=64（decode） | 132.8 µs | 54.0 µs | 2.46x |
| N=512 | 137.5 | 59.6 | 2.31x |
| N=2048 | 569.7 | 107.7 | 5.29x |
| N=8192（prefill） | 2039.4 | 281.5 | 7.24x |

引擎（官方 4k，同日）：

| | 总吞吐 | TPOT | TTFT |
|---|---|---|---|
| off | 2538.69 | 97.70 | 5499 |
| off2 | 2534.52 | 97.84 | 5489 |
| **on** | **2388.44（−5.9%）** | **100.03** | **9968（翻倍）** |

## 为什么输

不是核慢（隔离里快 2.5–7.2 倍）。IR op 在编译期会被 **inline 展开**，所以 Triton 核必须包成
opaque 的 `torch.library.custom_op`（否则 inductor 会 trace 到核里，报
`Cannot access data pointer of Tensor (FakeTensor)`）。一旦 opaque，
**inductor 就失去了对相邻算子的融合**（例如把 residual add 融进前一个 GEMM 的 epilogue）；
prefill 张量大、最吃融合，于是 TTFT 翻倍，decode 也跟着变慢。

结论：在"native 链能被 inductor 融合"的前提下，IR 级替身是净亏。已全部回退
（`platform.py` 回到 `b0cda5e`，`ir_rmsnorm.py` 删除，FlagGems 侧文件仅作留档）。

脚本：`bench_rmsnorm*.py`、`ir_rmsnorm.py`、`fused_add_rms_norm_exact.py`。
