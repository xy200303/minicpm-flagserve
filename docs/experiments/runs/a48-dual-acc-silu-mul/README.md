# A48：双累加器 GEMM＋SiLU 乘法门控（2026-11-03）

## 与 A43 的区别

A43 在单个 GEMM 累加器上使用 `tl.split`/切片，Iluvatar 后端实际数据布局与假设不符，探针出现严重数值错误，已否决。A48 不采用该路线：两个独立 FP32 累加器分别计算 gate 与 up，输入 A 每个 K tile 只加载一次，B 的 gate/up 指针独立递增，最后直接计算 `gate * sigmoid(gate) * up` 并写出，不对累加器做切片。

当前仅为独立探针，基线为 A31 `9c06143c`，工作分支 `exp/a48-dual-acc-silu-mul`。没有改动交付栈，也没有性能或正确性结论。

## 探针范围

- decode：M=64、K=2048、N=6144；混合 prefill：M=2111，其余相同。
- 对照为当前 pinned `mm` 加现有 SiLU 乘法，且一次 GEMM 只执行一次。
- 每个候选配置先检查有限值、最大/平均绝对差和 `assert_close(rtol=atol=0.05)`，再做 20 次 CUDA event 计时。
- 扫描少量保守 tile，失败配置保留错误文本，不用错误结果计时。

探针排在 A47 官方 16k A/B 完全退出后运行。只有数值通过且总 GEMM＋activation 路径有明确收益，才会集成到 FlagGems 并执行官方 benchmark；不会沿用 A43 的无效加速数字。
