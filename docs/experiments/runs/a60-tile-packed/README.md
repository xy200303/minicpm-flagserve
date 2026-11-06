# A60：tile 预打包权重布局（待实例验证）

设计一种不同于 C500 nn 的 BI-V150 专用布局：将 gate/up 权重按 `BK×BN` tile 连续存放。当前 A48 的 B 矩阵通过 `weight.t()` 跨步读取，预打包后每个 K/N tile 在显存中连续，目标是在不改变双 FP32 累加器和输出语义的前提下改善权重访问。

探针位于 `docs/verification/bi_a60_tile_packed_probe.py`，覆盖：

- decode `M=64`，warps=8 与 warps=16；
- prefill `M=2111`，保持当前最优 tile；
- 当前布局与预打包布局的 A/B/A/B 计时；
- 数值差异、预打包时间和原始样本。

不使用 `tl.split`，不修改模型、serve 参数或 benchmark。当前仅完成探针实现，实例连接连续超时，**尚未运行，没有任何性能或正确性结论**。实例恢复后需先下载并校验未完成的 ixsys 原始包，再执行本探针。
