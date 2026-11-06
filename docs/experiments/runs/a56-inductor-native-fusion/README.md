# A56：Inductor 原生融合图审计（2026-11-06，关闭）

在当前 A48 缓存的生成代码中逐图检查 RoPE、RMSNorm、residual、MLP 和 copy 链，寻找无需 opaque custom op 的原生融合机会。证据来自 `/workspace/a48-compile-cache-v2` 的真实服务编译产物，不依赖算子名称猜测。

## 结论

- RoPE 的 split/index_select/chunk/mul/sub/add/cat 已归入 `triton_poi_fused_1`，没有独立的 Q/K 两个逐元素内核可供简单合并。
- residual add 与 RMSNorm 已由 `triton_red_fused_fused_add_rms_norm_1` 融合。
- A48 前置 RMSNorm、权重转置与 `fl::mm_silu_mul` 图节点已由 `triton_red_fused_fused_add_rms_norm_mm_silu_mul_t_0` 覆盖；GEMM 与 SiLU 门控已在 A48 合入。
- 剩余 `triton_poi_fused_2` copy 方向已由 A47 及其 A48 组合 A55 做过官方验证，未形成可复现正收益。

因此没有新的原生融合候选，不修改代码、不执行官方 benchmark。关闭此方向，不再新增 opaque custom op 破坏现有图融合。生成代码摘录见 `graph-evidence.txt`，版本与机器状态见 `machine-state.txt`；回滚无需操作。
