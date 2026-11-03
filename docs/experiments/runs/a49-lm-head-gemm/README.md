# A49：lm_head 大词表 GEMM 配置（2026-11-04）

A31 的 pinned GEMM 表只覆盖 QKV、O、gate/up 和 down 投影，没有 `M=64、K=2048、N=130560` 的 lm_head 形状。该投影在每个 decode step 执行，通用 libtuner 配置与 host 查询是否适合大词表尚未单独验证。

探针以当前 `mm` 为对照，使用同一原始 Triton kernel 扫描少量配置。每个配置先检查有限值、最大/平均绝对差和 `assert_close(rtol=atol=0.05)`，再进行 20 次 CUDA event 计时。A49 不是重复 A34 的既有 QKV/MLP tile 扫描，只有该新形状出现明确且可复现的收益才会进入官方 benchmark。

基线为 A31 `9c06143c`，不改 `vllm serve` 参数、模型或 benchmark。

探针已完成，所有配置均通过数值检查。通用 `mm` 中位耗时为 **1.01296ms**，最佳配置 `BM=64、BN=256、BK=32、SPLIT_K=1、warps=16、stages=2` 为 **1.00948ms**，仅加速 **0.34%**，没有官方长测价值。A49 不合入、不执行 MATH-500。原始数据见 `results.json` 和 `probe.log`。
