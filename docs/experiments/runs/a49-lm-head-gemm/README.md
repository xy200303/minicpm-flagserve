# A49：lm_head 大词表 GEMM 配置（2026-11-04）

A31 的 pinned GEMM 表只覆盖 QKV、O、gate/up 和 down 投影，没有 `M=64、K=2048、N=130560` 的 lm_head 形状。该投影在每个 decode step 执行，通用 libtuner 配置与 host 查询是否适合大词表尚未单独验证。

探针以当前 `mm` 为对照，使用同一原始 Triton kernel 扫描少量配置。每个配置先检查有限值、最大/平均绝对差和 `assert_close(rtol=atol=0.05)`，再进行 20 次 CUDA event 计时。A49 不是重复 A34 的既有 QKV/MLP tile 扫描，只有该新形状出现明确且可复现的收益才会进入官方 benchmark。

探针排在 A48v3 官方 4k A/B 后执行，结果尚未产生。基线为 A31 `9c06143c`，不改 `vllm serve` 参数、模型或 benchmark。
