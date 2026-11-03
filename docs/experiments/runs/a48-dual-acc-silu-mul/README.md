# A48：双累加器 GEMM＋SiLU 乘法门控（2026-11-03）

## 与 A43 的区别

A43 在单个 GEMM 累加器上使用 `tl.split`/切片，Iluvatar 后端实际数据布局与假设不符，探针出现严重数值错误，已否决。A48 不采用该路线：两个独立 FP32 累加器分别计算 gate 与 up，输入 A 每个 K tile 只加载一次，B 的 gate/up 指针独立递增，最后直接计算 `gate * sigmoid(gate) * up` 并写出，不对累加器做切片。

基线为 A31 `9c06143c`，FlagGems 工作分支 `exp/a48-dual-acc-silu-mul`，实现提交 `f9085b6d`。插件从干净 A20 的 `0207b7e` 新建 `exp/a48-mlp-fusion`，路由提交 `061753d`，不包含已否决的 A45/A47。

## 探针范围与结果

- decode：M=64、K=2048、N=6144；混合 prefill：M=2111，其余相同。
- 对照为当前 pinned `mm` 加现有 SiLU 乘法，且一次 GEMM 只执行一次。
- 每个候选配置先检查有限值、最大/平均绝对差和 `assert_close(rtol=atol=0.05)`，再做 20 次 CUDA event 计时。
- 扫描少量保守 tile，失败配置保留错误文本，不用错误结果计时。

所有探针配置均通过数值检查，decode/prefill 最大绝对差分别为 1.53e-05/3.05e-05。最佳配置：

| 场景 | 基线总路径 | A48 最佳 | 加速比 | 配置（BM/BN/BK/warps/stages） |
|---|---:|---:|---:|---|
| decode | 0.11474ms | 0.09616ms | 1.193× | 64/128/32/8/2 |
| prefill | 1.76628ms | 1.48752ms | 1.188× | 256/128/32/8/2 |

原始结果见 `results.json` 和 `probe.log`。这些数值仅用于筛选，不作为端到端结论。

## 集成与验证

FlagGems 的 `mm_silu_mul` 使用上述双累加器内核，两个官方形状测试 **2 passed**。插件将 MiniCPM 的 LlamaMLP 在未量化、指定权重视图和官方 token 形状下路由到 `fl::mm_silu_mul` 自定义算子，其余场景回退原路径；两项路由测试 **2 passed**。`VLLM_FL_DISABLE_MLP_FUSION=1` 可一键恢复原路径，恢复 `GEMS_ROOT=/workspace/a31-gems` 和 `PLUGIN_ROOT=/workspace/audit-plugin` 可完整回滚。

官方 4k A1/B1/A2/B2 已启动：A 为 `/workspace/audit-plugin + /workspace/a31-gems`，B 为 `/workspace/a48-plugin + /workspace/a48-gems`。尚未产生端到端结果；达到 3% 且 TTFT 合格后才执行 MATH-500 和合入。
