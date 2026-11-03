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

FlagGems 的 `mm_silu_mul` 使用上述双累加器内核，两个官方形状测试 **2 passed**。插件将 MiniCPM 的 MLP 在未量化、指定权重视图和官方 token 形状下路由到 `fl::mm_silu_mul` 自定义算子，其余场景回退原路径；`VLLM_FL_DISABLE_MLP_FUSION=1` 可一键恢复原路径，恢复 `GEMS_ROOT=/workspace/a31-gems` 和 `PLUGIN_ROOT=/workspace/audit-plugin` 可完整回滚。

模型配置为 `LlamaForCausalLM`，实际走 `LlamaMLP`；仓库同时存在独立的 `MiniCPMMLP`。`a8349b5` 同时覆盖两个类，路由测试 **3 passed**。初版未记录命中，旧 B1 不作为正式结果。

后续诊断发现 vLLM 直接复用了不含补丁的 AOT 缓存。候选改用独立的 `VLLM_CACHE_ROOT=/workspace/a48-compile-cache-v2` 强制重新编译；`4ea4a6e` 移除了 Dynamo 不支持的日志和全局计数副作用。新图已通过编译和 CUDA graph 捕获，生成代码中可检索到 `fl::mm_silu_mul`，因此确认不是回退。首次冷运行出现回退后，使用已生成缓存重新执行完整 4k A1/B1/A2/B2 以区分冷编译与稳态性能，结果尚未产生。

补充的转置权重探针保留在 `results-transposed.json`：decode 基线/A48 为 0.12065/0.10421ms，prefill 为 1.91356/1.56034ms，仍分别为 1.158×/1.226×。它只用于说明布局影响，不替代官方成绩。
