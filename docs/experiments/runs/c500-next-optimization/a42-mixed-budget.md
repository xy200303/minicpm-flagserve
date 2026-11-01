# A42：混合 prefill+decode 动态 token 预算（2026-11-01，验证中）

## 机制

分支 `exp/a42-mixed-prefill-budget`，提交 `44630c9`。仅在同一调度步同时存在未完成的 prefill 与 decode 时，将 token budget 限制为 4096；纯 prefill 和纯 decode 保持原始预算。`vllm serve` 参数与 benchmark 不变，未启用投机采样或量化。

实现位于 `vllm_fl/worker/scheduler_fl.py`：根据 running 请求的 `num_computed_tokens` 与 `num_prompt_tokens` 判断混合状态，调用父类 schedule 后恢复原预算。四个状态组合测试通过：空/纯 prefill、纯 decode、混合步、默认预算小于 cap。

## 官方结果

基线 A1 为当前交付 16k 的 11143.77 tok/s、TTFT 17961.76ms。B1 为 **11153.29 tok/s、TTFT 17933.78ms**，相对 A1 仅约 +0.09%，尚不能构成有效优化。A2/B2 完成前不合入，也不执行精度评测。

原始证据：`c500-a42-16k-b1/`、对应 tar 包和 `c500_a42_ab_chain.sh`。
