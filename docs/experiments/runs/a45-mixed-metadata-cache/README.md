# A45：混合批 attention metadata 索引缓存（2026-11-03）

## 候选与实现

基线为 BI-V150 的 A20+A31 交付栈。原实现每个混合步用 Python 列表对 `query_start_loc`、`seq_lens` 和 `block_table` 做三次高级索引。A45 缓存 decode 行号对应的 GPU 索引张量，并将三处操作改为 `torch.index_select`；`query_start_loc` 仍按 decode 请求数生成，prefill 的 `cu_q/cu_k` 和其余执行路径不变。

- 实验分支：`xy200303/vllm-plugin-FL` 的 `exp/a45-mixed-metadata-cache`，提交 `16aa58d`。
- 回滚：恢复 `PLUGIN_ROOT=/workspace/audit-plugin`，或对实验分支提交执行 `git revert 16aa58d`。
- 单元测试：2 passed，覆盖索引缓存复用，以及 prefill/decode 行号、序列长度、页表和 decode metadata 的切分结果。

## 微基准

`metadata-probe.json` 记录 B=64、63 个 decode 请求加 1 个 2048-token prefill 请求、页表宽度 8192 的 `_build_split` 隔离测量。每点 9 组、每组 30 次，组末同步并取中位数；这只是筛选证据，不是官方吞吐结论。

| prefill 行位置 | 原实现 | A45 | 加速比 |
|---|---:|---:|---:|
| 0 | 0.4981 ms | 0.3970 ms | 1.255× |
| 32 | 0.4966 ms | 0.3924 ms | 1.266× |
| 63 | 0.4952 ms | 0.3948 ms | 1.254× |

## 官方验证

官方 4k `[4096,1024,64,256]` 按 A1/B1/A2/B2 顺序运行，A 为 `/workspace/audit-plugin + /workspace/a31-gems`，B 为 `/workspace/a45-plugin + /workspace/a31-gems`。启动参数与基线一致，使用 `FULL_DECODE_ONLY`。

A1/A2 分别为 **5090.41/5148.34 tok/s**，B1/B2 分别为 **5151.92/5166.25 tok/s**。基线中位数为 **5119.38 tok/s、TTFT 4126.51ms**，候选中位数为 **5159.09 tok/s、TTFT 4103.36ms**：吞吐 **+0.78%**，TTFT **−0.56%**。TTFT 满足对照中位数 +1% 的限制，但吞吐低于 3% 准入门槛，A45 不合入、不执行 MATH-500。

四次运行的原始逐轮 CSV、汇总、日志和机器状态均归档在 `official-4k/`，自动汇总见 `official-4k-summary.json`。微基准的约 1.25× 仅说明 `_build_split` 本身更快，完整模型收益不足，不能作为采纳依据。
