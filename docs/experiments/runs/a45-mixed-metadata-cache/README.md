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

已完成的 A1 为 **5090.41 tok/s、TTFT 4115.48ms**，B1 为 **5151.92 tok/s、TTFT 4126.53ms**；单组相对变化为吞吐 +1.21%、TTFT +0.27%，不能代替两轮中位数。A1/B1 的原始逐轮 CSV、汇总、日志和机器状态已归档在 `official-4k/`。A2/B2 尚未完成；未达到 3% 总吞吐门槛则不运行 MATH-500，也不合入交付栈。
