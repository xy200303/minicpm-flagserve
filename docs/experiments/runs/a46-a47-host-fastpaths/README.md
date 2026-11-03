# A46/A47：LibEntry 与连续 copy host 快速路径（2026-11-03）

## 动机与实验分组

A31 的 16k host profile 中，`LibEntry.run`、key 构造和动态 copy 包装有大量重复 Python 工作。为减少设备启动次数，将相关改动作为一个实验包，在同一设备窗口完成正确性测试和交错微基准；官方性能仍单独验证，不以微基准代替结论。

实验分支：`xy200303/FlagGems` 的 `exp/a46-a47-host-fastpaths`。

- A31 对照：`9c06143c`，远端 `/workspace/a31-gems`。
- A46：`1be4ae77`，预计算参数名、复用 key helper、减少 descriptor 判断、普通 dict 保序、缓存命中少一次哈希、grid 参数少一次字典重建。远端 `/workspace/a46-gems`。
- A47：`59a73960`，在 A46 上为连续、同形状、同 dtype 的 `copy_` 增加 1D Triton 内核，跳过通用动态形状包装；广播、类型转换、非连续和异常路径不变。`f20e89d3` 增加同形状非连续输入的回退测试，`bd3ff824` 采用实测 `BLOCK=256、num_warps=2`。远端 `/workspace/a47-gems`。

回滚：恢复 `GEMS_ROOT=/workspace/a31-gems`；只撤销 copy 可 checkout `1be4ae77`，撤销全部实验可 checkout `9c06143c`。

## 正确性与微基准

`run_micro.sh` 对 A31/A46/A47 运行完整 `test_copy.py`，并对 A46/A47 运行 LibEntry key、decorator cascade 和默认参数测试。A31/A46/A47 分别为 **46/51/51 passed**；最佳配置复测 copy 为 **47 passed**。

真实 `copy_` 探针按 A31/A46/A47/A47/A46/A31 执行两次，每个进程保留 9 组原始时序：

| 变体 | 两次中位数（ms） | 汇总中位数（ms） | 相对 A31 |
|---|---:|---:|---:|
| A31 | 0.09713 / 0.09739 | 0.09726 | 1.000× |
| A46 | 0.09149 / 0.09263 | 0.09206 | 1.056× |
| A47 | 0.03067 / 0.03111 | 0.03089 | 3.149× |

直接内核扫描覆盖 `BLOCK=256/512/1024/2048` 与 `num_warps=1/2/4`，最佳为 256/2（0.02193ms），原 1024/4 为 0.02249ms。所有配置均通过数值相等检查。原始 JSON、测试日志与自动汇总见 `a46-a47-micro/` 和 `summary.json`。

## 官方验证

A47 的官方 4k `[4096,1024,64,256]` A1/B1/A2/B2 已完成：A 为 `/workspace/audit-plugin + /workspace/a31-gems`，B 为 `/workspace/audit-plugin + /workspace/a47-gems`，启动参数不变。

- A1/A2：5211.79 / 5195.62 tok/s；B1/B2：5205.34 / 5202.58 tok/s。
- 基线中位数：5203.71 tok/s、TTFT 4091.40ms；候选中位数：5203.96 tok/s、TTFT 4047.31ms。
- 吞吐 **+0.0049%**，TTFT **−1.08%**。4k 无有效吞吐收益，不执行 MATH-500，不与 A45 叠加刷取组合数字。原始证据与自动汇总见 `official-4k/` 和 `official-4k-summary.json`。

A31 的 16k profile 中 copy/host 占比更高，因此也完成了官方 16k `[16384,1024,64,128]` A1/B1/A2/B2：A1/A2 为 4514.12/4509.72，B1/B2 为 4520.03/4514.41 tok/s。基线中位数 **4511.92 tok/s、TTFT 111861.26ms**，候选中位数 **4517.22 tok/s、TTFT 111732.36ms**，吞吐 **+0.12%**、TTFT **−0.12%**。原始证据与自动汇总见 `official-16k/` 和 `official-16k-summary.json`。

A47 在 4k 和 16k 均未达到 3% 门槛，正式否决，不合入、不执行 MATH-500。微基准的 3.149× 只代表连续 `copy_` 调用本身，完整模型关键路径不受该耗时支配。
