# A58：A48 decode 双累加器内核 warps 调优（2026-11-06）

ixsys 显示当前 A48 的 `_mm_silu_mul_kernel` 是最大单项设备热点，累计约 **15.351s**、平均约 **1.395ms**。扩展扫描 BM/BN/BK、warps 和 stages 后，prefill 仍由当前 `BM=256/BN=128/BK=32/warps=8/stages=2` 最佳；decode 的 `warps=16` 出现稳定小幅收益。

## 微基准筛选

- decode 当前配置 `64/128/32/8/2`：0.10364ms；候选 `64/128/32/16/2`：0.09987ms，五次复测约快 **3.64%**。
- prefill 当前配置为 1.56703ms，新增组合均更慢，保持不变。
- 全部候选均通过数值检查。微基准只用于筛选，不替代官方结论。

FlagGems 分支 `exp/a58-dual-acc-warps16`，提交 `bc1d847e`，仅将 M≤128 的 warps 从 8 改为 16。官方形状测试 **2 passed**。候选使用独立 `VLLM_CACHE_ROOT=/workspace/a58-compile-cache`。

## 官方 4k 结果

| 运行 | 总吞吐（tok/s） | 平均 TTFT（ms） |
|---|---:|---:|
| A1 | 5250.67 | 3887.52 |
| A2 | 5240.77 | 3910.65 |
| B1 | 5227.04 | 3913.94 |
| B2 | 5260.88 | 3927.78 |
| A 中位数 | 5245.72 | 3899.09 |
| B 中位数 | 5243.96 | 3920.86 |

吞吐变化 **−0.0336%**，整体持平，没有可复现的正收益。A58 不合入、不继续 16k 或 MATH-500，交付保持 warps=8。自动汇总见 `official-4k-summary.json`。

设备汇总见 `ixsys-summary.json`。原始 trace 保留在实例 `/workspace/experiments/runs/a58-current-ixsys/serve.ixsys`，压缩包 `/workspace/a58-current-ixsys.tar.gz` 的 SHA256 为 `3dd52c433c437bc0edd06e1d7b00e6aac3465f171d88c2cf11c6c79c95a7f193`。大文件下载多次超时，损坏的本地片段未纳入提交。回滚为恢复 FlagGems `3642366c` 与当前 A48 缓存。
