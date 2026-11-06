# A58：A48 decode 双累加器内核 warps 调优（2026-11-06）

ixsys 显示当前 A48 的 `_mm_silu_mul_kernel` 是最大单项设备热点，累计约 **15.351s**、平均约 **1.395ms**。扩展扫描 BM/BN/BK、warps 和 stages 后，prefill 仍由当前 `BM=256/BN=128/BK=32/warps=8/stages=2` 最佳；decode 的 `warps=16` 出现稳定小幅收益。

## 微基准筛选

- decode 当前配置 `64/128/32/8/2`：0.10364ms；候选 `64/128/32/16/2`：0.09987ms，五次复测约快 **3.64%**。
- prefill 当前配置为 1.56703ms，新增组合均更慢，保持不变。
- 全部候选均通过数值检查。微基准只用于筛选，不替代官方结论。

FlagGems 分支 `exp/a58-dual-acc-warps16`，提交 `bc1d847e`，仅将 M≤128 的 warps 从 8 改为 16。官方形状测试 **2 passed**。候选使用独立 `VLLM_CACHE_ROOT=/workspace/a58-compile-cache`。

官方 4k A1/B1/A2/B2 正在执行。若两场景有可复现正收益，再补 MATH-500 Level 3 并决定合入。回滚为恢复 FlagGems `3642366c` 与当前 A48 缓存。
