# A60：K 连续 tile 预打包权重布局（2026-11-06）

设计一种不同于 C500 nn 的 BI-V150 专用布局：将 gate/up 权重按 `BK×BN` tile 连续存放，并在 tile 内保持 K 维连续，匹配当前后端 `tl.dot` 的访存假设。目标是减少 A48 的 B 矩阵跨步访问，同时保持双 FP32 累加器和输出语义不变。

## 正确实现与验证

- FlagGems 分支 `exp/a60-tile-packed`，最终提交 `ffe47a67`。
- 预打包权重只缓存一次，不进入 decode 稳态路径；缓存保留原权重引用，避免地址复用。
- 官方形状和缓存内容测试 **3 passed**，日志见 `gems-tests-v5.log`。
- 修正后的微基准见 `results-corrected.json`、`bench-corrected.py`、`bench-corrected.log`：

| 场景 | 当前布局 | 预打包布局 | 加速比 |
|---|---:|---:|---:|
| decode，warps=8 | 0.10373ms | 0.09431ms | 1.100× |
| decode，warps=16 | 0.10442ms | 0.09425ms | 1.108× |
| prefill，warps=8 | 1.55868ms | 1.50092ms | 1.038× |

微基准只用于候选筛选，不替代官方结论。

## 调试记录与无效数据

早期探针存在三个问题：多余转置、tile 内 K/N 顺序错误，以及输出使用 tile 内 `rn_local` 导致第二个及后续 N tile 写回第一 tile。最终修复为 K 连续打包、逻辑 `[BK,BN]` 点积形状和全局 `rn` 输出偏移。早期结果不作为有效数据，尤其不能引用未修正输出偏移前的加速比。

原始 ixsys 压缩包已完整下载到 `a58-dual-acc-warps16/ixsys.tar.gz`，SHA256 与服务器一致：`3dd52c433c437bc0edd06e1d7b00e6aac3465f171d88c2cf11c6c79c95a7f193`。

## 官方验证

使用独立 `VLLM_CACHE_ROOT=/workspace/a60-compile-cache`，官方 serve 参数、模型和 benchmark 不变。

### 官方 4k 结果

| 运行 | 总吞吐（tok/s） | 平均 TTFT（ms） |
|---|---:|---:|
| A1 | 5237.00 | 3926.43 |
| A2 | 5239.56 | 3905.50 |
| B1 | 5296.55 | 3848.30 |
| B2 | 5331.74 | 3822.73 |
| A 中位数 | 5238.28 | 3915.97 |
| B 中位数 | 5314.15 | 3835.52 |

吞吐变化 **+1.448%**，TTFT 变化 **−2.06%**，两组均复现正收益。自动汇总见 `official-4k-summary.json`，原始工件位于 `a1/a2/b1/b2`。

继续执行 16k A1/B1/A2/B2 和 MATH-500，随后决定是否合入。回滚为恢复 FlagGems `3642366c` 与当前 A48 缓存。
