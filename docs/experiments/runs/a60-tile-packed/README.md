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

使用独立 `VLLM_CACHE_ROOT=/workspace/a60-compile-cache`，官方 serve 参数、模型和 benchmark 不变。4k A1/B1/A2/B2 正在执行，结果待完整对照后填写。回滚为恢复 FlagGems `3642366c` 与当前 A48 缓存；合入前还需 16k 对照和 MATH-500 Level 3 ≥0.95。
