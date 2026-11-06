# A57：FlagGems 默认日志级别优化（2026-11-06）

当前 A48 栈的 4k host profile 显示，`logging.debug` 累计约 **11.861s**，LogRecord 构造约 **6.284s**。调用分布于 copy、unbind、index_select、add 等大量 FlagGems 算子，不是单一 copy 热点。原始 profile 见 `baseline-profile/`。

A57 在 `flag_gems.logging_utils.configure_default_logging()` 中将根 logger 默认设为 INFO；设置 `FLAGGEMS_LOG_LEVEL=DEBUG` 可恢复原行为。已有 `setup_flaggems_logging()` 的显式诊断路径仍设置 DEBUG 并写文件，不改变排障能力。FlagGems 分支 `exp/a57-logging-default`，提交 `eea51ec7`。

相关测试 **122 passed、1 skipped、3 deselected**，见 `gems-tests.log`。候选使用独立 `VLLM_CACHE_ROOT=/workspace/a57-compile-cache`，基线为当前 A48。官方 serve 参数、模型和 benchmark 不变。

## 官方 4k 结果

| 运行 | 总吞吐（tok/s） | 平均 TTFT（ms） |
|---|---:|---:|
| A1 | 5241.33 | 3902.44 |
| A2 | 5246.42 | 3913.87 |
| B1 | 5213.34 | 3929.05 |
| B2 | 5249.30 | 3911.61 |
| A 中位数 | 5243.88 | 3908.15 |
| B 中位数 | 5231.32 | 3920.33 |

吞吐变化 **−0.2394%**，TTFT 变化 **+0.31%**。默认关闭 debug 日志虽减少了 host 函数调用，但没有形成端到端正收益，说明这些累计 host 时间不在关键路径上或被设备等待掩盖。A57 不合入，不继续 16k 或 MATH-500。自动汇总见 `official-4k-summary.json`，原始工件位于 `a1/a2/b1/b2`。回滚为恢复 FlagGems `3642366c` 和 A48 缓存。
