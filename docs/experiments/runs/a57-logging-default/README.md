# A57：FlagGems 默认日志级别优化（2026-11-06）

当前 A48 栈的 4k host profile 显示，`logging.debug` 累计约 **11.861s**，LogRecord 构造约 **6.284s**。调用分布于 copy、unbind、index_select、add 等大量 FlagGems 算子，不是单一 copy 热点。原始 profile 见 `baseline-profile/`。

A57 在 `flag_gems.logging_utils.configure_default_logging()` 中将根 logger 默认设为 INFO；设置 `FLAGGEMS_LOG_LEVEL=DEBUG` 可恢复原行为。已有 `setup_flaggems_logging()` 的显式诊断路径仍设置 DEBUG 并写文件，不改变排障能力。FlagGems 分支 `exp/a57-logging-default`，提交 `eea51ec7`。

相关测试 **122 passed、1 skipped、3 deselected**，见 `gems-tests.log`。候选使用独立 `VLLM_CACHE_ROOT=/workspace/a57-compile-cache`，基线为当前 A48。官方 serve 参数、模型和 benchmark 不变。

官方 4k A1/B1/A2/B2 正在执行，结果待完整对照后填写。回滚为恢复 FlagGems `3642366c` 和 A48 缓存；合入前还需 MATH-500 Level 3 ≥0.95。
