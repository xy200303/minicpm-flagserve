# C500 后续优化：A37 算子清单日志（2026-11-01）

## 当前交付基线

- 实例：MetaX C500 64GB，128GB RAM，MACA 3.7.0.38。
- 插件 `0207b7e1`，FlagGems `9c06143c`，从 Git bundle 克隆到 `/data/src`。
- 官方 4k A1：**12527.62 tok/s**，Mean TTFT **1623.73ms**；原始四轮 CSV、serve/bench 日志、机器状态见 `c500-current-4k-a1/`。
- 首次冷启动 1129 秒，用于生成 autotune 配置；后续热缓存启动约 127 秒。冷启动不计入吞吐。

## 热点证据

对当前交付栈执行官方 4k 诊断并包装 `ModelRunnerFL.execute_model` 做 cProfile。诊断只用于归因，不作为性能成绩。

最终 host profile 累计 218.867 秒，其中：

- `logging.debug`：**28.140s / 719401 次调用**；
- `Logger._log`：26.128s；`makeRecord`：15.041s；`LogRecord.__init__`：13.914s。
- 主要调用来源是 FlagGems `copy_`、`transpose`、`unbind`、`fill_scalar_` 等高频算子。
- `FLAGGEMS_ENABLE_OPLIST_PATH` 默认启用清单记录。`once=True` 只在 handler 前去重，不能避免每次调用构造 `LogRecord`；最终文件只有 29 行，却构造了 719401 次 debug 记录。

原始证据：`c500-current-4k-host/`、`c500-current-4k-host.tar.gz`；早期快照 `host-snapshot.{prof,txt}` 仅用于中途检查，结论以最终文件为准。

## A37 候选

分支 `exp/a37-oplist-opt-in`，提交 `cdf952c`。只有 rank 0 且显式设置 `FLAGGEMS_ENABLE_OPLIST_PATH` 时才记录算子清单；默认服务不再构造这些 debug 记录。显式设置后仍使用原路径和 once 行为。模型计算、算子选择、采样与 serve 参数不变。

官方序列：已有 A1，随后 B1/A2/B2，案例均为 `[[4096,1024,64,256]]`。至少两组中位数吞吐提升 ≥3%、TTFT 满足对照 +1% 后，再执行 MATH-500 Level 3；未完成前不合入。

回滚：恢复 `vllm_fl/worker/worker.py` 中 `should_record = (rank == 0)`，或将 `ENV_FILE` 切回 `/data/c500-env.sh`。
