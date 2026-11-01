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

官方 A/B/A/B 已完成，案例均为 `[[4096,1024,64,256]]`：A1/A2 为 12527.62/12554.57，B1/B2 为 12572.44/12552.06 tok/s。中位数 **12541.10→12562.25，仅 +0.17%**；TTFT 中位数 1621.56→1615.28ms，满足对照 +1%。提升低于 3% 门槛，A37 **否决、不合入**，不再执行 MATH-500。完整计算见 `a37-official-summary.json`。

回滚：恢复 `vllm_fl/worker/worker.py` 中 `should_record = (rank == 0)`，或将 `ENV_FILE` 切回 `/data/c500-env.sh`。

## 最终栈 GPU 与采样归因

- 短负载 4k/512、64 请求的 rank0 trace 共 17323037 个事件。kernel span 23.27s，GPU busy 17.48s（75.1%），空闲 24.9%。该诊断负载不用于计算官方成绩。
- decode `flash_fwd_splitkv_kernel` 为 4.642s/26838 次，平均 173.0µs，与 4k KV 读取带宽下限一致；不支持再次重写 vendor attention。
- 采样相关 `_hist_zoom` 与 `_masked_sample_partial` 仅约 0.166s。对 `sample_tokens` 单独 cProfile：600 次累计 0.889s，平均约 **1.48ms/次**，其中 fused top-p 0.501s、bookkeeping 0.167s；采样不是 GPU 空闲的主要来源。
- 剔除前 5s 启动/JIT 后，span 18.27s、GPU busy 14.55s（79.6%），总间隙 3.72s。较大的成对间隙为 prefill copy→GEMM 0.690s、DtoH→HtoD 0.478s、sample_reduce→DtoH 0.203s；其余分散在 metadata/copy 和微秒级 launch。没有单项能合理支撑 3% 端到端目标，不能把这 20.4% 全当作可消除开销。
- 因缺少可归因到具体可改路径的证据，暂不重复 A29/A32 或采样融合。原始摘要见 `torch-profile-analysis.txt`、`torch-profile-gaps-steady.txt`、`sampler-host.{prof,txt}`；大 trace 保留在实例 `/data/experiments/runs/c500-current-torch-profile/trace/`。
