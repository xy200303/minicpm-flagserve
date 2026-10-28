# A31：FlagGems 配置缓存与数据库反射开销（2026-10-27）

## 根因证据

当前交付的 4k `ixsys` trace 中，`sub_func_tensor_scalar_kernel_rank_1 -> _index_linearized_jit_function` 前的反复间隙累计约 67.97s。`ModelRunnerFL.execute_model` 的 cProfile 进一步显示：

- `_index_linearized` 调用累计约 66.03s；
- `get_config` 累计约 65.14s；
- SQLAlchemy 全库 reflection 8 次，约 53.76s。

设备热点与 host profile 证明该问题不是 index kernel 计算慢，而是配置读取和模型反射占用了提交线程。A30 的尾页拆分是独立实验，本项从 `integration/flaggems@aff68555` 新建，不包含 A30。

## 改动

实验分支：`exp/a31-config-local-cache`，提交 `e95c5f97`。

1. `ConfigCache` 增加进程内结果缓存，同时记录配置存在和缺失；`set()` 同步更新，`FLAGGEMS_CONFIG_CACHE_LOCAL=0` 可回退到直接数据库读取。
2. `build_sql_model_by_db` 从全库 automap reflection 改为只加载目标表后映射。

测试：`test_config_cache_local_hits_missing_and_set`、`test_sql_model_loads_requested_table`，均通过。

## 真实配置库探针

读取同一张现有配置表并加载模型：

| 实现 | 耗时 | 结果 |
|---|---:|---|
| 原版全库 reflection | 4212.64 ms | 同一模型加载成功 |
| 单表加载 | 26.35 ms | 同一模型加载成功 |

单表加载约为原版的 1/160，但这是初始化/配置读取微基准，不等同于端到端收益。

文件：`reflection-{current,cached}.json`、`pytest-2.log`。

## 候选 host profile

使用与原版相同的 `ModelRunnerFL.execute_model` cProfile 方法：

| 指标 | 原版 | A31 |
|---|---:|---:|
| 累计时间 | 129.68s | 65.17s |
| `_index_linearized` 累计时间 | 66.03s | 4.47s |
| 全库 reflection | 约 53.76s | 不再处于主要耗时中 |

原版证据：`../a30-decode-pipeline/current-4k-host.prof`；候选证据：`host.prof`、`host.txt`。这些数字用于定位 host 瓶颈，不替代官方 benchmark。

## 官方验证

按 A/B/A/B 执行官方 4k 四轮去首均值：

| 运行 | 路径 | 状态 |
|---|---|---|
| A1 | 当前交付 | 4667.18 tok/s，TTFT 4196.06ms |
| B1 初版 | A31 e95c5f97 | 首轮 5187.70，但后续因缺失表异常处理失败，整组无效 |
| A2 | 当前交付 | 4459.34 tok/s，TTFT 4265.06ms |
| B2 | A31 cf11897a | **5209.16 tok/s**，TTFT 4043.57ms，四轮完整 |
| B1 修复复测 | A31 cf11897a | **5208.95 tok/s**，TTFT 4025.01ms，四轮完整；MATH-500 正在执行 |

缺失表问题由 `cf11897a` 修复：原版不存在配置表时返回 `None`，初版单表反射错误抛出 `NoSuchTableError`；修复后保持原回退语义，并补充缺失表测试。初版 B1 不用于任何成绩计算。

B2 四轮原始值依次为 5179.70（弃首）、5209.43、5209.82、5208.22，后三轮均值 5209.16；B1 修复复测为 5208.95。最终中位数：A20 4563.26 → A31 **5209.06 tok/s（+14.15%）**；TTFT 4230.56 → **4034.29ms（−4.64%）**。16k A/B/A/B 已启动，精度和 16k 完成后再合入。
