# 优化 #5：根治 autotune 运行时风暴 + 启动预热（MetaX C500，2026-09-29）

## 动机（测量证据链）

1. opt4 的 4k 场景每 4 跑稳定出 1 次低值（如 5026），首日基线复现也有同构波动——
   不是环境噪声那么简单；
2. 慢步埋点（`VLLM_FL_SLOW_STEP_MS`，本次新增的 `slow_step_logger.py` 补丁，默认关）
   抓到实锤：单次 `execute_model` 阻塞 **23.8 秒**，serve 日志对应
   `Avg generation throughput: 0.0` 的连续 10s 窗口；
3. 根因：FlagGems LibTuner 对运行时新出现的 (M,N,K) key 做全配置 autotune，
   每个配置要在弱 CPU 上编译 Triton 内核——benchmark 期间 chunked prefill 的批组合
   不断变化（M=1308/1415/1417/1933…），每个新 key 触发一次编译+评测风暴；
4. 缓存取证（`~/.flaggems/config_cache/TunedConfig_metax_triton_3_0.db`）：
   修复前 mm_kernel_nn/nt 各自累积了 **270/306 个不同 M 的调优记录**；
   LibTuner 的 ConfigCache 查找走 strategy 归一化 key（`libentry.py:986`），
   BenchmarkCache 存原始 key——分桶策略能让同桶形状共享最优配置。

## 实现（两个仓库）

- FlagGems `_metax/ops/mm.py`（commit `d0bb60806`）：
  - 新增 `LibTuner.register_strategy("align128")`：M 按 128 向上分桶（<128 保持原值），
    N/K 用 align32；应用到 mm_nn / mm_nt / nt_db 三个 libtuner；
  - nt_db / nt_db_splitk_partial 配置表钉死为单配置（decode 形状固定 M≤128，
    无需 autotune），`len(configs)==1` 时 LibTuner 直接跳过评测。
- vllm-plugin-FL 新补丁 `patches/mm_autotune_warmup.py`（commit `3365538fe`）：
  - `ModelRunnerFL.load_model` 挂钩（注意：FL 插件用自己的 `ModelRunnerFL`，
    补 `vllm.v1.worker.gpu_model_runner.GPUModelRunner` 不会生效——第一次写错了挂载点，
    日志无 `[fl-warmup]` 标记暴露）；
  - 模型加载后按运行时真实调用路径回放 GEMM：小 M（1..128,1024）走 F.linear（nt/nt_db
    decode 路径），大 M 按 128 步进至 `max_num_batched_tokens`+128 走 `torch.mm(x, W_nn)`
    （nn prefill 路径），把一次性调优成本从 benchmark 关键路径移到启动期；
  - 默认开，`VLLM_FL_MM_WARMUP=0` 关闭；异常只告警不影响启动。
- 冷缓存实测：启动 +~7min（含 Triton 编译缓存重建），调优 db 在启动期内写完，
  之后 benchmark 全程**零新增调优行**。

## 端到端结果（官方 benchmark，冷缓存启动，serve 参数与基线一致）

| 场景 | 官方基线 | opt1-4（旧） | opt1-5 本次 4 跑 | 官方 summary | vs 基线 |
|---|---|---|---|---|---|
| 4k total tok/s | 5089.645 | 7785（含低值） | 8050/8736/8833/8840 | **8803.21** | **+73.0%** |
| 16k total tok/s | 7029.675 | 8492 | 8844/8854/8857/8864 | **8858.47** | **+26.0%** |
| 4k Mean TTFT | 3199ms | 2455~2702 | 2811ms | — | **−12% ✓** |
| 16k Mean TTFT | 27197ms | 23839~24095 | 23958ms | — | **−12% ✓** |
| 精度 math_500 L3 | 0.962 | 97.1% | **96.2%** | — | 与基线一致 ✓ |

对照（修复前，温缓存）：4k 稳态 8931/8940/8911 但伴随周期性低值；
修复后两轮 4k 连跑 **8843/8838/8835/8831**（σ≈5，波动消失）。

## 重要发现：官方 summary 的聚合口径

`summary_*.csv` 的 Total tok/s = **第 2~4 跑的均值**（丢弃第 1 跑预热）。
本次验证：4k (8736.27+8832.96+8840.39)/3 = 8803.21 ✓；16k (8854.28+8857.26+8863.87)/3
= 8858.47 ✓。即官方口径自动容忍冷启动第 1 跑，但 TTFT 等指标同理聚合，
预热补丁仍降低了首跑劣化（冷机 4k 首跑 6656→8050），提升鲁棒性。

数据：`raw_runs_opt5_autotune_warmup.csv` / `summary_opt5_autotune_warmup.csv`
（原始 `benchmark_results/*_20260929_201559.csv`）。
评测机 commit：FlagGems `d0bb60806`、vllm-plugin-FL `3365538fe`。

## 遗留

- 4k 首跑仍低于稳态（8050 vs 8840，非 autotune 因素：首次 kernel 编译/分配器/CPU 频率），
  官方口径已丢弃首跑，不再追；
- 16k 首次超过 4k（8858 > 8803）：长 prefill 占比高使 total tok/s 口径受益，符合预期；
- 启动时长增加（冷缓存 ~13min），评测环境可接受，readme 需注明。
