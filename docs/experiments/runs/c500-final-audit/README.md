# C500 当前交付复测记录（2026-10-25）

## 代码与实例

- 交付插件：`0207b7e10f93dfa6e912b00c372f268d618b4c2c`
- 交付 FlagGems：`aff68555e266582f68c6554898f4c795c869578d`
- 原生基线：插件 `13eb9be`，FlagGems tag v5.3.5（`a7620cc1`）
- 健康复测实例：MetaX C500 64GB，CPU 内存 128GiB
- 源码路径：`PYTHONPATH=/workspace/audit-plugin:/workspace/audit-gems/src`

## 健康实例官方 benchmark

原生基线与交付版分别使用独立源码和 venv，serve 参数与官网一致。每项连续运行四轮，官方脚本丢弃首轮后对后三轮取均值。原始 CSV 已逐行复算，与 summary 一致。

| 配置 | 场景 | 总吞吐 tok/s | 平均 TTFT ms | 启动 s |
|---|---|---:|---:|---:|
| 原生基线 | 4k | 4210.20 | 3944.83 | 161 |
| 当前交付 | 4k | **12399.31** | **1719.18** | 293 |
| 原生基线 | 16k | 5867.59 | 33566.37 | 161 |
| 当前交付 | 16k | **11046.19** | **18141.47** | 293 |

当前交付相对本机原生基线为 4k **2.95×**、16k **1.88×**；相对官网基线 5089.645／7029.675 为 **2.44×／1.57×**。本机原生基线明显慢于官网基线，两种参照不得混用。

- `healthy-baseline-{raw,summary}.csv`、`healthy-baseline-{serve,bench}.log`
- `healthy-final-{raw,summary}.csv`、`healthy-final-{serve,bench,driver}.log`
- `healthy-final-manifest.txt`、`healthy-run-final.sh`
- 当前交付后三轮吞吐极差：4k 0.80%、16k 0.29%；serve 日志中的 prefix cache 命中率低于 1%。

## MATH-500 Level 3

当前交付在同一服务上执行两次官方 evalscope 命令，每次均为 105/105 请求成功、0 错误，精度均为 **98.1%**。原始 JSON 的 `metrics[0].score=0.981`，`execution_summary.succeeded=105`。

- `healthy-eval1-math500.json`、`healthy-eval1.log`
- `healthy-eval2-math500.json`、`healthy-eval2.log`
- `healthy-eval-serve.log`、`healthy-run-eval.sh`

## 健康实例微基准

脚本：`docs/verification/c500_micro_audit.py`；原始输出：`micro-healthy.json`、`micro-healthy.log`。计时使用 CUDA event，表中为 30 次中位数。

### top-p 采样

形状：B=64、V=130560、bf16、temperature=1.0、top_p=0.95。PyTorch 对照在 `eager_top_p` 中实现，包括 fp32 softmax、排序、累积质量掩码、归一化和 multinomial；当前实现为 `fused_top_p_sample(..., zoom_rounds=1)`。

| 路径 | 中位耗时 | 均值 | 最小–最大 |
|---|---:|---:|---:|
| 同语义 PyTorch top-p | 4.085760 ms | 4.094242 ms | 4.071680–4.241664 ms |
| 当前 fused top-p | 0.380800 ms | 0.384691 ms | 0.367872–0.470528 ms |
| 加速比 | **10.73×** | — | — |

### lm_head GEMM

形状：M=64、K=2048、N=130560、bf16。先在 FlagGems 全局注册前测原生路径，再执行 `flag_gems.enable()` 测 Triton，最后直接调用 `mcblas_mm`。

| 路径 | 中位耗时 | 均值 | 最小–最大 |
|---|---:|---:|---:|
| 原生 `torch.mm` | 0.385024 ms | 0.386236 ms | 0.380672–0.415232 ms |
| FlagGems Triton | 0.666880 ms | 0.669561 ms | 0.650752–0.727040 ms |
| 直接 mcBLAS | 0.430464 ms | 0.431309 ms | 0.418560–0.456192 ms |

mcBLAS 相对 FlagGems Triton 快 **1.55×**，但比同次原生 `torch.mm` 慢约 11.8%。早期实例的微基准保留在 `micro.json`，不作为当前报告数字。

## 过程异常与无效数据

- 首个异常实例中，交付版和原生基线均在 1200 秒内未完成初始化，未进入 benchmark。对应 `serve-startup-failure.log` 与 `baseline-serve-startup-failure.log`。
- 健康实例第一次交付部署时，`--system-site-packages` 中的基线 `vllm_fl` 覆盖了 editable 安装，导致误跑基线。对应 `invalid-wrong-import-{serve,driver}.log`，不用于报告；随后固定 `PYTHONPATH` 并确认 `mcblas_mm` 与采样融合实际加载后重跑。
- 一次预热超过 1200 秒外层等待上限，未产生 benchmark，见 `invalid-warmup-timeout-{serve,driver}.log`。保留缓存后重启，启动完成，官方测试参数和预热均未削减。
- `micro-native-first.json` 是未区分原生与 Triton 的初步微基准，已由健康实例的三路结果取代。

## 收尾

官方 benchmark、两次精度评测和健康实例微基准均已完成。所有 serve 和微基准进程已停止，`mx-smi` 无进程，显存回落至 826 MiB。
