# BI-V150 当前交付复测记录（2026-10-26/27）

## 状态与代码

当前交付为插件 `0207b7e1`、FlagGems `aff68555`，官方 BI-V150 32GB 环境，serve 参数未改。性能 A/B、MATH-500、微基准和 9 个测试均已完成。

## 官方四轮去首均值

| 组别 | 4k tok/s | 4k TTFT ms | 16k tok/s | 16k TTFT ms |
|---|---:|---:|---:|---:|
| A7-1 | 3855.31 | 4240.65 | 3568.31 | 136998.74 |
| A20-1 | 4087.91 | 4327.86 | 4520.19 | 111637.06 |
| A7-2 | 3840.80 | 4218.40 | 3570.61 | 136894.05 |
| A20-2 | 4722.22 | 4139.85 | 4520.10 | 111666.77 |

相对 A7 的增益：4k 为 **6.03%／22.95%**，16k 为 **26.68%／26.59%**。16k TTFT 分别改善 **18.51%／18.43%**。两组均保留，不以单组高值代替整体结果。

## 精度

当前交付两次 MATH-500 Level 3：

- 第一次 `metrics[0].score=0.9619`，105/105 成功、0 错误。
- 第二次 `metrics[0].score=0.981`，105/105 成功、0 错误。

原始文件：`eval1-math500.json`、`eval2-math500.json`、`eval1.log`、`eval2.log`、`eval-serve.log`。

## 内核测试与微基准

- `pytest-a20.log`：BI-V150 上 **9 passed**。
- 等长 kv=4096：stock 0.938592 ms → A20 0.587688 ms，**1.60×**，有效带宽约 **457 GB/s**；maxdiff `3.05e-05`。
- ragged：stock 0.676704 ms → A20 0.439950 ms，**1.54×**；maxdiff `6.10e-05`。
- 原始文件：`micro.json`、`micro.log`、`docs/verification/bi_micro_audit.py`。
- 混合步日志：`used=132000、skipped=0`；未见 decode fast-path fallback warning。采样器的 top-k 回退是既有兼容路径，不是 A20 回退。

## 文件索引

- `<run>-<mode>/serve.log`、`bench.log`、`benchmark_results/`：完整四轮原始 CSV 和官方 summary。
- `<run>-<mode>/16k-serve.log`、`16k-bench.log`：补跑的 16k 原始记录，未覆盖此前 4k 日志。
- `manifest.txt`：提交、实例与依赖信息。
- `run_bi_ab.sh`、`run_bi_16k.sh`、`run_bi_eval.sh`：复现脚本。
- `invalid-mode-a20-1/`、`invalid-a7-1/`：误启动记录，不计入结果。
- 第一次 A7-1 的外层超时记录已保留，后续完整 16k 已补跑。

完整压缩包：`tmp/bi-final-audit.tar.gz`（关机前版本）和 `tmp/bi-final-audit-complete.tar.gz`（补跑完成版本）。
