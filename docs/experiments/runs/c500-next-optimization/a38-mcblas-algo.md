# A38：mcBLAS tensor-op 算法扫描（2026-11-01，否决）

当前 `mcblas_mm` 固定 `MCBLAS_GEMM_DEFAULT(-1)`。依据 `/opt/maca/include/mcblas/mcblas.h` 中公开的 `99/100–115` tensor-op 算法，对 MiniCPM5-2B 的 8 个 GEMM shape 逐一测量，每点 20 次 CUDA event 计时并检查与默认输出差异。

## 全扫描结果

| shape | 首轮最佳 algo | 首轮加速比 | max diff |
|---|---:|---:|---:|
| decode64_qkv | 106 | 1.0780× | 0 |
| decode64_o | 111 | 1.0780× | 0 |
| decode64_gateup | 107 | 1.0228× | 0 |
| decode64_down | 104 | 1.0487× | 0 |
| prefill_qkv | 106 | 1.0133× | 0 |
| prefill_o | 112 | 1.0117× | 0 |
| prefill_gateup | 107 | 1.0024× | 0 |
| prefill_down | 109 | 1.0045× | 0 |

原始文件：`a38-<shape>.json`。这些首轮结果存在顺序和热状态偏差，不能直接用于路由。

## 复测与否决

短列表复测时 qkv/o 的收益分别降至约 3.0%/0.3%，见 `a38-*-confirm.json`。进一步执行 `-1,algo` 三轮交替、每点 30 次：

- qkv：默认中位数 0.040448ms，algo106 中位数 0.041344ms，候选反而慢约 2.2%。
- o：默认中位数 0.037120ms，algo111 中位数 0.036352ms，候选仅快约 2.1%，不足以推断端到端 ≥3%。

因此 A38 否决，不修改集成代码，不执行官方长测。`best_speedup` 字段使用首轮默认作分母，只保留为原始探针输出；本结论按全部交替轮次的中位数重新计算，未直接采用该字段。

探针：`c500_mcblas_algo_probe.py`、`c500_a38_probe_chain.sh`、`c500_a38_confirm_chain.sh`。
