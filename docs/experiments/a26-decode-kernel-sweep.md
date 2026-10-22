# A26: 官方 16k 当日基线 + decode attention 内核形状扫描（shape-aware dispatch 的否定结论）

日期: 2026-10-22  卡: 天数 BI-V150 32GB（开发机）  栈: vllm-plugin-FL b0cda5e（A20 交付态）

## 1. 官方 16k 当日基线（A20）

`vllm bench serve --random-input-len 16384 --random-output-len 1024 --max-concurrency 64 --num-prompts 128`：

| 指标 | 值 |
|---|---|
| Successful requests | 128/128 |
| Total token throughput | **3515.89 tok/s** |
| Mean TTFT | 130772.51 ms（130.8 s） |
| Mean TPOT | 126.49 ms |

历史同期记录 3541.34 tok/s（TTFT 155.5 s）→ 当日 3515.89，差 −0.7%，属正常机器波动。
**这是后续 16k 实验的当日对照基线。**

## 2. 内核形状扫描

`flag_gems.fused.paged_decode_attention` 在 (B, kv) × NSEG 上逐点测（20 reps，block_n=16，
warps=1，stages=1，page_aligned=1，随机 block_table，dtype bf16）：

| B | kv | NSEG=16 | NSEG=32（当前默认） | NSEG=64 | NSEG=128 |
|---|---|---|---|---|---|
| 8 | 4096 | **0.168** | 0.173 | 0.190 | 0.325 |
| 8 | 16384 | **0.363** | 0.392 | 0.390 | 0.509 |
| 16 | 4096 | 0.248 | **0.242** | 0.258 | 0.438 |
| 16 | 16384 | 0.679 | 0.651 | **0.632** | 0.800 |
| 32 | 4096 | 0.377 | **0.368** | 0.403 | 0.639 |
| 32 | 16384 | 1.194 | **1.121** | 1.124 | 1.347 |
| 64 | 4096 | **0.620** | 0.629 | 0.703 | 1.168 |
| 64 | 16384 | 2.131 | **2.084** | 2.121 | 2.566 |

（单位 ms/layer，28 层）

## 3. 结论：官方并发下没有可分的档

- **B=64（官方 c64）时，kv=16384 的最优点就是当前的 NSEG=32**；kv=4096 换 16 只快 **1.4%**。
- NSEG=128 在所有形状上都明显更慢（+30%~90%），说明 reduce 开销已经压过并行收益。
- 小 batch（B≤16）确实能靠换 NSEG 拿到 **3%–7.4%**，但官方两个用例都是 c64，
  且请求数量固定，decode 阶段 B 稳定在 64——**分档表在官方口径下发不出收益**。

带宽视角（B=64）：

| kv | 最佳 GB/s | 卡的理论带宽 | 利用率 |
|---|---|---|---|
| 4096 | 433 | 587 GB/s | 74% |
| 16384 | 515 | 587 GB/s | **88%** |

16k（decode 占主导的那个用例）已经跑到 88% 带宽，剩余 12% 是清零/写回等固定开销，
继续压 NSEG/BLOCK 也只能拿到个位数百分比——**这是 kernel 层的实际天花板**。

## 4. 处置

**不实现 shape-aware dispatch。** 理由：

1. 在官方 c64 口径下，分档表的收益上限是 1.4%（4k）与 0%（16k），远低于 3% 准入；
2. 实现它要引入"按 B 查 NSEG + slab 按最大 NSEG 分配"，改动的复杂度与风险
   （图捕获期 slab 尺寸、越界）与 1.4% 的收益不匹配；
3. 组委会允许这类做法（10-12 已明确"可以的"），但允许不等于值得做——
   这一条应作为"已评估并放弃"写进技术报告，而不是当作未探索的空白。

## 5. 复现

```bash
cd /workspace/opt_experiments && python3 a26_sweep.py | tee a26_sweep.log
# 16k 官方用例
vllm bench serve --backend vllm --model minicpm --tokenizer /workspace/MiniCPM5-2B \
  --endpoint /v1/completions --host localhost --port 9031 --dataset-name random \
  --ignore-eos --random-input-len 16384 --random-output-len 1024 \
  --max-concurrency 64 --num-prompts 128
```

产物：`/workspace/opt_experiments/a26_sweep.py`、`a26_sweep.log`、`a20_16k_run1.log`。
