# opt10: 消灭 prefill attention 元数据的每层 DtoH 同步

日期：2026-10-07 ｜ 机器：C500 64GB（140.207.205.81）

## 动机（空洞分析实锤）

对 16k prefill trace 做 GPU 空闲分析（analyze_gaps.py）：跨度 19.29s 中
gap>20µs 合计 3.49s（**18.1%**），分布：

| gap 大小 | 次数 | 合计 |
|---|---|---|
| 50-100µs | 6890 | 608ms |
| 100-200µs | 16809 | 2197ms |
| ≥200µs | 1475 | 656ms |

主体是上万个 100-200µs 小空洞 = 主机发射跟不上 GPU（eager 模式的
Python/dispatcher/launch 开销）。其中一类有明确元凶：

## 元凶：每层一次的 `.tolist()` 同步

metax flash_attn 后端的 prefill 分支（flash_attn.py forward）里：

```python
cu_prefix_kv_lens = torch.tensor(
    [0] + attn_metadata.prefill_seq_lens.tolist(),  # GPU→CPU 同步！
    device=..., dtype=torch.int32,
).cumsum(dim=0, dtype=torch.int32)
```

这段在**每层每步**执行：pageable DtoH（tolist 强同步，42µs × 4956 次
= 210ms）+ HtoD 上传 + GPU cumsum（aten::cumsum 主机侧 87µs × 4956 =
429ms）。42 层共享完全相同的输入，纯属重复劳动；且同步会等 GPU 排空，
直接制造流水线气泡。

## 修复（vllm_fl metax flash_attn.py，一处）

- 改为**纯 GPU 计算**：`cat([0], lens).cumsum(int32)`，无任何主机同步；
- 结果**缓存在当步的 attn_metadata 对象上**（`_fl_cu_prefix_kv_lens`），
  42 层只算一次（metadata 每步由 builder 重建，无过期风险）。
- 整数 cumsum，数值与原版完全一致，零精度风险。

## 实测（官方 benchmark，4 跑弃首取均值）

| 场景 | opt8 | opt10 | Δ | 相对官网基线 |
|---|---|---|---|---|
| 4k `[4096,1024,64,256]` | 10946.86 | **12145.31** | **+10.9%** | **+138.6%** |
| 16k `[16384,1024,64,128]` | 10001.65 | **10991.50** | **+9.9%** | **+56.4%** |

- 4k runs：11981.24（弃）/ 12174.57 / 12163.25 / 12098.11
- 16k runs：10996.31（弃）/ 10997.42 / 10998.54 / 10978.54
- TTFT：4k 2174→**1623ms**，16k 20814→**18048ms**（prefill 流水线不再被
  每层同步打断的直接体现）
- 4k 收益大于以往任何单项：4k 场景步数多（256 prompt × 2 chunk + decode），
  每步 42 次同步的固定开销占比更高。
- CSV：`benchmark_results/summary_20261007_145016.csv`
- 精度：MATH-500 Level 3 = **98.1%**，与基线完全一致（纯整数路径改动，无精度风险）。
