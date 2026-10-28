# A32：纯 decode 元数据快速路径（2026-10-28）

## 依据

A31 后的 16k host profile 中，`_build_split` 仍是主要热点之一。原实现即使全部请求都是连续 decode 行，也会创建索引、执行 HtoD 拷贝，并对 `seq_lens` 和 `block_table` 做 `index_select`。

A32 在 `n == len(seq_l) and cu_l[n] == n` 时：

- 用 `torch.arange` 生成 decode 行号；
- 直接复用 `m.seq_lens` 和 `m.block_table`，不再 gather；
- 保持 `dec_meta` 的字段和值与原版一致。

混合批继续使用原索引和 gather 路径。该改动只涉及元数据，不改变 attention 计算、调度参数或启动命令。

## 代码与测试

- 分支：`exp/a32-decode-metadata-fastpath`
- 提交：`e1ea441`
- 基线：插件 `0207b7e1` + A31 FlagGems；候选为 A32 + A31 FlagGems。

`tests/unit_tests/dispatch/test_iluvatar_vendor_split.py`：

1. 纯 decode 直接复用元数据。
2. prefill+decode 混合批保持原 gather 语义。

两项测试均通过。

## 微基准

`docs/verification/bi_a32_split_probe.py`，B=64 纯 decode，1000 次 `_build_split`：

| 实现 | 单次耗时 | 结果检查 |
|---|---:|---|
| 原版 | 108.76μs | dec_rows、block_table 正确 |
| A32 | 29.37μs | 直接复用 seq_lens 和 block_table，全部正确 |

约 3.70×。该数字只用于候选筛选，不代表端到端收益。

## 官方验证

4k 四轮去首均值：A1/A2 为 5197.01/5200.34，B1/B2 为 5194.39/5204.85；中位数 **5198.68→5199.62 tok/s（约 +0.02%）**，基本持平，暂不合入。完整证据见 `a32-4k-evidence.tar.gz`。

16k 四轮去首均值：A1/A2 为 4518.85/4512.96，B1/B2 为 4517.02/4514.24；中位数 **4515.91→4515.63 tok/s（约 −0.01%）**，同样基本持平，不合入。完整证据见 `a32-16k-evidence.tar.gz`。

结论：`_build_split` 微基准加速未转化为官方负载收益，说明这些元数据操作在关键路径上并非真正瓶颈。实验分支和测试保留，交付集成不包含 A32。
