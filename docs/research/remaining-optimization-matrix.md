# 剩余加速方案研究矩阵

更新时间：2026-10-28。目标不是穷举参数，而是为每个方向保留可复核的采纳、否决或阻塞依据。

| 方向 | 已完成与结论 | 后续状态 |
|---|---|---|
| BI prefill attention | A1/A6 vendor ixAttnBkd、varlen、dual-gather，已形成主要收益 | 接近 vendor 上限；仅接受新实现证据 |
| BI decode attention | A7 pin、A20 页对齐内核；A30/A33 否决；A36 全页/尾页分离微基准 1.08–1.14× | A36 仅完成官方 A1，实例关机导致 B1 无汇总；需新实例重做交错 A/B，未合入 |
| C500 attention | vendor prefill/decode 已达约 1.42 TB/s，接近实测纯读上限 | 不照搬 BI A20；等待最终栈新 profile |
| GEMM | C500 nn/mcBLAS；BI pinned vendor config；W8 集成回退；A34 GROUP_M 两轮仅 0–2.87%，复测最高 1.73%，否决 | 不再重复 tile 顺序；仅接受新 epilogue/访存机制证据 |
| 采样 | bf16 直读、温度折叠、免排序 top-p、Gumbel-Max 已采纳 | 当前占比较低，不优先 |
| KV Cache / 布局 | 页对齐、静态 slab、dual-gather、混合步 decode 已采纳 | int8/fp16 无收益；不再重复 |
| Norm / RoPE / elementwise | 走现有融合路径；A21 IR 融合因破坏 Inductor 融合而否决；A34 确认 BI 无 FlagGems IR 注册实现 | 停止 opaque custom op；仅研究原生可融合改动 |
| Host / metadata 同步 | C500 GPU cumsum、BI builder 修复已采纳 | A32 调用级优化无端到端收益，需先证关键路径 |
| 配置与 JIT | A31 配置缓存、单表反射已采纳，4k +14.15% | 继续分析 `libentry.run` 的剩余 CPU 成本 |
| CUDA graph | 图安全 NSEG、device seq_lens、预分配 slab 已采纳 | A28/A29 无收益；只接受新图内证据 |
| 调度 / 混合批 | prefill/decode 拆分、KV 容量分析、路径回退已覆盖 | 不改 serve 超参；投机调度不可用 |
| 量化 / 投机采样 | W8、int8 KV、投机均实测否决 | 不重复，除非出现新的芯片特化证据 |
| 正确性 / 泛化 | 官方 4k/16k、MATH、B=16/32/64、等长/ragged 已覆盖 | 长稳和极端分布仍可补充，不替代官方分数 |

## 当前优先研究项

1. BI decode 页内核的分支、寄存器和中间结果流量；A34 已关闭 GEMM tile 顺序方向。
2. A31 后 `libentry.run` 与 metadata/copy 的剩余关键路径，避免 A32 式无效优化。
3. C500 最终栈的图内归因，当前缺少可用实例，不能用旧 profile 代替。

## 实验纪律

每个新候选必须有独立 exp_id、配置、serve/bench 日志、原始 CSV/JSON、机器状态、数值检查和回退方式。微基准只用于筛选；最终采纳仍要求官方交错 A/B、TTFT 和 MATH-500。
