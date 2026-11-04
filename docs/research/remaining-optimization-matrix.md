# 剩余加速方案研究矩阵

更新时间：2026-11-03。目标不是穷举参数，而是为每个方向保留可复核的采纳、否决或阻塞依据。

| 方向 | 已完成与结论 | 后续状态 |
|---|---|---|
| BI prefill attention | A1/A6 vendor ixAttnBkd、varlen、dual-gather，已形成主要收益 | 接近 vendor 上限；仅接受新实现证据 |
| BI decode attention | A7 pin、A20 页对齐内核；A30/A33 否决；A36 微基准正收益但官方 −12.46%，否决；A44 combine 在 16k 仅占 1.34%，收益上限不足 | 组合回退已证实，不再采用全页/尾页分离或 combine 消除 |
| C500 attention | vendor prefill/decode 已达约 1.42 TB/s，接近实测纯读上限 | 不照搬 BI A20；等待最终栈新 profile |
| GEMM | C500 nn/mcBLAS；BI pinned vendor config；A34/A38 配置复测否决；A48 双累加器门控融合官方 4k/16k 为 +0.679%/+0.707%，作为小幅正收益合入；A49 lm_head 最佳配置仅快 0.34% | 不重复低收益 tile/algorithm 扫描；保留已验证的 A48 |
| 采样 | bf16 直读、温度折叠、免排序 top-p、Gumbel-Max 已采纳 | 当前占比较低，不优先 |
| KV Cache / 布局 | 页对齐、静态 slab、dual-gather、混合步 decode 已采纳；A50 厂商 batch paged 接口执行失败，A51 MTP 接口与 NHD 布局不兼容 | 不以全局 HND 改造换取有限 gather 收益；需要新的厂商可用接口证据 |
| Norm / RoPE / elementwise | 走现有融合路径；A21 IR 融合因破坏 Inductor 融合而否决；A34 确认 BI 无 FlagGems IR 注册实现 | 停止 opaque custom op；仅研究原生可融合改动 |
| Host / metadata 同步 | C500 GPU cumsum、BI builder 修复已采纳；A45 官方 +0.78%，A46/A47 官方 4k/16k 为 +0.0049%/+0.12%，均不采纳 | host 累计时间不等于关键路径，不再叠加同类微优化 |
| 配置与 JIT | A31 配置缓存、单表反射已采纳，4k +14.15%；A46 LibEntry 微基准收益未转化为官方收益 | 保持 A31，不重复 LibEntry 壳优化 |
| CUDA graph / A20 配置 | 图安全 NSEG、device seq_lens、预分配 slab 已采纳；A52/A53 确认 NSEG=16 在等长 4k 仅快约 1.5%，16k 回退，更小 NSEG 更慢 | NSEG=32 保持；不再扫描 4/8/16/32 |
| 调度 / 混合批 | prefill/decode 拆分、KV 容量分析、路径回退已覆盖 | 不改 serve 超参；投机调度不可用 |
| 量化 / 投机采样 | W8、int8 KV、投机均实测否决 | 不重复，除非出现新的芯片特化证据 |
| 正确性 / 泛化 | 官方 4k/16k、MATH、B=16/32/64、等长/ragged 已覆盖 | 长稳和极端分布仍可补充，不替代官方分数 |

## 当前研究状态

A48 在两种官方场景均取得小幅正收益且 TTFT 改善，补齐两组 16k A/B 后纳入交付；3% 仅是此前研究 goal 的自定义筛选线，不是官方合入门槛。lm_head 配置、厂商 paged prefill 接口和 A20 NSEG 配置线已关闭；后续优先研究新的芯片特化机制、可用的厂商 paged prefill 实现，或能解释完整模型收益的 Inductor 原生融合证据。

## 实验纪律

每个新候选必须有独立 exp_id、配置、serve/bench 日志、原始 CSV/JSON、机器状态、数值检查和回退方式。微基准只用于筛选；最终采纳仍要求官方交错 A/B、TTFT 和 MATH-500。
