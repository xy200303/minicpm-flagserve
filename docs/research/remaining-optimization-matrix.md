# 剩余加速方案研究矩阵

更新时间：2026-11-07。目标不是穷举参数，而是为每个方向保留可复核的采纳、否决或阻塞依据。

| 方向 | 已完成与结论 | 后续状态 |
|---|---|---|
| BI prefill attention | A1/A6 vendor ixAttnBkd、varlen、dual-gather，已形成主要收益 | 接近 vendor 上限；仅接受新实现证据 |
| BI decode attention | A7 pin、A20 页对齐内核；A30/A33 否决；A36 微基准正收益但官方 −12.46%，否决；A44 combine 在 16k 仅占 1.34%，收益上限不足 | 组合回退已证实，不再采用全页/尾页分离或 combine 消除 |
| C500 attention | vendor prefill/decode 已达约 1.42 TB/s，接近实测纯读上限 | 不照搬 BI A20；等待最终栈新 profile |
| GEMM | C500 nn/mcBLAS；BI pinned vendor config；A34/A38 配置复测否决；A48 双累加器门控融合官方 4k/16k 为 +0.679%/+0.707%；A60 K 连续 tile 预打包在 A48 上继续取得 +1.448%/+1.513%，两者均已合入；A49 lm_head 最佳配置仅快 0.34%；A58 decode warps=16 微基准快 3.64%，官方整体 −0.0336% | 保留 A48+A60；不重复无新机制的 tile/warp 扫描 |
| 采样 | bf16 直读、温度折叠、免排序 top-p、Gumbel-Max 已采纳 | 当前占比较低，不优先 |
| KV Cache / 布局 | 页对齐、静态 slab、dual-gather、混合步 decode 已采纳；A50 厂商 batch paged 接口执行失败，A51 MTP 接口与 NHD 布局不兼容 | 不以全局 HND 改造换取有限 gather 收益；需要新的厂商可用接口证据 |
| Norm / RoPE / elementwise | 走现有融合路径；A21 IR 融合因破坏 Inductor 融合而否决；A56 真实图审计确认 RoPE、residual+RMSNorm 与 A48 前置链均已融合 | 原生融合方向关闭，不新增 opaque custom op |
| Host / metadata 同步 | C500 GPU cumsum、BI builder 修复已采纳；A54 在 A48 栈上两场景平均 −0.2397%，A55 在 A48 栈上 16k −0.478%，均否决 | host 累计时间不等于关键路径，不再叠加同类微优化 |
| 配置与 JIT | A31 配置缓存、单表反射已采纳，4k +14.15%；A46 LibEntry 微基准收益未转化为官方收益 | 保持 A31，不重复 LibEntry 壳优化 |
| CUDA graph / A20 配置 | 图安全 NSEG、device seq_lens、预分配 slab 已采纳；A52/A53 确认 NSEG=16 在等长 4k 仅快约 1.5%，16k 回退，更小 NSEG 更慢 | NSEG=32 保持；不再扫描 4/8/16/32 |
| 调度 / 混合批 | prefill/decode 拆分、KV 容量分析、路径回退已覆盖 | 不改 serve 超参；投机调度不可用 |
| 量化 / 投机采样 | W8、int8 KV、投机均实测否决 | 不重复，除非出现新的芯片特化证据 |
| 正确性 / 泛化 | 官方 4k/16k、MATH、B=16/32/64、等长/ragged 已覆盖 | 长稳和极端分布仍可补充，不替代官方分数 |

## 当前研究状态

A48 与 A60 均已纳入交付。A60 的 K 连续 tile 预打包在 A48 上使官方 4k/16k 吞吐分别提高 1.448%/1.513%，MATH-500 Level 3 为 96.19%；4k TTFT 改善 2.06%，16k TTFT 较 A48 对照增加 7.63%，但仍远低于官网基线及其 +1% 上限。3% 仅是此前研究 goal 的自定义筛选线，不是官方合入门槛。A54/A55 已复核小幅 host 候选与 A48 的组合，均未复现旧栈正收益；A56 已关闭 Inductor 原生融合方向。第二轮 host/ixsys 复核中，A57 日志优化官方 −0.2394%，A58 decode warps 调优官方 −0.0336%，均不合入。当前 BI-V150 研究矩阵没有待验证项，后续仅在接受新的芯片特化机制或厂商可用实现证据时重开对应方向。

## 实验纪律

每个新候选必须有独立 exp_id、配置、serve/bench 日志、原始 CSV/JSON、机器状态、数值检查和回退方式。微基准只用于筛选；最终采纳仍要求官方交错 A/B、TTFT 和 MATH-500。
