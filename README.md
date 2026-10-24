# MiniCPM-FlagServe

基于 FlagOS 统一栈（vllm-plugin-FL + FlagGems）的 MiniCPM 大模型多芯片高吞吐推理优化：面向天数 BI-V150 / 沐曦 C500 等国产加速卡的算子融合、内核重写与 KV/调度优化。

> 当前为 FlagOS 开放计算全球大赛 S2 赛道二的参赛工作区（XLANG2026, @xy200303）。

FlagOS 开放计算全球大赛 S2 · 赛道二（MiniCPM5-2B 推理吞吐优化）· 队伍 XLANG2026（邓乾 @xy200303）

容器可能随时销毁，代码和实验记录必须当天推送持久化。源码现在维护在两个独立的上游 fork 中；本仓库只负责报告、实验台账和复现工具。

## 仓库与分支布局

| 仓库 | 用途 | 主要分支 |
|---|---|---|
| [xy200303/vllm-plugin-FL](https://github.com/xy200303/vllm-plugin-FL) | `flagos-ai/vllm-plugin-FL` 的正式 fork，插件与双平台集成开发 | `integration/vllm-plugin-fl`（当前交付）；`baseline/vllm-plugin-fl-flagos-2026-s2`（官方基线）；`archive/*`（历史开发线） |
| [xy200303/FlagGems](https://github.com/xy200303/FlagGems) | `flagos-ai/FlagGems` 的正式 fork，采样、GEMM、A20 等算子开发 | `integration/flaggems`（当前交付）；`baseline/flaggems-v5.3.5`（官方基线）；`archive/*`（历史开发线） |
| [xy200303/minicpm-flagserve](https://github.com/xy200303/minicpm-flagserve) | 报告、实验台账、复现文档和工具 | `main` |

后续实验从两个 `integration/*` 分支分别创建 `exp/<编号>-<简述>`，验证通过后回合；基线分支只用于 A/B，归档分支不再继续写入。两个 fork 均可直接向上游提交 PR。

## 评测机上恢复环境

```bash
cd /workspace
git clone --single-branch --branch integration/vllm-plugin-fl \
  https://github.com/xy200303/vllm-plugin-FL.git vllm-plugin-FL
git clone --single-branch --branch integration/flaggems \
  https://github.com/xy200303/FlagGems.git FlagGems
# 基线分支用于 A/B；安装：pip install --no-build-isolation -e .
```

## 关键约束（官网口径 + 赛题组答疑）

- vllm serve 参数与基线一致；禁止 benchmark 调参/直接 merge 上游新分支/使用 MiniCPM5-2B 以外的模型
- 精度 math_500 Level3 ≥ 0.95（基线 0.962）；TTFT ≤ baseline+1%；total tok/s ≥ baseline−1%
- 沐曦 C500 64GB 基线：4k = 5089.645 tok/s（TTFT 3199ms）；16k = 7029.675 tok/s（TTFT 27197ms）
- 天数 BI-V150 基线：4k = 2028.01；16k = 915.15（TTFT 599262ms）
- 提交截止 2026-11-20 23:59；获奖后 3 个工作日内向 flagos-2026-s2 提 PR

### 答疑澄清（赛题组陈卓，按时间）

- **2026-10-08 投机采样**：可以"改进投机采样策略本身"（如无 draft 的
  n-gram/prompt-lookup 策略改进）；**自训 draft 模型不支持**（不属于
  框架层面优化，且违反"禁用其他模型"）
- **2026-10-10 shape-aware dispatch**：允许针对同一算子实现多个等价
  Triton kernel / 执行配置，按输入 shape 动态选择（不修改模型行为、
  benchmark、serve 参数，不删除框架算子选择逻辑）——A4/A7 的合规依据
- **量化（shchy 问答）**：禁止"修改超参直接启用框架能力"（含超参开
  量化/投机）；**自研**优化策略（含自写量化算子）集成到框架不限制，
  优先鼓励泛化能力高的策略
