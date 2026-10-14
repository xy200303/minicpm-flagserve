# 天数 BI-V150 A5：sampler 尾部融合（C500 opt7 移植）

日期：2026-10-14 ｜ 实例：FlagOS OnlineLab BI-V150（32GB 显存版）

## 起因

A3 的 stacked trace 显示每步 decode 在 host 侧花 ~29ms 做采样尾部：
eager 路径对 [64,130560] logits 做 cast fp32 → div temperature →
**全词表 sort（top-p）** → gumbel-argmax。C500（opt1/opt7）已验证
sort-free 融合内核是同形问题的高效解（C500 4k +27%），直接移植。

## 移植内容（与 C500 代码完全一致，零芯片改动）

1. **FlagGems** `fused/sampler_fused.py` + `fused/top_p_sample.py`
   （已在 FlagGems/dev，C500 commit e776b2ed/d890dc42）：
   - bf16 直读 lm_head 输出，temperature 折叠进 Gumbel-Max
     （argmax(x/T+g) == argmax(x+T·g)，T<1e-5 行退化为 argmax）
   - top-p 用单轮 1024-bin 质量直方图 zoom 求阈值 tau（取跨越 bin
     下沿，保留集质量恒 ≥ p·Z），替代全词表 sort
   - 词表维 split-parallel（32 splits × BLOCK 4096），batch=64
     不打不满芯片
2. **vllm-plugin-FL** `iluvatar/impl/sampler_tail_fusion.py`
   （commit 72a83bf3，patch 本体与 MetaX 版逐字节一致）：
   `Sampler.forward` 快路径——无 logprobs/penalties/top-k/
   logitsprocs/generators 的"纯采样"批次走融合内核，其余回退
   stock。`VLLM_FL_DISABLE_SAMPLER_FUSION=1` 可 A/B。

## 天数实测（[64,130560] bf16，单 decode 步）

| 路径 | eager | fused | 加速 |
|---|---|---|---|
| top-p 采样全链 | 12394µs | **666µs** | **18.6x** |
| greedy argmax | 699µs | 97µs | 7.2x |

正确性：greedy 与 torch.argmax 100% 一致；top-p 采样全部落在核集
（+直方图粒度 slack）内；与 stock 核采样 TV 距离 0.206（N=4000，
优于 C500 验收值 0.29，纯采样噪声）。

## 端到端（官方 benchmark，4 跑弃首取后 3 均值）

| 场景 | A4 | A5 | Δ |
|---|---|---|---|
| 4k total tok/s | 2579.14 | 2587.84 | +0.34% |
| 16k total tok/s | 2485.52 | **2532.68** | **+1.90%** |

16k 三次有效跑 2532.72/2532.60/2532.72，组内极差 0.005%，与 A4
区间（2484.9-2486.4）完全不重叠——收益是实的。16k 提升大于 4k
符合预期：16k decode 步占比更高，采样尾部每步省 ~11.7ms GPU。
TTFT 16k 191.8s→188.8s，无回退。

精度实测 **97.1%**（MATH-500 L3，105 题，evalscope 同口径），
与 A3/A4 完全一致，≥0.95 达标。直方图 zoom 的阈值粒度对生成
质量无可测影响。

## 备注

- 测试插曲：自写单测曾报 TV=0.99"内核错误"，实为测试脚本把
  sorted 序的 keep 掩码乘到未排序 prob 上；内核一次通过
- FlagGems 侧无需新 commit：内核为后端无关 Triton，FlagGems/dev
  已含同文件
