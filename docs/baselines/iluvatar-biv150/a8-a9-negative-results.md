# 天数 BI-V150 A8/A9：配置层收尾扫描与投机采样探针（阴性结果）

日期：2026-10-14 ｜ 实例：FlagOS OnlineLab BI-V150（32GB 显存版）

A7 之后对剩余候选做了一次系统性排除。三项阴性 + 一项算账否决 +
一项探针否决，全部留有数据，作为"配置层油水已榨干"的证据链。

## A8a dec host 壳子：算账否决（未动工）

vendor_flash_attn 混合步里 decode 行走 stock `TritonAttentionImpl.
forward`，timing 日志实测 1.22ms/call ×42 层 ≈ 51ms host/混合步。
但按 benchmark 步数构成算账：

| 场景 | 混合步 | 纯 decode 步（cudagraph 重放） | 混合步占比 |
|---|---|---|---|
| 4k | 64（1M tokens ÷ 16384/chunk） | 4096 | 1.5% |
| 16k | 128 | 2048 | 6% |

且 A7 后混合步 GPU ~250-300ms ≫ host ~60ms，GPU-bound，host
完全被掩盖。砍光壳子收益 4k ~0.4% / 16k ~0.75%，与改动 stock
forward 内部的风险不成比例，否决。

## A8b lm_head GEMM pin：阴性（带宽墙）

A4 pin 表未覆盖 lm_head 形状（M=64, K=2048, N=130560）。48 组
配置全空间扫描（复用 A4 的 `_launch_pinned` 生产路径）：

| 路径 | µs |
|---|---|
| aten vendor | **1094.4（最快）** |
| gems libtuner | 1122.7 |
| 最优扫描配置 (64,128,32,1,4,2) | 1135.0 |

该 GEMM 每步读 535MB bf16 权重，1094µs 已贴带宽墙，任何 Triton
配置无法超越 vendor 库。无需 pin。脚本 `/tmp/a8_lmhead_sweep.py`。

## A8c sampler 内核重调：阴性（已最优）

A5 采样融合内核的配置是 C500（warp 32）直搬，在天数（warp 64）
上补扫 num_splits × num_warps（[64,130560] bf16）：

| 配置 | µs |
|---|---|
| **splits× w4（现配置）** | **323-328（最优，splits 8/16/32 不敏感）** |
| splits8 w8 | 314.4（±3% 噪声内并列） |
| w2 | 590-679（慢 2x） |
| w1 | 2409-2670（慢 8x） |

现配置即甜点，无需改动。脚本 `/tmp/a8_sampler_sweep.py`。
（top_p 支路为评测路径、不计分，未展开。）

## A9 n-gram 投机采样：探针否决

规则口径：可"改进投机采样策略本身"，自训/外部 draft 模型禁止
（陈卓 2026-10-08），故只剩无 draft 的 n-gram 路线。零侵入探针：
用 vLLM 0.24 RandomDataset 同款 prompt（`(offset+arange(n)) %
vocab`，并非真随机）、benchmark 默认采样（temp=1.0, top_p=1.0）
向在线 serve 发 12 条请求（6×4k + 6×16k），离线模拟 vLLM v1
ngram 式提议（最长 n=5→2 匹配，k=5 提议，前缀接受）：

| 命中情况 | 序列数 |
|---|---|
| ~0%（乱码输出无结构） | 9/12 |
| 3.7-16.4% | 2/12 |
| 75.6%（退化复读循环） | 1/12 |

聚合 43.5% 命中是单条复读序列的假象（贡献 3565/3668 次命中），
且文本 re-tokenize 会高估（字节级重复产生假 n-gram）。按面值
估算端到端期望收益 <10%（很可能 <5%），而代价是：verify 步
token 数 ×(k+1) 使 GEMM M=320 掉进 A4 pin 表空白区、与
FULL_DECODE_ONLY cudagraph 兼容性未知、合规上需实质性"改进
策略"开发量。风险收益不匹配，否决。脚本 `/tmp/a9_ngram_probe.py`。

## A9 结论修订（2026-10-16，见 A10/A11）

A9 的**探针做法有偏差、否决理由需要修订**：探针只按"文本离线模拟"估命中率，
低估了复读序列在 token 流中的占比。真实端到端实测（`docs/experiments/a10-a11-spec-decode-negative.md`）：

- n-gram draft 质量是**完美的**：接受率 100%（82685/82685）、接受长度 6.00、A/G 63.1%、
  理论 decode 加速 2.71x —— 官方 benchmark 的随机 prompt 负载恰恰是 n-gram 投机的理想场景。
- 但**净吞吐仍然全面亏损**：CPU proposer 107.68 tok/s、stock `ngram_gpu` 30–100 tok/s、
  均匀批补丁(A11) 2–9 tok/s，对照无投机为 364.11 tok/s。
- 真实原因不是负载，而是**这套 vLLM 0.24 + vllm-plugin-FL(fork) 栈上投机步的执行开销**：
  裁剪同步 + 非均匀批 → FULL 图被绕过走 eager（78–131 ms/step）；
  强行均匀化进入 FULL 图后，图内原生 attention 在 q_len=6 下病态（5–55 s/step）。

最终取舍不变（天数侧不做投机解码），但对外表述应改为
"draft 完美、上限 2.71x，亏在栈的投机步开销"，而非"负载不适配"。

## 结论

shape-aware dispatch（配置层）的已测收益来自 A4/A7；
其余方案未进入交付，原因见对应实验记录。天数侧以 A7 收官：

| 场景 | 基线 | A7 | 累计 |
|---|---|---|---|
| 4k total tok/s | 1902.15 | 3778.65 | **+98.7%** |
| 16k total tok/s | 861.98 | 3567.33 | **+313.7%** |
