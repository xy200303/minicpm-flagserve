# C500 2026-10-23：环境重建、基线复现与 decode-attention 天花板

机器：全新 C500 实例（`140.207.205.81:32222`，MXC500 64 GB，**128 GB CPU 内存 / 128 核**）
栈：`vllm-plugin-FL@dev`（f4971e7，opt1–opt11）+ `FlagGems@dev`（5eac6b39）

## 1. 环境重建

新实例只有官方原版（vllm-plugin-FL `13eb9be`、FlagGems `a7620cc1`），GitHub 直连不可达，
用**增量 git bundle** 从本地 mirror 传入：

```
git bundle create vfl.bundle 13eb9be6..vllm-plugin-FL/dev   # 21 KB
git bundle create fg.bundle  a7620cc19..FlagGems/dev        # 27 KB
git fetch /tmp/vfl.bundle refs/heads/vllm-plugin-FL/dev:refs/heads/dev
```

两个仓库均为 editable 安装。环境变量固定在 `/workspace/env.sh`：

```bash
export PATH=/opt/conda/bin:$PATH
export MACA_PATH=/opt/maca
export MACA_HOME=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/lib64
export VLLM_PLUGINS=fl
```

## 2. MM autotune 预热：6 分钟，不是死锁

首次带 `VLLM_FL_MM_WARMUP=1`（默认）启动时日志停在
`[fl-warmup] mm autotune warmup start` 十多分钟，进程状态 `R`、CPU 满载，
一度被误判为记录里的 `torch.cuda.Stream()` 死锁。

实测第二次（triton cache 已存在）**367.4 秒完成**，随后正常启动。结论：

- **不是死锁，是慢**（5 个权重形状 × 120 次 mm 调用，每次含 autotune 搜索）；
- 关掉预热（`VLLM_FL_MM_WARMUP=0`）启动快，但 4k 掉 **5.9%**（11707.77 vs 12403.21），
  TTFT 从 1766 ms 涨到 3016 ms —— 预热是有价值的，评测时多等 6 分钟即可。

## 3. 基线复现（当日、cold cache）

| 场景 | 官网基线 | 历史 opt11 | **今日复现** | 相对官网 |
|---|---|---|---|---|
| 4k `[4096,1024,64,256]` | 5089.645 | 12529 | **12403.21** | **2.44x** |
| 16k `[16384,1024,64,128]` | 7029.675 | 11088 | **11104.26** | **1.58x** |

- 4k：TTFT 1766.08 ms，Mean TPOT 23.87 ms（Median 24.91）
- 16k：TTFT 18051.91 ms，Mean TPOT 77.94 ms（Median 87.67）

复现偏差 −1.0% / +0.1%，**opt11 栈在新实例上完全成立**。

## 4. decode attention 已到 HBM 天花板

微基准（B=64, q_len=1, 16 q heads / 2 kv heads, d=128, bf16）：

| 调用路径 | kv=4096 | kv=16384 |
|---|---|---|
| `flash_attn_varlen_func`（prefill 用） | 1.454 ms/层，185 GB/s | 5.704 ms/层，188 GB/s |
| **`flash_attn_with_kvcache`（真实 decode 路径）** | **0.209 ms/层，1284 GB/s** | **0.732 ms/层，1466 GB/s** |

vLLM 的 decode 走 `flash_attn_with_kvcache`（paged），不是 varlen。**1466 GB/s 已超过
MXC500 标称的 1.2 TB/s，贴近 HBM 峰值**。

推论：16k 的 42 层 attention 每步约 **30.8 ms**，而 Mean TPOT 是 77.94 ms——
**attention 只占 40%，且这 40% 已经没有优化空间**（和天数 A26 的结论一致：
decode attention 是带宽地板）。剩下的 ~47 ms 必须去别处找，而不是内核。

## 5. 下一步（按证据排序）

1. **先定位那 47 ms**：用 `VLLM_FL_SLOW_STEP_MS` 或逐阶段埋点看 16k decode 步构成；
   attention 已排除，嫌疑集中在 lm_head GEMM、采样、以及 64 并发下的调度/同步。
2. **16k 的 TTFT 18 s** 是另一个大头（prefill 占端到端约一半），opt8 的 mcBLAS 已是
   vendor 峰值，要看是否还有 host 侧空洞。
3. 不做的事：decode attention 自研内核（带宽已满，天数 A20 的经验在这里不适用）。

## 6. 产物

`/workspace/c500_attn_probe.py`、`c500_attn_probe2.py`、`c500_4k_warm2.log`、
`c500_16k_warm2.log`、`opt11_serve4.log`（含 warmup 367.4s 与完整启动）。
