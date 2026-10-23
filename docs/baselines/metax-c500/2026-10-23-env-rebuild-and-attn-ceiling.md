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

## 7. 步时间账本（eager 逐项 + 微基准交叉验证）

用 `c500_timing.py`（CUDA event，挂 `FlashAttentionImpl.forward` / `compute_logits` /
`_model_forward` / `_sample`）在 `--enforce-eager` 下拿到的逐项（64 并发，短负载）：

| 场景 | attention | `_model_forward` | 其中非 attention | logits(lm_head) | sample |
|---|---|---|---|---|---|
| 4k | 0.39 ms/层 × 42 = 16.4 ms | 72.5 ms | **56 ms** | 0.49 ms | 0.37 ms |
| 16k | 1.70 ms/层 × 42 = 71.4 ms | 126.0 ms | **54.6 ms** | 0.48 ms | 0.27 ms |

关键：**非 attention 部分与 KV 长度无关（55 ms 左右）**，所以 16k 的 TPOT 增长
完全来自 attention。lm_head 与采样各不到 0.5 ms，可以排除。

## 8. 微基准交叉验证：GEMM 与 attention 各自的天花板

**decode GEMM（M=64, bf16, nn 布局）**：

| 算子 | 耗时 | 带宽 |
|---|---|---|
| qkv (K2048/N2560) | 0.021 ms | 494 GB/s |
| o (K2048/N2048) | 0.019 ms | 437 GB/s |
| gate_up (K2048/N12288) | 0.046 ms | 1104 GB/s |
| down (K6144/N2048) | 0.043 ms | 592 GB/s |
| lm_head (K2048/N130560) | 0.378 ms | 1415 GB/s |
| **单层合计** | **0.50 ms** | **1260 GB/s 有效** |

42 层 ≈ **5.4 ms** + lm_head 0.4 ms。**GEMM 不是瓶颈**。

**attention 的 num_splits 敏感性（16k, 64 并发, 2.4 GB KV 池）**：

| num_splits | ms/层 | GB/s | 42 层/步 |
|---|---|---|---|
| None（FA 启发式） | 0.796 | 1349 | 33.4 ms |
| 1 | 2.372 | 453 | 99.6 ms |
| 2 | 1.438 | 747 | 60.4 ms |
| 4 | 0.934 | 1150 | 39.2 ms |
| 8 | 0.928 | 1157 | 39.0 ms |
| 16 | 0.803 | 1338 | 33.7 ms |

**num_splits 从 1 到 16 有 3 倍的差距。** 但当前栈并没有强制 1：
`_batch_invariant_MODE` 默认 False，`flash_attn_max_num_splits_for_cuda_graph` 默认 32，
所以 `max_num_splits` 走 `0`（FA 启发式）。这一点已排除。

## 9. 尚未闭合的一环

图模式（TPOT 77.94 ms）按微基准应为 attention 33.4 + GEMM 5.4 + lm_head 0.4 ≈ **39 ms**，
但实测 78 ms，**约 39 ms 无法归因**。

在 cudagraph 下 Python hook 拿不到这个数：`_model_forward` 与 `FlashAttentionImpl.forward`
在捕获时各执行一次，replay 时整段是图节点，Python 不参与。eager 下的 1.70 ms/层 混合了
kernel launch 间隙，不能直接当作图内的 device 时间。

**下一步（明确）**：在捕获期把 attention 的 CUDA event 对**保留下来**（不清空 pending），
让 `execute_model` 每步读这些 event 的 `elapsed_time`。图节点持有 event 引用，replay 时时间戳会更新，
这是唯一能在图内拿到真实 layer 时间而不引入 profiler 的办法。

## 10. num_splits 实测结论：真实服务无收益（2026-10-24）

在 `flash_attn_with_kvcache` 的 decode 调用上显式指定 `num_splits` 并跑官方用例：

| 配置 | 前置同长度负载 | 16k 吞吐 | 备注 |
|---|---|---|---|
| 默认（不传 num_splits） | 无 | 11104.26 | 基线 |
| `num_splits=16` | **有**（先跑了 `[16384,128,64,64]`） | 14779.03 | **被 prefix cache 污染，作废** |
| `num_splits=16` | 无 | 10894.59 | 冷缓存 |
| `num_splits=32` | 无 | 10953.44 | 冷缓存 |

**16k 上 num_splits 无收益（±2% 噪声内）**。微基准里 `num_splits=1` 比 16 慢 3 倍，
但真实路径用的是 FA 启发式，显式传值并不能带来端到端收益。

4k 上 `num_splits=16` 测到 12739.83（基线 12403.21，+2.7%），
**未达 3% 准入，且 16k 无收益，已整体回滚**（`flash_attn.py` 恢复原文件）。

### 方法论：prefix cache 会静默污染 A/B

benchmark 用 `seed=0`，**同一个 serve 上跑两次相同 `input_len` 的负载，prompt 完全一致**。
第二次会大量命中 prefix cache（本次日志中出现过 35.9% 的命中率），
prefill 近乎免费，总吞吐虚高 30% 以上。

**任何"先跑短负载预热、再跑正式用例"的流程都是错的。**
正确做法：
1. 每个数据点都在**新启动的 serve** 上跑，且只跑**一种** `input_len`；
2. 若必须连跑，先跑长 `input_len` 再跑短的（不同长度不会互相命中），
   或者用 `--no-enable-prefix-caching`（注意：那是 serve 参数，提交时不能改，
   只能用于本地诊断）；
3. 交错 A/B 时，A 与 B 都必须从冷缓存开始。

## 11. 图内 CUDA event 不可读，且带计时的短负载不能用于性能对比（2026-10-24）

尝试用"捕获期保留 attention 的 event 对，replay 后由 `execute_model` 读取"的办法测图内 attention：

- 现象：`[AG] read failed: CUDA error: invalid resource handle`，每次 replay 都报。
- 结论：**这套后端不支持从 Python 侧读取图内 event 的时间戳**。图内分解到此为止，
  需要硬件 profiler 才能继续。

**同时发现一个更严重的测量陷阱**：同一短负载 `[16384,128,64,64]`，
带 `C500_TIMING=1` 时吞吐 9339.93，带 `C500_AG=1` 时 15269.73，
**同负载差 60%**。原因是每次 attention 都创建 CUDA event 并入 list，
42 层 × 每步的 Python/驱动开销直接改变了被测对象。

**因此本文第 7 节（eager 分解）的绝对数值只能用于相对比较，不能用于预测图模式下的 TPOT。**
所有性能结论一律以**不带任何计时的正式用例、冷缓存**为准：

| 场景 | 可靠基线 |
|---|---|
| 4k `[4096,1024,64,256]` | **12403.21 tok/s**（2.44x） |
| 16k `[16384,1024,64,128]` | **11104.26 tok/s**（1.58x） |

## 12. 本轮（2026-10-23/24）的可交付结论

- 环境在新实例上完整重建，opt1–opt11 栈复现（bundle 传输 + editable 安装）。
- num_splits：微基准 3 倍差异，真实服务冷缓存无收益，已回滚。
- 图内 event：不可用；短负载计时对比：不可用。
- **没有任何 ≥3% 的新优化产出**，C500 交付成绩维持 2.44x / 1.58x。
