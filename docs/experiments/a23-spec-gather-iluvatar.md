# A23: 投机混合步 4 s/步的真凶是 `logits[idx]` 在 Iluvatar 上的退化索引

日期: 2026-10-22  卡: 天数 BI-V150 32GB（开发机）  栈: vLLM 0.24 + vllm-plugin-FL(flagos-2026-s2) b0cda5e + `spec_experiment_bundle.patch`

## 1. 背景

A10–A13 把投机解码的**纯 decode 步**做到了 13 ms / 48 token（≈3700 tok/s，基线 364），
但混合 prefill+decode 步始终慢 3–7 倍，官方 `[4096,1024,64,256]` 口径下 prefill 只有
409 tok/s、生成 0.7–1.2 tok/s，因此 A10–A13 的结论是"投机不可用"。

当时归因到"混合批 eager 前向 + 采样器第二条慢路径"，并要求设备级 profiler——
但 `torch.profiler` 会让 EngineCore 崩（厂商 `External init callback must run in same thread`），
`flagtree-profiler` 不落 trace。这一步一直没有做成。

## 2. 方法

把 `RejectionSampler.forward`（`vllm/v1/sample/rejection_sampler.py`）整段复刻，
在 8 个位置插 CUDA event，每步输出各段 device 时间：

```
bonus_index / bonus_sample / target_index / to_fp32 / clone
logits_procs / constraints / rejection
```

诊断代码：`/tmp/fl_sampler_timing.py`（由 `iluvatar.py` 在导入时加载，`VLLM_FL_SAMPLER_TIMING=1` 生效）。
负载：8 并发 × 4096 random token prompt × 64 output。

## 3. 结果

### 3.1 慢步全落在一行上

```
n=9  total=4127.5ms | bonus_index=4112.8  bonus_sample=0.2  target_index=4.9  to_fp32=0.0  clone=0.1  logits_procs=0.0  constraints=1.8  rejection=1.4
n=10 total=4120.5ms | bonus_index=3997.0  ...
n=14 total=4142.9ms | bonus_index=4112.9  ...
n=15 total=6438.7ms | bonus_index=6364.7  ...
```

`bonus_index` 就是 forward 的第一行 `bonus_logits = logits[bonus_logits_indices]`。
**它一个操作吃掉 99.7% 的时间**；紧随其后的 `target_index` 是同样的花式索引、形状更大，只要 4 ms。
之前怀疑的 `.to(torch.float32)`（混合步 783 MB）实测 **0.0 ms**。

### 3.2 微基准排除"算力"解释

```
(38, 130560) bf16，38 个索引：
  x[idx] int64              0.07 ms
  index_select              0.06 ms
  noncontig view x[idx]     0.06 ms
  fp32 x[idx]               0.13 ms
  1500 行 tensor x[idx]     0.06 ms
```

差 6.8 万倍 → 不是算力问题。

### 3.3 同一 forward 内的 A/B（决定性）

在同一张量、同一索引、同一条流上紧挨着执行两种写法：

| 步 | `logits[idx]` 花式索引 | `torch.index_select(logits, 0, idx)` |
|---|---|---|
| n=9 | 4113.1 ms | 0.2 ms |
| n=10 | 3986.6 ms | 0.3 ms |
| n=11 | 4031.1 ms | 0.3 ms |
| n=12 | 3997.2 ms | 0.3 ms |

**`aten::index` 在 Iluvatar 后端对"由 cumsum 得到的非连续索引张量"退化到 ~4 s/次；
同一形状改为 `index_select` 后回到 0.3 ms。**

这与 A12 在 `_calc_spec_decode_metadata` 里发现的 `input_ids.gpu[logits_indices]` = 3.68 s/步
是同一类退化（A12 当时用 CPU numpy gather 绕开，没有意识到根因是索引算子本身）。

### 3.4 替换后的效果

把 `logits[bonus_logits_indices]` 与 `logits[target_logits_indices]` 都换成 `index_select`：

```
n=10 total=121.7ms | bonus_sel=0.3 bonus_sample=0.2 target_index=0.4 to_fp32=0.0 clone=0.1 constraints=5.1 rejection=0.2
```

**RS 整体 4100 ms → 122 ms（34x）。**

首次调用的固定成本仍然存在且与本次改动无关：
`n=1 rejection=4105.9 ms`（`rejection_sample` 的 triton kernel 首次编译）、
`n=3 constraints=2121.1 ms`（`apply_sampling_constraints` 首次）。

## 4. 仍然存在的问题（待测）

端到端 probe（8 并发 × 4096 prompt × 64 output）仍是 **67.3 s**，即 ≈7.6 tok/s；
serve 侧 throughput：

```
17:39:08  prompt 819.2 tok/s, generation 0.7 tok/s, Running 4
17:39:28  prompt 1228.6 tok/s, generation 2.9 tok/s, Running 8
17:39:48  prompt 409.6 tok/s, generation 40.6 tok/s, Running 7
```

对照无投机基线：prompt ≈2800 tok/s、generation ≈360 tok/s。
即 prefill 仍慢 2–3 倍、decode 仍慢 ~20 倍，**这份修复只消掉了一项**。

下一步：启用 `fl_timing3`（`VLLM_FL_TIMING=1`）拿整步分段，定位剩余的时间落在
`propose_draft_token_ids` / `_update_states` / `_prepare_input_ids` 中的哪一段。

## 5. A24：同一退化在 `_prepare_input_ids` 里的第二处

`fl_timing3` 显示混合步的主项是 `_prepare_input_ids` = **2458/2459 ms（wall/device）**。
该函数在 spec + async-scheduling 路径上有三处 GPU 高级索引：

```python
src=self.input_batch.prev_sampled_token_ids[prev_common_req_indices_tensor, 0]   # 高级索引
src=draft_token_ids.flatten()[prev_draft_token_indices_tensor]                    # 高级索引
self.input_ids.gpu.scatter_(dim=0, index=..., src=...)                            # scatter_
```

前两处换成等价写法：

```python
src=self.input_batch.prev_sampled_token_ids.index_select(0, prev_common_req_indices_tensor)[:, 0]
src=torch.index_select(draft_token_ids.flatten(), 0, prev_draft_token_indices_tensor)
```

`scatter_` 未改。结果：

| 指标 | 改前 | 改后 |
|---|---|---|
| `_prepare_input_ids`（混合步） | 2458 ms | **1 ms** |
| `step_wall`（纯 decode，8 并发） | 1094 ms | **8 ms** |
| `_sample`（纯 decode） | 474 ms | **16 ms** |
| probe 端到端（8×4096 prompt × 64 out） | 67.3 s | **35.5 s** |

## 6. 官方 c64 口径仍然不可用（决定性）

修复后在官方 `[4096,1024,64,256]` 下：

- **decode 阶段：454 tok/s（基线 364，+25%）**、step 周期 138 ms（`step_wall` 18 ms + gap 120 ms）
- **prefill 阶段：2867–3686 tok/s（已回到基线水平）**
- **但完成数极低：5 分钟只完成 3/256 个请求**，进度条外推 18 小时

`fl_timing3` 在 c64 下的混合步：

```
step_wall=165 ms  _model_forward=119/175
step_wall=7594 ms _model_forward=954/989
step_wall=605 ms  _model_forward=586/623
```

即：**混合步的 eager 前向（100–950 ms，单步可达 7.6 s）吃掉了调度预算**。
padonly 让每个 decode 请求恒定带 1+k=6 个 token，混合批里 decode 部分膨胀 6 倍，
prefill 请求拿不到足够的 token 预算 → 请求在 Waiting/Running 之间饥饿。

这已经与算子无关，是"投机 token 记账 × 调度预算"的交互；要解决必须动调度，
而组委会对"在插件内部改 `max_num_batched_tokens` 一类默认值"的判罚是负面的。
**因此投机路线在官方 c64 口径下仍然关闭，A10–A13 的结论不变。**

## 7. 本次修复的实际价值

- 它**解释并统一了** A12 当时观测到的 `input_ids.gpu[logits_indices]` = 3.68 s/步：
  根因不是"GPU 高级索引 O(max index)"，而是**该后端的 `aten::index` 在索引张量非连续时退化到秒级**，
  同形状 `index_select` 只要 0.3 ms。A12 的 CPU-gather 只是绕开，没有修根因。
- 修复只在 spec 路径上生效（无投机基线不经过这两处），而官方评测不传 `--speculative-config`，
  所以**对提交成绩没有直接影响**；它的定位是"发现的框架/算子缺陷"，写入技术报告的缺陷分析章节。

## 8. 复现

```bash
# 诊断（不改仓库）
VLLM_PLUGINS=fl VLLM_FL_UNIFORM_SPEC_MODE=padonly VLLM_FL_SAMPLER_TIMING=1 \
  vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --speculative-config '{"method": "ngram_gpu", "num_speculative_tokens": 5, "prompt_lookup_max": 5, "prompt_lookup_min": 2}'
```

产物：`/workspace/opt_experiments/rs_serve{2..5}.log`、`spec_probe{1..4}.log`、
`official4k_spec1.log`、`model_runner.py.before_a24`、
`monitor/monitor.log`（10 s 采样的 GPU/内存/进程/HANG 判据）。
