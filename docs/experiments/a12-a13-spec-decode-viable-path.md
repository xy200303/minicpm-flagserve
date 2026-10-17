# A12/A13: 让投机采样在天数 BI-V150 上真正跑起来的配置（附 prefill 遗留问题）

日期: 2026-10-16 晚  栈: vLLM 0.24 + vllm-plugin-FL(fork, ModelRunnerFL)

## 1. 结论（重要突破）

**找到了能让投机解码在官方 serve 配置（FULL_DECODE_ONLY cudagraph）下生效的写法，decode 阶段提速近一个数量级。**

| 配置 | decode 每步 | 每步 token | 折合吞吐 | 接受长度 |
|---|---|---|---|---|
| 基线（无投机，A1–A7 栈，图内） | **19.3 ms** | 8 | **364 tok/s**（实测） | 1 |
| stock ngram_gpu（裁剪 → 批不均匀 → 无图走 eager） | 78–131 ms | ~16 | 30–100 tok/s | 1.6–2.2 |
| **A13: padonly + FULL 图 + opt-A12** | **13 ms** | **48** | **≈3700 tok/s** | **6.00** |

13 ms/步的构成（阶段计时实测）：`update_states 5.5 + forward 0.1(图重放) + sample 4.1 + propose 5.0`。

## 2. 生效配置的三个要点

1. **padonly 均匀化（本方向的关键）**
   在 proposer 里把每个请求的 draft 槽补满 k 个（不足处用该请求自己的最后一个 token 占位），
   同时把 `num_valid_draft_tokens` 置为 k。这样：
   - 每请求调度 token 数恒为 1+k → `_is_uniform_decode()` 成立 → FULL_DECODE_ONLY 图可用
     （实证：`cgmode=FULL bdesc=BatchDescriptor(num_tokens=48, num_reqs=8, uniform=True)`）
   - **stock 的裁剪逻辑原样保留**（有效数就是 k，裁剪无事可做）→ 调度器/KV 记账不被破坏
   - 验证仍然逐 token 把关，精度不受影响；占位恰好命中时是合法的接受
2. **不能用"彻底禁用裁剪"（早期 A11 的写法）**
   禁用 `update_scheduler_for_invalid_drafts` 会破坏记账：请求在 `Waiting` 与 `Running` 间反复、
   KV 利用率仅 2%、吞吐掉到 ~1 tok/s。padonly 用"让裁剪无事可做"达到同样的批均匀性，不碰记账。
3. **opt-A12：draft token 的 CPU 侧 gather**
   `_calc_spec_decode_metadata` 里 `input_ids.gpu[logits_indices]` 这类高级索引在混合批
   （索引可达数千）下每步耗 **3.68 秒**（逐行埋点实证：函数总 3684.8 ms，其中 gathers=3684.4 ms，
   其余环节均为微秒级）。改为在 CPU 副本上 numpy gather + 一次小 H2D（≤48 元素）后，该项从热点消失。
   开关：`VLLM_FL_SPEC_GATHER_CPU=0` 可回退原实现。

## 3. 对早前结论的更正

- 之前的"A11 灾难（5–55 s/step）"**几乎全是 prefill 阶段的读数**：decode 阶段一直是好的
  （padonly 无图时实测 86 ms/步）。"spec × FULL cudagraph 被打破"的说法不成立。
- 现在真正的遗留问题是 **混合（prefill+decode）步**，见下节。

## 4. 遗留问题：混合步比基线慢 3–7 倍（阻塞官方口径）

| 阶段 | 每步耗时 | 备注 |
|---|---|---|
| 纯 decode（48 token，图内） | 13 ms | ✅ 远优于基线 |
| 混合 prefill+decode（700–1400 token） | 1.4–3.8 s | ❌ 基线同批约 0.5 s |

已排除：attention 不是主因（vendor 计时：混合步 attention ≈85–105 ms）；
小 H2D/高级索引不是主因（微基准 0.015 ms；A12 已修掉最坏那处）。
阶段计时显示耗时"落"在 `_prepare_input_ids`（1.2–2.4 s）与 `sample_tokens/_sample`（0.6–3.7 s）——
但这些很可能是**吸收了等待**（CPU 跑到 GPU 前面，第一个触碰结果的主机调用被计时）。
因此待查方向：混合步里 spec 路径的 GPU 工作/同步比无投机多出数秒，需继续定位。

官方 4k/16k 用例 prefill 占大头，**这一项不解决，端到端打不过基线**。

## 7. 混合步阻塞点的修复尝试（opt-A5b，进行中）

**定位到具体调用点**：当 `spec_decode_metadata` 非空时，runner 不走 `Sampler`，而是直接调
`RejectionSampler`；在它内部（`vllm/v1/sample/rejection_sampler.py` 的 `forward`）有这样一行：

```python
bonus_sampler_output = self.sampler(
    logits=bonus_logits,
    sampling_metadata=replace(sampling_metadata, max_num_logprobs=-1),
    predict_bonus_token=True,
    logprobs_mode_override="processed_logits" / "raw_logits",
)
```

即 **bonus token 的采样走的是 Sampler.forward**，而我们的 A5 融合采样正是以
`predict_bonus_token` 为由拒绝了它 → 回落到本栈上极慢的 stock top-p 采样。
二级计时实证（wall/device 成对，单位 ms/call）：

| 调用 | wall/device | 说明 |
|---|---|---|
| `Sampler.sample` | **974/1016** | 就是这笔开销 |
| `rejection_sample` | 17/6 | rejection 本身很便宜 |
| `Sampler.apply_logits_processors` | 0/0 | 无 |

**A5b 改动**：在 `predict_bonus_token` 且无 penalties/bad words/processors/logprobs 语义需求时
允许融合核处理（并返回 logits 作为 logprobs 张量，保证边缘情况不崩）；同步修正拒绝原因的报告顺序。

### 7.1 今日逐个打通的拦截条件（2026-10-17）

| 拦截条件 | 真相 | 处理 |
|---|---|---|
| `predict_bonus_token` | 投机的 bonus token 采样走 `Sampler.forward` | 允许（分布不变） |
| `logprobs_mode_override` | RejectionSampler 内部强制 `max_num_logprobs=-1` + override | 允许（无用户 logprobs 时该参数只影响返回值），并回传 logits 作 logprobs 张量 |
| `top_k` | **该 fork 的 OpenAI 默认把 top_k 设成 vocab-1（130559 = 130560-1），只裁掉一个 padding token**，语义上等同不裁剪 | 判定放宽为"仅当 top_k < vocab-1 才算启用"；并通过 runner 钩子发布真实 vocab（`model_config.get_vocab_size()`） |
| `generators` | 投机模式下 runner 给**每个** sampler 调用都挂 per-request generator | 放开（分布不变；放弃"同 seed 逐 token 复现"，可用 `VLLM_FL_DISABLE_SAMPLER_FUSION=1` 回退） |

**结果**：拒绝日志归零，融合路径覆盖全部采样调用。实测某些混合步的 sampler 已降到 **47 ms wall / 11 ms device**
（此前 762/756 ms），说明修复有效。

### 7.2 仍未解决：混合步还有 1.4–2.5 s

同一轮（F9）里另外两类步仍然慢：`sample_tokens=3695/3636`、`_prepare_input_ids=2449/2451`。
同时 GPU 利用率采样为 **92–99%**，而 vendor attention 计时只有 **~5–26 ms/步** →
说明瓶颈是**真实的 GPU 工作**（eager 混合批前向 + 部分仍走慢路径的采样），不是 attention、也不是单纯的 CPU 等待。
注意：`_prepare_input_ids` 自身只做 KB 级拷贝，其 wall≈device 的 1.2–2.4 s 很可能是
"默认流事件"在等其它流上的工作（vLLM 的 `synchronize_input_prep()` 表明输入准备在独立流上），
因此这块的归因需要用 nsys 级别的工具或逐 kernel 计数才能钉死。

**下一步**：(1) 用逐 kernel/流级工具确认那 1.4–2.5 s 到底花在哪个 kernel；
(2) 若确认是 eager 混合批前向的 GEMM/MLP，考虑把 spec 的混合步也做成可入图（或按 q_len 拆分两段执行）；
(3) 采样器剩余慢步（3.6 s）需确认是否还有第二条调用路径未走融合。


## 5. 复现命令

```bash
VLLM_PLUGINS=fl VLLM_FL_UNIFORM_SPEC_MODE=padonly \
vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm \
  --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --speculative-config '{"method": "ngram_gpu", "num_speculative_tokens": 5, "prompt_lookup_max": 5, "prompt_lookup_min": 2}'
```
（`VLLM_FL_UNIFORM_SPEC_MODE=notrim` 为早期被否决的写法，留作对照。）

日志与证据：`/workspace/opt_experiments/`（含 E7/E12/E13 日志、specmeta 逐行埋点输出）。

## 6. 混合步慢的真凶：sampler（E14，wall/device 双计时实证）

用 CUDA event 同时记录"墙钟/设备"时间（`wall/device` 成对）：

```
step_wall=2725 ms | _prepare_input_ids=1261/1264  _calc_spec_decode_metadata=1230/1228
                    sample_tokens=814/810  _sample=762/756  _model_forward=56/167
step_wall=5647 ms | sample_tokens=3699/3641  _sample=3673/3614
```

- `_sample` 的 wall ≈ device（如 3673/3614），说明这是**真实的 GPU 队列时间**，不是 CPU 等待假象。
- 成本随批内 token 数放大：decode（48 token）时 sampler 只要 4 ms，混合步（700–1400 token）要
  0.6–3.7 s → 约 2.6 ms/token，说明 spec 模式下 sampler 对**批内全部 token**做全词表采样。
- 而无投机路径上这个成本不存在：我们的 A5 采样器融合（FlagGems 融合核）在无投机时走 fast path，
  在投机时被 `predict_bonus_token` 主动拒绝（日志：`fl opt7: fast path rejected: predict_bonus_token`），
  于是回落到慢的 stock 采样路径。

**下一步（明确、可做、合规）**：把 A5 的融合采样（bf16 Gumbel-Max + histogram-zoom top-p，无全词表排序）
扩展到投机路径（bonus token + 拒绝采样），落在 FlagGems。预计混合步从 1.4–5.6 s 回到 ~0.2–0.5 s
（与无投机同量级），届时端到端才有望在官方 prefill 重用例上取胜：
decode 13 ms/48 token（基线 19.3 ms/8 token）×  prefill 回到基线水平。

## 8. 2026-10-17 第二轮：k 扫描、varlen 对照与 profiler 尝试

### 8.1 关掉 varlen prefill 无效（G1）

`VLLM_FL_DISABLE_VARLEN_PREFILL=1` 下混合步仍 1.4–1.5 s（`_prepare_input_ids=1215–1221`、
`forward=183–191 ms`），与开启时一致 → A6 的 varlen 路径不是原因。

### 8.2 k=1 扫描（G2）：混合步随 (1+k) 近似线性

| 配置 | 混合步 wall | propose | forward | sample |
|---|---|---|---|---|
| k=5 | 1.4–2.5 s | 27–51 ms | 167–191 ms | 11–3610 ms（不稳定） |
| **k=1** | **144–301 ms** | 150–316 ms | 167–171 ms | 608–883 ms（窗口平均，含个别重步） |

结论：混合步成本随每请求投机 token 数（1+k）近似线性放大 → 指向"混合批里投机行的前向/采样开销"，
不是某个固定成本。k=1 时 execute_model 之外的 `gap` 仍 0.8–1.2 s，说明调度/输出侧另有秒级开销，两者叠加。

### 8.3 设备级 profiler 尝试（均未果）

- `flagtree-profiler -b iluvatar -d trace -n specmix vllm serve ...` 能启动、能跑负载，
  但进程退出后未在任何候选路径落 trace（/workspace、/root、/tmp 均查过）。
- torch profiler（`--profiler-config`）会因厂商 `External init callback must run in same thread`
  错误导致 EngineCore 崩溃。
- 下一步建议：用 `flagtree-profiler` 的 `-c python -d tree` 或 ixKN（`--ixkn`）后端；
  或在 runner 侧记录"每步 batch 形状（token 数/请求数/是否混合）+ 耗时"，用形状-耗时关联定位。

## 9. 当前可交付与未完成

**已验证可交付**：`padonly + FULL_DECODE_ONLY` 的纯 decode 步 = **13 ms / 48 token ≈ 3700 tok/s**
（基线 19.3 ms / 8 token ≈ 364 tok/s），接受长度 6.00，采样器已全部走融合路径。

**未完成（阻塞官方口径）**：混合 prefill+decode 步仍 0.14–2.5 s（随 1+k 放大），
官方 4k/16k 用例 prefill 占大头，端到端尚不能确认优于 A7 基线（3778.65 / 3567.33）。
在混合步解决前，**提交仍应使用 A1–A7 栈（无投机）**，投机作为"已验证 decode 收益 + 待收尾"的探索记录。

## 10. 官方口径实测（4k c64，2026-10-17）：投机模式不可用

按官方 test-case `[4096,1024,64,256]` 实测（`padonly + FULL_DECODE_ONLY + k=5 + A5b`）：

```
Avg prompt throughput: 409.5 tokens/s, Avg generation throughput: 0.7 tokens/s, Running: 5, Waiting: ...
Avg prompt throughput: 409.6 tokens/s, Avg generation throughput: 1.2 tokens/s, Running: 8, Waiting: ...
```

- 并发起不来：c64 上限下引擎只跑到 ~8 个请求（Waiting 持续积压），KV 池 514k token 不是瓶颈
- prefill 只有 **409 tok/s**（无投机基线约 2800 tok/s，差 7 倍）
- 生成只有 **1 tok/s**（纯 decode 步本身是 13 ms/48 token，说明几乎所有步都被混批拖住）
- 按此速度该用例需数小时（基线 347 s）→ **官方口径下投机模式不可用**

**交付决定：提交使用 A1–A7 栈（无投机）。** 投机部分作为已验证的探索记录写入技术报告：
decode 侧 9x 收益已验证（13 ms/48 token ≈ 3700 tok/s vs 基线 19.3 ms/8 token ≈ 364 tok/s），
混合批（prefill+decode）开销被放大 7–20 倍且无法定位（缺设备级 profiler），官方口径不可用。

## 11. 交付栈的官方 c64 剖面与 A13（2026-10-17 下午）

### 11.1 剖面（无投机，官方 4k c64，带 wall-only 埋点）

```
[T] (25 steps): step=4.2 ms gap=52.3 tok/step=64 reqs=64 | _prepare_inputs=2.8 sample=0.4 metadata=0.2
[T] (25 steps): step=200.2 ms gap=2.4 tok/step=2048 reqs=44.1 | _model_forward=120.0 metadata=74.0
```

- **decode 迭代 ≈ 4.2 ms（execute_model）+ ~52 ms（等 GPU 完成 + 调度/输出）** → 64 请求下 ~1100–1276 tok/s
- **prefill 步 ≈ 200 ms**：forward 120 ms + `_build_attention_metadata` 74–82 ms + 其余
- 即 prefill 阶段约 40% 时间在 attention metadata 构建

### 11.2 元凶：补丁每步访问 GPU→CPU 拷贝属性

`_build_split` 本体实测约 1 ms（idx 0.09 + rows/seq H2D 0.11 + arange 0.13 + block_table index_select 0.60），
但外层计时显示 split 段为 **85–98 ms**。差异来自
`common_attn_metadata.query_start_loc_cpu` / `seq_lens_cpu_upper_bound`：
在这套栈里 `query_start_loc_cpu` 是 `decode_query_start_loc.to("cpu", non_blocking=True)` 的产物，
**每步 `.tolist()` 会触发一次 D2H 同步，等掉整个 GPU 队列**（85–98 ms）。

### 11.3 A13 尝试（未验证，已回退）

实现：runner 在 `_prepare_inputs` 里把 CPU 侧 `cu_l`/`seq_l` 发布给 vendor 补丁，构建器不再碰 D2H。
实测该配置的官方 4k 用例反而更慢（11 分钟未完成）；但同一天用**未改动代码**复测基线也只有
**2346.64 total tok/s / 558.55 s**（历史 3778.65 / 347 s）→ **今天机器本身慢约 38–60%**，
A13 的读数被机器漂移污染，结论不成立。已回退 A13，保留为下次"交错 A/B"的候选。

### 11.4 结论与纪律

- 交付：A1–A7（无投机）。今日复测 2346.64 total tok/s 是**机器变慢**，不是代码回退
- 下一个可验证优化点：去掉 metadata 构建里的每步 D2H（A13 方向）
- **纪律：同一天的绝对数字不可跨天比较；任何优化必须当天交错 A/B（开/关交替各 2–3 次）**
