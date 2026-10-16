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
