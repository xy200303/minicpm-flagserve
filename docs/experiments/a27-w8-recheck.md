# A27: W8 gate_up 重测——隔离里 1.82x，端到端 −19%

日期: 2026-10-22  卡: 天数 BI-V150 32GB  栈: vllm-plugin-FL b0cda5e（A20）

## 1. 为什么重测

A22 用 0.77x 否决了 W8。复查发现两处对比不公平：

1. **量化开销被高估**：A22 的"激活量化"是
   `amax = x.abs().max().float().reshape(1)` + 一个量化核 = **4 个 kernel**。
2. **基线选错**：A22 拿"裸 GEMM 120.9 µs"比"W8 含反量化核 157.5 µs"，
   而真实 bf16 路径是 GEMM + SwiGLU = 137.5 µs。

## 2. 隔离测量（单层 gate_up，M=64, K=2048, N=12288，100 次平均）

| 路径 | 耗时 | 备注 |
|---|---|---|
| bf16 GEMM + SwiGLU | 137.4 µs | 真实路径 |
| bf16 裸 GEMM | 118.7 µs | A22 的基线 |
| `torch._int_mm` 单独 | 55.6 µs | 读 25 MB int8，450 GB/s |
| 单核 per-token 量化 | 21.8 µs | 一个 Triton 核，无 sync |
| **W8（单核量化 + int_mm + 融合 dequant-silu）** | **75.4 µs** | **1.82x**，maxdiff 0.031 |
| W8（A22 的四核量化） | 203.5 µs | maxdiff 0.035 |

**A22 的 0.77x 是实现的产物，不是量化的固有权衡。** 这一段结论成立。

## 3. 接进引擎：两次都是负收益

实现（`w8_mlp.py`，patch `LlamaMLP.forward`，只对 M ≤ 128 的 decode 形状生效）：

- per-token 激活量化单核；
- 权重首次使用时 per-channel 量化为 int8，与 bf16 副本共存（prefill 仍走 bf16）；
- dequant 融进 SwiGLU epilogue。

**第一次**（用 `data_ptr` 做缓存 key）：官方 4k **2512.10 tok/s**（基线 2571.12，−2.3%），
TPOT 98.15。事后判断缓存 key 可能失效（vLLM 会重绑权重），W8 大部分没生效，
那 0.98 GB int8 副本只是白占显存。

**第二次**（改成 `id(module)` + 权重对象身份校验，惰性量化、捕获期跳过）：

| 指标 | 基线 b0cda5e | W8 | 变化 |
|---|---|---|---|
| Total token throughput | 2571.12 tok/s | **2080.18 tok/s** | **−19.1%** |
| Mean TTFT | 5529.69 ms | **24185.14 ms** | **4.4x 恶化** |
| Mean TPOT | 96.58 ms | **116.57 ms** | +20.7% |
| KV cache | 515,520 tokens | 523,248 tokens | 未缩水 |

decode 步的 `pace` 读数出现 10–23 ms 与 4233 ms 的尖峰（基线 ~4.2 ms + gap）。
cudagraph 捕获本身成功（35/35），日志里没有 capture 失败。

## 4. 结论：隔离收益不能外推

**隔离里 1.82x，端到端 −19%，两者背离。** 最可能的解释是微基准的测量方式本身失真：

- 微基准把**同一个** `gate_up` 连续调 100 次，25 MB int8 权重和前一层激活很容易留在 L2；
- 真实引擎是 42 层 × 1 GB 权重轮转，每层都是冷数据，且同时驻留 bf16（4 GB）+ int8（1 GB）两份；
- TTFT 恶化 4.4 倍说明连 prefill（走 bf16）都被拖累，更像是显存/TLB 压力而非 decode kernel 本身。

**没有继续追**：要定论需要 `_int_mm` 在图内外的逐 kernel 计数，而 A23 已经证明这套栈拿不到
设备级 profiler；且即使定论，收益上限也只是 2.6 ms/步。

## 5. 处置

- **代码已全部回退**，交付栈恢复 `b0cda5e`，site-packages 同步完成，GPU 回落到 68 MiB。
- A22 的**否决结论维持**，但理由改为：
  "隔离测量在修正量化实现后确实有 1.82x（A22 的 0.77x 是实现缺陷 + 基线选错），
  但接进真实引擎后端到端 −19%（4k 2080 vs 2571），且 TTFT 恶化 4.4x；
  micro-benchmark 的‘热权重’条件与 42 层轮转的‘冷权重’工况不符，不足以外推。"
- 这条要写进报告的方法论章节：**本卡的 micro-benchmark 只能用来筛掉明显更差的设计，
  不能用来预测端到端收益；任何候选都必须走当日交错 A/B。**

## 6. 复现

```bash
# 隔离测量
python3 /workspace/opt_experiments/w8_recheck.py
# 集成（已回退，补丁需重放）
#   vllm_fl/dispatch/backends/vendor/iluvatar/impl/w8_mlp.py
#   iluvatar.py 中 sampler_tail_fusion 之后注册 w8_mlp.apply()
VLLM_PLUGINS=fl VLLM_FL_W8_DEBUG=1 vllm serve ... (官方参数)
```

产物：`/workspace/opt_experiments/w8_recheck.py`、`a27_4k_w8on.log`、`a27b_4k.log`、
`a27b_serve.log`。
