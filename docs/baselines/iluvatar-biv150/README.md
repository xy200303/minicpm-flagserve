# 天数 BI-V150（Iluvatar，OnlineLab 实例 32GB 版）

赛道二第二张卡的工作区。官方开发与评测均使用 **32GB** 显存版。
本实例 KV 池约 52 万 token，16k×64 并发会排队；性能数据均按 32GB 口径分析。

## 基线（stock flagos-2026-s2，官方 serve 命令 + FULL_DECODE_ONLY）

| 场景 | total tok/s | TTFT |
|---|---|---|
| 4k [4096,1024,64,256] | 1902.15 | 12.3s |
| 16k [16384,1024,64,128] | 861.98 | 636.5s |

根因：iluvatar 后端没有 flash_attn 包，attention 落到 vLLM 通用 Triton
参考实现 `kernel_unified_attention`——16k prefill 每层每 2048-chunk
36.8ms，占 GPU busy 91.8%。

## A1.5（最终版）：拆掉 kv≥8192 阈值门，全 kv 走 vendor（2026-10-12）

A1 曾因 4k pace A/B 误判加过 kv≥8192 门控；内核级实测 vendor 在
所有 kv 长度赢 12-20 倍，门拆掉后 4k 大涨。

| 场景 | 基线 | A1.5 | Δ |
|---|---|---|---|
| 4k | 1902.15 | **2635.85** | **+38.6%** |
| 16k | 861.98 | **2466.23** | **+186.1%** |

TTFT：4k 12.3s→4.8s；16k 636.5s→193.7s。
精度 MATH-500 L3 = **97.1%**（≥0.95 达标；A1 版测过 98.1%，同一路径
数值语义未变，波动在评测噪声内）。

详见 `a1-iluvatar-vendor-fa.md`（探针矩阵、cmt 语义、三个连环坑：
per-layer tolist 同步 / 分配器风暴 / 阈值门控误判与拆除）。

decode 侧（A2 调研，见 `a2-decode-attention-investigation.md`）：
初版自研 Triton flash-decoding 与 stock 默认配置接近，该结果只适用于当时实现与配置。
后续 A7 发射配置和 A20 页对齐内核继续优化 decode，不再沿用“A2 已无优化空间”的旧结论。

## A3：host 侧修复（2026-10-13，见 `a3-host-sync-fix.md`）

补抓 A1.5 的 16k 端到端 profile 发现 builder 里 `aten::index` 每步
2 次隐式阻塞式 HtoD+全流同步（占混合步 wall 57%）。修复后混合步
wall 677→353ms、GPU busy 41%→76%；32GB 卡端到端持平（16k 受
decode/尾部/KV 排队支配），不引用其他显存规格推断收益。
最终官方 benchmark：4k 2582.04 / 16k 2488.58。

## A4：GEMM pinned-config（2026-10-13，见 `a4-gemm-pinned-configs.md`）

libtuner 运行时 autotune 在 event-timing 降级模式下误选 mm_kernel
配置。对 8 个 serve 形状离线全空间扫描后在派发层 pin 最优配置：
decode GEMM 微基准 1.22-1.81x，prefill 追平/反超 vendor。
端到端 32GB 持平（4k 2579.14 / 16k 2485.52，噪声内），微基准收益未转化为整网收益。
精度 97.1% 达标（与 A3 一致）。
FlagGems commit `5eac6b39`（FlagGems/dev）。

## A5：sampler 尾部融合（2026-10-14，见 `a5-sampler-fusion.md`）

C500 opt7 移植：sort-free 融合采样内核（bf16 直读 + 温度折叠
Gumbel-Max + 1024-bin 直方图 zoom top-p），每 decode 步采样尾部
12.4ms→0.67ms（18.6x）。端到端 **16k 2532.68（+1.90%）** / 4k
2587.84（+0.34%），TTFT 无回退，精度 97.1% 达标（与 A4 一致）。
FL commit `72a83bf3`（vllm-plugin-FL/iluvatar-vendor-fa）；
内核 FlagGems/dev 已含。

## A6：varlen vendor flash prefill + Triton dual-gather（2026-10-14，见 `a6-varlen-prefill.md`）

prefill 段每请求 kernel 发射 4→2：自研 Triton 内核单发射收集
K+V 分页（bit-exact，GPU 106→62µs），ixformer varlen flash 经
`out=` 直写输出切片消掉 copy-out（全形状 bit-exact 探针验证）。
pre 段 host 1.77→~0.13ms/call。端到端 **4k 2738.65（+5.83%）** /
**16k 2558.37（+1.01%）**，TTFT 双场景均改善，精度 98.1% 达标。
FL commit `ef194318`（vllm-plugin-FL/iluvatar-vendor-fa）。

## A7：decode attention 发射配置 pin（2026-10-14，见 `a7-decode-launch-pin.md`）

赛题组确认 shape-aware dispatch 合规后补 A2 的配置维度：stock
Triton 默认 w4 s3 是 NVIDIA 甜点，在本芯片上严重失配。离线扫描
后 3D 路径 pin w1 s1（TILE=16/SEG=16 本已最优），bit-exact。
微基准 16k 6464→3524µs（1.83x）、4k 1938→1099µs（1.76x）。
端到端 **4k 3778.65（+38.0%）** / **16k 3567.33（+39.4%）**，
TTFT 大降（16k mean 186.4s→137.0s），精度 98.1% 达标。
FL commit `3f05c36d`（vllm-plugin-FL/iluvatar-vendor-fa）。

## A8/A9：收尾扫描（阴性，2026-10-14，见 `a8-a9-negative-results.md`）

lm_head pin（带宽墙）、sampler 重调（现配置已最优）、dec host
壳子（算账 <1%）、n-gram 投机（75% 请求零命中）全部否决，
配置层油水榨干。**天数侧以 A7 收官：4k +98.7% / 16k +313.7%**。
