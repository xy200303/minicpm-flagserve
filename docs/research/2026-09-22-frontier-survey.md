# MiniCPM-FlagServe 前沿调研与调优候选报告

日期：2026-09-22 ｜ 范围：LLM 推理吞吐优化前沿（开源项目 / 论文 / 社区实践）
对象：MiniCPM5-2B（Llama 架构，42 层，hidden 2048，GQA 16Q/2KV，head_dim 128，vocab 130560）
平台：沐曦 C500 64GB（104 SM，warp 64）/ 天数 BI-V150，vLLM 0.24 + vllm-plugin-FL + FlagGems v5.3.5
评测场景：[4096,1024,64,256]（4k）与 [16384,1024,64,128]（16k），指标 total tok/s，TTFT ≤ 基线+1%，精度 ≥0.95

## 0. 当前状态与瓶颈画像

已完成：基线复现（16k +0.13% 对齐）、优化 #1 免排序 top-k/top-p 采样内核（4k +27%）、
优化 #2 skinny-M 双缓冲 GEMM + split-K（累计 4k **+44.6%** / 16k **+17.4%**）。

剩余瓶颈（profile 实证）：

| 瓶颈 | 证据 | 量级 |
|---|---|---|
| GPU 空闲 ~40%（CPU 喂不动） | 弱 CPU（i7-8550U）上每步大量 aten 调用、pageable DtoH、event 同步 | TPOT 46ms 中 ~8ms 空闲 |
| Attention（16k 主导） | flash_fwd_splitkv 6.9s / 24.3s GPU busy | 16k decode 每步 KV 读 ~45GB |
| GEMM 距厂商库仍有差距 | 双缓冲后 45-90µs vs 厂商 16-42µs | C500 LSU throttle / 布局转换 |
| 采样尾部小内核群 | softmax/argmax/div/multinomial 分散在图外 | ~0.4ms GPU + launch 开销/步 |

---

## 1. 运行时/调度开销（CPU-bound decode）

**核心发现**：这是行业已知并被系统性治理的问题，vLLM 0.24 已内置答案，但 **MetaX 自定义平台路径很可能没有吃到**。

### 1.1 Async Scheduling（最高优先级，先排查）

- 机制：GPU 执行第 N 步时 CPU 调度第 N+1 步；采样结果的 D2H 走 pinned buffer + event 异步拷贝，移除每步同步点。vLLM v0.10 引入（PR #19970），与 SGLang overlap scheduler（future-token 机制）、TRT-LLM overlap scheduler（默认开启）同构——三大引擎 2024Q4-2025Q3 已收敛到同一设计。
- **我们的风险点**：`SchedulerConfig.async_scheduling = None`（auto），对自定义 platform 的 auto 判定很可能落到"关"。
- 落地（合规）：插件 platform 的 `check_and_update_config` 钩子强制 `async_scheduling=True`，不改 serve 参数；同时确认 runner 支持 async bookkeeping。
- 预期：直接命中 40% GPU 空闲的大头；小模型收益最大（同类机制在 SGLang 小模型上 1.1-1.3×）。

### 1.2 配套项

- cudagraph 模式确认 decode 全图覆盖（含 lm_head/sampler 段），消灭图外 aten::mm 的 CPU launch；
- 进程拓扑：i7-8550U 仅 4 核，API server/EngineCore/worker 要 taskset 绑核避免争抢；
- pageable DtoH → pinned + non_blocking（profile 中 5460 次 × 42µs）；
- stream_interval 默认值、detokenization 降频/fastokens；
- Model Runner V2（输入张量 GPU 化构造）是上游终态，可评估 backport。

参考：WukLab《Can Scheduling Overhead Dominate LLM Inference Performance?》（调度开销可占一半以上，大头是输入构造/metadata/detokenize 而非调度算法本身）；[vLLM V1 架构博客](https://vllm.ai/blog/2025-01-27-v1-alpha-release)；[SGLang v0.4](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/)；[TRT-LLM overlap scheduler](https://nvidia.github.io/TensorRT-LLM/features/overlap-scheduler.html)；Blink (arXiv 2604.07609)；arXiv 2603.22774（CPU 配置是一阶变量）。

---

## 2. Attention（GQA 16:2，16k 场景）

**Roofline**：decode 纯 memory-bound；bs64/16k 每步 KV 读 ~45GB → 理论下限 ~28ms@1.6TB/s。
occupancy 警示：不 split-KV 时并行单元 = 64×2 = 128 CTA / 104 SM = 1.23 wave，尾部浪费 ~20%。

### 2.1 GQA head packing（首要排查项）

TRT-LLM XQA / Colfax FA3-for-inference：同组 8 个 Q head 打包进 GEMM 的 M 维（M=8 pad 16），
**KV 从 HBM 只读一次**、SMEM 广播给组内 heads，走 tensor core 而非 GEMV。
若 vendor 内核逐 Q head 读 KV，流量放大 8 倍——**先用 profiler 实测 vendor splitkv 的实际
HBM 流量**，决定这是数倍级还是 5-15% 的优化。

### 2.2 统一 max / 条件 rescale（quick win）

FlashDecoding++（arXiv 2311.01282）与 FA4（2026）共识：online softmax 的 per-tile max 同步/rescale
是 decode kernel 第一可省开销（~20%）。GQA 下每 CTA 维护 8 个 Q head 的 softmax 状态，rescale
按 head 数放大。用预设统一 max 或条件 rescale 替代。Triton 1-2 天。

### 2.3 FlashInfer plan/run 自适应 split（工程量大但命中痛点）

host 端 plan（10-50µs）按每请求真实 KV 长度计算 split 划分 → 静态配置供 cudagraph 捕获；
长序列多切、短序列不切，消除 wave 尾部。混合 4k/16k 长度下预期 attention +15-30%。
PersistentKV（arXiv 2606.26666，2026）验证同思路在 GQA 商品卡上的收益。工程量 2-3 周。

### 2.4 KV block size 调优（几乎零成本）

block 16→32/64：减少 block-table 间接寻址、提高访存合并度，paged decode 通常 +3-10%。
前提是内核按 tile 流式扫而非按 block 跳转。

参考：[FlashDecoding](https://pytorch.org/blog/flash-decoding/)；[FlashInfer arXiv 2501.01005](https://arxiv.org/abs/2501.01005)；[XQA kernel blog](https://nvidia.github.io/TensorRT-LLM/blogs/XQA-kernel.html)；[Colfax FA3 GQA head packing](https://research.colfax-intl.com/flashattention-3-for-inference-int8-quantization-and-query-head-packing-for-mqa-gqa-external/)；LeanAttention（arXiv 2405.10480，Stream-K 均衡，二期）；Pod-Attention（arXiv 2410.18038，prefill/decode 同 grid 融合）；vAttention（arXiv 2405.04437）。

---

## 3. GEMM（skinny-M 已优化后的下一程）

**C500 微架构一手资料**（[kungal 踩坑实录](https://www.kungal.com/topic/4254)）：每 AP smem 仅 64KB；
每线程 ≤255 寄存器；MMA latency=throughput=16 cycle（不需要 NV 式多 accumulator ILP）；
**LSU throttle 低**——in-flight 访存指令过多会阻塞发射端，workaround 是减小单次请求粒度、
更细粒度流水。这很可能就是我们双缓冲内核与厂商库差距的来源。

### 3.1 沐曦专属：native layout 内核库（最对口蓝本）

[QingCheng-AI/muxi_native_layout_kernels](https://github.com/QingCheng-AI/muxi_native_layout_kernels)
（Apache-2.0）：权重在 global memory 预排成 C500 `mma_16x16x16f16` 指令的 TV layout，
省 smem layout 转换与 swizzle——Marlin weight-streaming 思想的 MetaX 落地版。
含 C500 实测调优 csv。建议先照其 layout 做一层 PoC。

### 3.2 单内核化 Split-K + StaggerU

- AMD FlyDSL（2026-06）：CTA 间 Split-K + warp 级 K-slice + 多级 LDS 环形流水 + **信号量单内核归并**
  （省掉 partial+reduce 两次 launch）；MI355X 上对最优基线 1.64×。
- StaggerU：K 为大 2 次幂时各 CTA 从 K=0 同时出发会撞同一 DRAM channel；给 K 起点加偏移错开，Triton 可实现，零成本试。
- Split-K 因子非单调（attn 投影要 8、FFN 要 4）——交给离线调优表，别写启发式。

### 3.3 lm_head：融采样进 GEMM（FlashSampling，2026-03）

采样只需 argmax/top-k，不需要完整 logits 落地：GEMM epilogue 按 tile 流式加 Gumbel 噪声、
每行每 tile 只留一个 (score, idx) 候选，跨 tile 小归并。**logits 永不写 HBM**
（我们场景每步省 ~33MB 流量 + 一串小内核）。端到端 TPOT 最多 -10%（H100-B300 实测）。
远期高价值项（要重写 lm_head GEMM）。[arXiv 2603.15854](https://arxiv.org/abs/2603.15854)

### 3.4 非 NV Triton 的通用结论

- Pipeliner 失效是长期事实（ROCm 上 num_stages≥4 多 load 编译崩溃等 issue 至今 open），
  **手动流水是正解不是权宜**；
- 排障标准动作：dump TTGIR/LLVM-IR 核对关键 pass 是否真生效；
- FlagTree（FlagOS 统一编译器）是沐曦/天数 Triton 的官方通道，其编译指导扩展值得关注。

参考：[AMD 低延迟 GEMM 博客](https://rocm.blogs.amd.com/software-tools-optimization/accelerating-llm-inference-on-amd-gpus-with-low-latency-gemms/README.html)；[AMD Triton 优化清单](https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html)；[MARLIN](https://arxiv.org/pdf/2408.11743)；tritonBLAS 解析式选参（arXiv 2512.04226）。

---

## 4. 采样器（已做 bisection 后的下一步）

### 4.1 Gumbel-Max 全融合（推荐，工程量小）

`argmax_i(x_i + g_i), g~Gumbel(0,1)` ≡ `multinomial(softmax(x))`。链路变为：
`load tile → ÷temperature → 阈值掩码（复用我们的 τ_k/τ_p）→ +gumbel → argmax`——
**单 kernel 单遍 vocab**（~30-80µs），softmax/cumsum/multinomial 全消失，归一化常数不需要。

**Graph-safety 关键规则**：seed/offset 必须以 **GPU tensor 指针**传入（kernel 内 tl.load），
绝不能作 Triton 标量参数（会被烘进 cudagraph，replay 时随机数冻结——SGLang Omni 出过这个事故）。
每次 replay 前用小内核 offset += step。u 要 clamp 到 [ε, 1-ε]；噪声 fp32 计算；
batch 内异构采样参数用 per-request tensor 在 kernel 内 load，不要 host 分支。
参考实现：vLLM `v1/worker/gpu/sample/gumbel.py`、FlashInfer tensor seed/offset 设计。

### 4.2 FlashInfer dual-pivot rejection sampling

单 kernel 完成过滤+采样（inverse transform + 双 pivot 拒绝采样，O(log(1/ε)) 轮收敛）。
vLLM 1×H100 采样耗时 -50%。需要 fp32 输入与 graph-safe seed 管理；可作为对照参考。
[FlashInfer sampling 博客](https://flashinfer.ai/2025/03/10/sampling.html)

### 4.3 SonicSampler（2026，远期参考）

Triton 两阶段（tile 内 top-128 + 跨 tile 合并）+ Gumbel epilogue；bit-level indicator 消灭 host 分支。
对 FlashInfer 5-16×（大 batch spec-verify 场景）；我们 B=64 收益温和，但其「噪声外置 +
indicator 驱动」设计模式值得借鉴。[arXiv 2607.20475](https://arxiv.org/html/2607.20475v1)

---

## 5. 专用资源/社区清单

| 资源 | 内容 | 用法 |
|---|---|---|
| [GPU MODE](https://gpumode.com)（Discord 24K） | 最大 GPU 内核社区；KernelBot 竞赛（2025 三场 $100K+ 赛、40 万提交）、92 场 YouTube 讲座 | 排行榜题目=生产级 LLM 内核模式（FP8/MLA/skinny GEMM），前排代码公开可参考 |
| [popcorn-cli](https://github.com/gpu-mode/popcorn-cli) | KernelBot 提交工具 | 本地验证后提交打榜 |
| [KernelIndex](https://kernelindex.com) | 聚合各平台内核成绩 | 查算子在各卡上的最优实现 |
| [FlashInfer 内核赛](https://mlsys26.flashinfer.ai/) | MLSys 2026 内核生成竞赛 | 采样/attention 内核设计参考 |
| [BackendBench](https://github.com/meta-pytorch/BackendBench) / KernelBench / TritonBench | 算子正确性+性能评测套件 | hot-swap 替换 ATen 内核的机制参考 |
| [HF Kernels](https://github.com/huggingface/kernels) | 内核共享 Hub | 分发/组织方式参考 |
| AMD ROCm kernel blogs / Tri Dao 博客 / FlashInfer 博客 / LMSYS / vLLM 博客 | 一手技术文章 | AMD 博客对非 NV Triton 迁移价值最高 |
| [muxi_native_layout_kernels](https://github.com/QingCheng-AI/muxi_native_layout_kernels) | **沐曦专用** GEMM 内核库 | 我们 GEMM 的直接蓝本 |
| FlagTree / FlagOS 生态 | 国产卡统一 Triton 编译器 | 沐曦/天数 Triton 扩展官方通道 |

---

## 6. 候选优化清单（按性价比排序）

| # | 项目 | 预期收益 | 工程量 | 风险 |
|---|---|---|---|---|
| 1 | **排查/强制 async scheduling**（插件 config 钩子） | 命中 40% GPU 空闲，可能两位数 % | 小时级排查 + 验证 | 低（0.24 已内置，验证正确性） |
| 2 | CPU 进程绑核 + pinned DtoH + stream_interval | 削调度抖动（4k 波动 ±10%） | 1 天 | 低 |
| 3 | **Gumbel-Max 采样全融合**（单 kernel，进 graph） | 每步省 ~0.4ms+launch | 3-5 天 | 中（RNG graph 坑已明确） |
| 4 | Attention head packing（先实测 vendor 是否已共享 KV） | 5-15% 或数倍 | 1-2 周 | 中 |
| 5 | 统一 max/条件 rescale | 5-15%（attention） | 1-2 天 | 低（需数值 fallback） |
| 6 | KV block size 16→32/64 | 3-10%（16k） | ~1 天 | 低 |
| 7 | GEMM：muxi native layout PoC + StaggerU + 单内核 Split-K | 逼近厂商 16-42µs | 1-2 周 | 中 |
| 8 | FlashInfer 式 plan/run 自适应 split | 10-20%（16k attention） | 2-3 周 | 中 |
| 9 | FlashSampling（lm_head 融采样） | TPOT 最多 -10% | 2 周+ | 高（重写 lm_head） |
| 10 | FP8 KV cache（官网已明文允许自研量化） | 16k KV 流量减半 | 2 周+ | **高（精度余量薄，暂缓）** |

**立即执行建议**：#1 → #3 → #4/#5（并行验证）→ #2。天数卡审批下来后优先做基线复现 +
两个已有内核的移植（warp size 同为 64，思路直接复用）。
