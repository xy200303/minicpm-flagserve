# MiniCPM5-2B 新一代算子优化方案

本文是面向 FlagOS 比赛场景的独立优化方案，不以 `算力申请-XLANG2026.pdf` 中的申请内容作为技术实现约束。

目标模型为 MiniCPM5-2B，重点覆盖以下场景：

- 4K decode：`[4096, 1024, 64, 256]`
- 16K decode：`[16384, 1024, 64, 128]`
- GQA：16 个 Q head、2 个 KV head、`head_dim=128`
- 精度门槛：`math_500 Level 3 >= 0.95`
- TTFT 不能明显劣化，最终必须使用官方评测命令验证

## 一、总体路线

不再把主要精力集中在通用 GEMM 参数调优上，而是围绕 decode 阶段的三个主要问题进行算子级融合：

```text
GQA KV 重复读取
        ↓
GQA 分组分页 Attention

QKV / RoPE / KV Cache 多次中间写回
        ↓
RMSNorm + QKV + RoPE + Cache 融合

MLP gate/up 中间 tensor 写回
        ↓
Gate/Up + SwiGLU Epilogue

采样尾部 softmax、multinomial 和同步开销
        ↓
Top-P 两阶段 Gumbel Sampling
```

推荐优先级：

| 优先级 | 算子 | 主要收益场景 | 预期端到端收益 | 风险 |
|---|---|---|---:|---|
| P0 | GQA 分组分页 Attention | 16K | 10%~25% | 中 |
| P1 | RMSNorm + QKV + RoPE + KV Cache | 4K/16K decode | 3%~8% | 中 |
| P1 | Gate/Up + SwiGLU Epilogue | 4K decode | 2%~4% | 低 |
| P2 | Top-P Gumbel Sampling | 4K decode | 3%~8% | 中 |
| P3 | INT8 KV Cache | 16K | 10%~25% | 高 |

收益为设计阶段估计，必须通过官方 benchmark 和逐算子 microbenchmark 验证，不能直接作为最终成绩承诺。

## 一、自主研发算子的边界与定位

本方案中的“自主研发新算子”不是改变 MiniCPM5-2B 的模型结构，也不是简单替换一个已有库函数，而是：

```text
保持原始数学语义
+ 重新设计线程映射、数据布局、显存访问和归约方式
+ 在 FlagGems 中实现新的 kernel
+ 在 vllm-plugin-FL 中接入真实推理路径
```

因此，以下两类工作都属于有效的算子创新：

1. **等价重写算子**：例如双缓冲 GEMM、Split-K GEMM、GQA Attention；
2. **复合融合算子**：例如 RMSNorm + QKV + RoPE + KV Cache、Gate/Up + SwiGLU。

GQA 本身是 MiniCPM5-2B 已有的模型结构，不能单独称为新算法；真正的创新点应表述为：

```text
GQA-aware Head-Packed Paged Decode Attention Kernel
```

即针对 16Q/2KV 结构重新设计 KV 加载复用、线程组织和 Split-KV 归并的专用内核。

以下方式不应作为主要创新点：

- 仅增加 Python wrapper；
- 仅修改 dispatch 路由而不重写 kernel；
- 仅切换到已有 FlashAttention/FlashInfer 实现；
- 仅修改 benchmark 参数；
- 通过开启框架已有量化或投机采样配置获得收益。

## 二、推荐的自主研发算子组合

### 2.1 核心创新：C500-GQA Adaptive Paged Decode Attention

设计目标：在不改变标准 Attention 数学语义的前提下，针对 MiniCPM5-2B 的 16Q/2KV 结构减少 KV Cache 重复读取。

算子包含四个关键机制：

```text
GQA 分组 KV 加载
+ Head Packing
+ Paged KV Cache
+ 自适应 Split-KV
+ Online Softmax
```

每个 CTA 负责一个 KV head 分组：

```text
一次加载 KV tile
→ LDS/共享内存广播给对应的 8 个 Q head
→ 寄存器中维护 8 组 online softmax 状态
→ 只归并 max/sum/output
```

推荐命名：

```text
C500-GQA Decode Attention
Head-Packed Paged Attention
Adaptive GQA Split-KV Kernel
```

该算子是本方案最主要的原创 Kernel，重点面向 16K decode 场景。

### 2.2 配套融合：RMSNorm-QKV-RoPE-Cache

将以下步骤合并为一个 decode 专用算子：

```text
RMSNorm
→ QKV Projection
→ Q/K RoPE
→ KV Cache Append
```

核心创新不是把多个 Python 函数放进一个函数，而是让：

- Norm 结果不单独写回显存；
- QKV 直接生成 Attention 需要的布局；
- Q/K 在寄存器或片上缓存中完成 RoPE；
- K/V 直接写入目标 KV Cache；
- 消除中间 tensor、reshape、transpose 和 copy。

### 2.3 低风险补充：Gate-Up-SwiGLU Fused GEMM

将：

```text
gate_proj
+ up_proj
+ SiLU
+ elementwise multiply
```

实现为一个 decode 专用复合算子：

```text
Fused Gate-Up-SwiGLU GEMM
```

内部执行：

```text
gate, up = GEMM(x, W)
output = SiLU(gate) * up
```

该算子减少 gate/up 中间结果的显存写回，工程风险低，适合作为核心 Attention 之外的第二个自主算子。

### 2.4 冲刺方向：LM Head + Sampling

可进一步研究：

```text
LM Head GEMM
→ Top-P threshold
→ Gumbel sampling
```

目标是避免完整 logits 落地，并消除 softmax、multinomial 和部分采样同步。该方案涉及大词表归约、随机数状态和 cudagraph，作为冲刺方向，不作为首版依赖。

### 2.5 高风险方向：低比特 KV Cache

可以设计：

```text
INT8 KV Cache
→ Attention 内核内反量化
```

这属于自主研发的存储和计算算子，但会改变 Attention 输入的数值误差，必须经过 math_500、多随机种子和长序列验证。精度不稳定时，保留 BF16 fallback，不作为主路径。

## 三、最终技术路线表述

建议在技术报告中将自主研发部分表述为：

> 在保持 MiniCPM5-2B 标准 Transformer/GQA 数学语义和权重接口不变的前提下，面向 MetaX C500 decode 场景自主研发 C500-GQA Adaptive Paged Decode Attention 内核，并配套实现 RMSNorm-QKV-RoPE-KVCache 与 Gate-Up-SwiGLU 融合算子。方案通过 GQA 组内 KV 复用、Head Packing、自适应 Split-KV、片上在线 Softmax 和中间结果消除，降低 KV Cache 带宽与 kernel launch 开销。

推荐提交组合：

```text
C500-GQA Adaptive Paged Decode Attention
+ RMSNorm-QKV-RoPE-KVCache Fusion
+ Gate-Up-SwiGLU Fused GEMM
+ Sort-Free Top-K/Top-P
+ Skinny-M Double-Buffered GEMM
```

其中前 3 项构成新增自主研发算子，后 2 项是当前已经完成的基础优化。

## 四、性能真实性原则：不为创新而过度优化

自主研发不等于堆叠更多 kernel，也不等于为了报告中的“创新点”强行重写已经足够高效的路径。本方案采用以下原则：

### 4.1 先定位瓶颈，再决定是否自研

任何新算子必须先有 profile 或 microbenchmark 证据，至少回答：

- 该算子在端到端时间中占比多少；
- 主要瓶颈是计算、显存带宽、kernel launch 还是 CPU 调度；
- vendor kernel 是否已经实现了目标优化；
- 新 kernel 是否可能被寄存器、LDS 或额外归约开销抵消。

如果 vendor Attention 已经完成 GQA KV 复用，则不再重复开发一个仅改变线程映射的 GQA kernel，而转向确认真正剩余的瓶颈，例如 KV layout、split-KV combine 或 launch gap。

### 4.2 采用“单项收益门槛”

每个新算子接入正式路径前，必须满足：

```text
算子 microbenchmark：目标形状至少快 10%
端到端 benchmark：至少快 3%
不能使 TTFT 超过限制
精度和输出语义通过验证
```

如果单项 kernel 很快，但端到端收益低于 3%，则不作为主路径，只保留为实验记录或后续融合候选。

### 4.3 优先选择低改造、高复用的算子

推荐顺序：

1. 修改现有热点 kernel 的布局、流水和归约；
2. 融合紧邻且存在中间 tensor 写回的算子；
3. 优化 KV Cache 物理布局和访问方式；
4. 最后才考虑改变存储精度或改变 Attention 算法。

不优先开发：

- 完全重写已经接近 vendor 峰值的 prefill Attention；
- 为少量 elementwise 操作设计大型 persistent kernel；
- 没有请求前缀复用证据时引入 Prefix Cache；
- 没有 draft model 时引入投机解码；
- 只为了创新分而引入稀疏 Attention 或低比特 KV。

## 五、收敛后的真正提速路线

### 5.1 P0：确认 Attention 的真实剩余瓶颈

先对当前 `flash_fwd_splitkv_kernel` 做硬件 profile，测量：

- HBM/L2 读写量；
- Q head 是否已经共享 KV tile；
- split-KV 的 partial/combine 时间；
- warp64 下的寄存器和 LDS 使用量；
- block table 和 KV cache layout 带来的访存损失。

只有确认存在重复 KV 读取或 combine/launch 成本后，才实现下面的专用 kernel。

### 5.2 P1：C500 GQA KV Layout + Adaptive Split-KV

这不是盲目重写 Attention，而是只改 profile 证明有问题的部分：

```text
KV cache 物理布局优化
+ Q head 组内 KV tile 复用
+ 根据 seq_len/batch 选择 split 数
+ 减少 partial tensor 和 combine 开销
```

优先保留 vendor Attention 作为 fallback，仅对确定收益的 decode shape 路由到新 kernel。

### 5.3 P1：现有双缓冲 GEMM 的 Native Layout 版本

当前 GEMM 已经证明有效，下一步只做两项有明确依据的增强：

1. 对 QKV、O、Gate/Up、Down 的实际权重做 C500 原生 tile layout 预排布；
2. 对反复出现的 decode shape 使用 persistent CTA，减少重复 launch 和 layout conversion。

不对 lm_head 或大 M prefill 强行套用该路径，避免破坏已有高吞吐实现。

### 5.4 P1：局部融合，而不是巨型 Super Kernel

优先实现收益明确的局部融合：

```text
RMSNorm + QKV + RoPE + Cache Append
Gate/Up + SiLU + Multiply
O Projection + Residual Add
Down Projection + Residual Add
```

不建议第一阶段把完整 Transformer block 融合成单个 kernel。巨型 kernel 容易带来寄存器溢出、occupancy 下降、编译时间增长和异常难以回退等问题。

### 5.5 P2：采样尾部和运行时间隙

只有在 Attention/GEMM 优化后仍然确认 sampler 或 CPU launch gap 占比较高时，才继续做：

- Top-P + Gumbel 单 kernel；
- pinned D2H；
- async scheduling；
- decode graph 覆盖。

这些优化要用端到端 TPOT、GPU busy ratio 和 kernel launch gap 证明收益，不能只看单个 kernel 时间。

### 5.6 暂缓：低比特 KV 和动态稀疏

INT8/低比特 KV Cache、动态稀疏 Attention 具有研究价值，但会改变数值误差或计算语义。除非 dense 路径已经优化到瓶颈明确且精度余量足够，否则不作为首版主线。

## 六、方案决策表

| 候选方向 | 是否立即做 | 接入条件 | 失败时处理 |
|---|---|---|---|
| C500 GQA KV layout | 做 profile 后做 | 证实 KV 重复读或 layout 低效 | 回退 vendor Attention |
| Adaptive Split-KV | 做 | combine/并行度成为瓶颈 | 使用现有 split-kv |
| Native-layout GEMM | 做 | 预排布后实际 shape 明显加速 | 保留当前双缓冲 GEMM |
| RMSNorm/QKV/RoPE/Cache | 做 | 中间写回和 launch 占比明显 | 保留独立算子 |
| Gate/Up/SwiGLU | 做 | MLP 中间 tensor 写回占比高 | 保留现有 GEMM |
| Gumbel Sampling | 后做 | sampler 仍是 decode 热点 | 保留 sort-free sampler |
| INT8 KV Cache | 暂缓 | dense kernel 已优化且精度余量充分 | BF16 主路径 |
| 动态稀疏 Attention | 暂缓 | 有明确稀疏性和无损证据 | dense Attention |

最终目标不是让方案包含最多的“新算子”，而是让每一个保留的自研算子都能回答：

```text
它解决了哪个已测量瓶颈？
它减少了哪些真实开销？
它在官方 4K/16K 场景带来了多少端到端收益？
```

## 二、P0：GQA 分组分页 Attention

### 2.1 设计目标

MiniCPM5-2B 中每个 KV head 对应 8 个 Q head。如果不同 Q head 分别加载 KV，会重复读取同一份数据，导致 16K decode 受显存带宽限制。

新增 decode 专用接口：

```python
gqa_paged_decode_attention(
    q,              # [batch, 16, 128]
    k_cache,        # paged KV cache
    v_cache,
    block_table,
    seq_lens,
    softmax_scale,
)
```

### 2.2 Kernel 组织

建议使用如下 grid：

```text
grid = (batch, num_kv_heads=2, split_kv)
```

每个 CTA 处理一个 KV head 组，即同时处理 8 个 Q head：

1. 从 KV cache 加载一个 K/V tile；
2. 将 K/V tile 放入 LDS/共享内存；
3. 广播给同一组的 8 个 Q head；
4. 在寄存器中维护 8 组 online softmax 的 `max/sum/output`；
5. 输出 split partial 结果；
6. 使用轻量 reduce kernel 合并 `(max, sum, output)`。

不要在 split 之间归并完整 attention 矩阵，只归并 online softmax 状态。

### 2.3 建议调优参数

- `BLOCK_N = 64 / 128`
- `head_dim = 128`
- `split_kv = 1 / 2 / 4 / 8`
- `seq_len < 4096`：优先少 split，降低归并开销
- `seq_len >= 8192`：提高 split，增加 CTA 数量
- 4K 与 16K 分别建立离线配置表，不使用单一启发式参数

建议只在以下条件下启用：

```text
decode 阶段
M <= 128
num_q_heads / num_kv_heads == 8
head_dim == 128
dtype 为 bf16 或 fp16
```

### 2.4 正确性要求

需要与现有 vendor Attention 对比：

- fp32 reference 与 bf16/fp16 输出误差；
- 不同 `seq_lens`、不同 block table；
- 4K、16K、32K 长度；
- split 为 1、2、4、8；
- batch 为 1、16、64；
- causal mask 和尾部不完整 block。

## 三、P1：RMSNorm + QKV + RoPE + KV Cache 融合

### 3.1 原始链路

```text
RMSNorm
→ QKV GEMM
→ reshape / split
→ RoPE
→ KV cache append
→ Attention
```

### 3.2 融合链路

```text
load hidden
→ 寄存器中计算 RMSNorm
→ QKV projection
→ Q/K 应用 RoPE
→ Q 写入 attention staging buffer
→ K/V 直接写入 KV cache
```

建议接口：

```python
rmsnorm_qkv_rope_cache(
    hidden_states,
    qkv_weight,
    norm_weight,
    cos,
    sin,
    k_cache,
    v_cache,
    slot_mapping,
)
```

### 3.3 优化重点

重点消除以下中间结果：

- normalized hidden 的中间写回；
- 完整 QKV contiguous tensor；
- Q/K reshape 和 transpose；
- 独立 RoPE kernel；
- 独立 KV cache append kernel。

只在 decode 场景启用：

```text
M <= 128
batch >= 16
dtype 为 bf16/fp16
```

prefill 继续使用原始高吞吐路径，避免融合 kernel 在大 M 下损失 GEMM 效率。

## 四、P1：Gate/Up + SwiGLU Epilogue

### 4.1 原始链路

```text
gate_up GEMM
→ split gate / up
→ SiLU(gate) * up
→ down GEMM
```

### 4.2 新链路

```text
gate, up = GEMM(hidden, gate_up_weight)
output = silu(gate) * up
```

建议新增 decode 专用算子：

```python
fused_gate_up_silu(
    hidden_states,
    gate_up_weight,
    output,
)
```

实现要求：

- GEMM 使用 fp32 累加；
- SiLU 和乘法在 fp32 中计算；
- 最终输出转换为 bf16/fp16；
- 不写回完整的 `[gate, up]` 双倍宽度结果；
- `M > 128` 时回退到原始路径。

该方案工程风险低，适合在 GQA Attention 之后实现。

## 五、P2：Top-P 两阶段 Gumbel Sampling

当前免排序 top-k/top-p 已经减少了排序开销，但采样尾部仍可能包含 softmax、累计概率和 multinomial 等多个操作。

新增接口：

```python
top_p_gumbel_sample(
    logits,
    top_k,
    top_p,
    temperature,
    seed_tensor,
    offset_tensor,
)
```

### 5.1 两阶段流程

第一阶段：

```text
计算 top-k/top-p 阈值
```

第二阶段：

```text
对保留 token 计算 Gumbel score
argmax(logits + Gumbel) 得到采样 token
```

### 5.2 关键约束

- `seed` 和 `offset` 必须作为 GPU tensor 传入；
- 不能把随机种子作为 Triton 编译期 scalar；
- 必须复用当前 top-p 阈值语义；
- 对 `p=1.0`、`k=None`、重复 logits 做边界测试；
- 使用随机分布 TV 距离和 token 保留集 IoU 验证一致性。

该方案的目标不是改变采样策略，而是消除 softmax、multinomial 和多次同步。

## 六、P3：Block-wise INT8 KV Cache

如果 GQA Attention 完成后 16K 仍然明显受 KV 带宽限制，可尝试 block-wise INT8 KV cache：

```text
BF16 KV cache
→ 每个 block 独立 scale 的 INT8 KV cache
→ Attention 内核中寄存器反量化
```

每个 block 保存：

```text
int8 values
scale
```

Attention 内部执行：

```text
int8 load
→ fp16/fp32 dequant
→ QK dot / PV
```

该方案潜在收益较大，但必须放在最后验证。若 math 评测出现不稳定下降，不应作为首版提交方案。

## 七、实施顺序

### 阶段 1：Attention 原型

1. 先实现非 paged 的 GQA decode microbenchmark；
2. 验证 8 个 Q head 复用一份 KV 的正确性；
3. 接入 paged KV cache；
4. 调整 `BLOCK_N` 和 `split_kv`；
5. 接入 vLLM vendor dispatch。

### 阶段 2：融合算子

1. 实现 `fused_gate_up_silu`；
2. 实现 `rmsnorm_qkv_rope_cache`；
3. 对比融合前后的单层耗时和端到端吞吐；
4. 对 prefill 和 decode 分别设置路由条件。

### 阶段 3：采样尾部

1. 实现两阶段 top-p threshold + Gumbel argmax；
2. 验证随机分布一致性；
3. 验证 cudagraph replay 下随机数不会冻结；
4. 再接入正式评测路径。

### 阶段 4：可选量化

只有前三阶段稳定后，才进入 INT8 KV Cache 评估。

## 八、统一验收标准

每个新算子都必须同时满足：

1. 与 reference 输出误差在 bf16/fp16 正常舍入范围内；
2. 官方 `math_500 Level 3` 每次运行均不低于 `0.95`；
3. TTFT 不超过官方基线限制；
4. 4K 和 16K 都进行单独 benchmark；
5. 至少完成 5 次正式场景重复测试；
6. 记录 cold start、warm start、平均值和最差值；
7. 新算子异常时必须有可观测的 fallback，不允许静默退化。

## 九、最终推荐组合

首选提交组合：

```text
GQA 分组分页 Attention
+ RMSNorm/QKV/RoPE/Cache 融合
+ Gate/Up + SwiGLU Epilogue
+ Top-P Gumbel Sampling
```

其中 GQA Attention 是 16K 的核心突破点，QKV 和 SwiGLU 融合负责降低 decode 的 launch 与中间访存开销，Gumbel Sampling 负责进一步压缩采样尾部。

INT8 KV Cache 作为冲刺方向，不应在精度和稳定性尚未确认前替代 BF16 路径。
