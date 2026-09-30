# MiniCPM5-2B 推理优化前沿调研

本文调研 LLM 推理中与 MiniCPM5-2B、MetaX C500、FlagGems/vLLM 组合相关的前沿论文和社区实践，并判断其是否适合当前比赛。

当前实现已经覆盖：

- 免排序 top-k/top-p sampler；
- skinny-M 双缓冲 GEMM + split-K；
- MetaX vendor Attention；
- 基础 FlagGems RMSNorm、RoPE、SiLU 等算子。

因此后续创新不应只继续堆 GEMM 参数，而应扩展到 Attention 数据复用、运行时重叠、KV cache 压缩、长上下文稀疏化和解码算法。

## 一、结论先行

最适合本比赛、且不需要重新训练模型的方向有四个：

1. **GQA head packing + 自适应 split-KV Attention**：首要方向，直接针对 16K decode 的 KV 带宽和并行度问题；
2. **FlashAttention-3/FlashInfer 风格的异步流水和 persistent decode kernel**：减少 Attention 内部等待和 kernel launch；
3. **KV cache 的 block-wise INT8/低比特压缩**：降低 16K 场景的 KV 读取量，但必须严格做精度验证；
4. **运行时 overlap scheduling + pinned D2H + CUDA Graph 覆盖**：不是单一算子，但可能比继续微调 GEMM 更能解决 4K 的 GPU 空闲。

可以作为研究性创新、但不应作为首版主方案的方向：

- MInference 风格动态稀疏 prefill；
- Medusa/EAGLE 投机解码；
- RadixAttention/Prefix Cache；
- vAttention/连续 KV 内存管理。

这些方向要么改变模型计算语义，要么依赖请求前缀复用、额外训练头或更大规模的运行时改造。

## 二、候选方向总览

| 方向 | 参考工作 | 主要命中场景 | 预计收益 | 改造量 | 比赛适配度 |
|---|---|---|---:|---:|---:|
| GQA head packing + split-KV | FlashDecoding++、Lean Attention、FlashInfer | 16K decode | 10%~25% | 中 | 很高 |
| Attention 异步流水 | FlashAttention-3 | 16K prefill/decode | 5%~15% | 中高 | 高 |
| Persistent decode kernel | FlashInfer、社区 persistent kernels | 4K decode | 5%~15% | 高 | 高 |
| KV cache 低比特压缩 | KIVI、BitDecoding | 16K decode | 10%~30% | 中高 | 中高 |
| 动态稀疏 prefill | MInference 1.0 | 16K prefill | 15%~50% | 高 | 中 |
| 投机解码 | Speculative Decoding、Medusa、EAGLE | decode | 1.3x~2x | 很高 | 低中 |
| Async scheduling/overlap | vLLM、SGLang、TensorRT-LLM | 4K decode | 5%~30% | 中 | 很高 |
| Prefix Cache/RadixAttention | vLLM、SGLang | 重复前缀请求 | 1.2x~数倍 | 中 | 低中 |
| vAttention/连续内存 | vAttention | 高并发 KV 管理 | 场景相关 | 高 | 中低 |

收益是公开工作和经验范围，不能替代 C500 实测。

## 三、方向 1：GQA Head Packing + 自适应 Split-KV

### 3.1 理论来源

- [Flash-Decoding++](https://arxiv.org/abs/2311.01282)：通过 split-KV 增加长序列 decode 并行度，并优化 online softmax 归并；
- [Lean Attention](https://arxiv.org/abs/2405.10480)：针对 decode 阶段的硬件感知分块和负载均衡；
- [FlashInfer](https://arxiv.org/abs/2501.01005)：将 paged KV、variable-length、split-KV 和 attention plan/run 做成可组合的推理引擎。

### 3.2 针对 MiniCPM5-2B 的改造

模型为 16Q/2KV，每个 KV head 对应 8 个 Q head。kernel 不应让 8 个 Q head 各自从 HBM 读取 KV，而应：

```text
一个 CTA 负责一个 KV head 组
→ K/V tile 只加载一次
→ LDS/共享内存广播给 8 个 Q head
→ 寄存器中维护 8 组 online softmax
```

建议 grid：

```text
(batch, num_kv_heads=2, split_kv)
```

`split_kv` 根据实际 KV 长度和 batch 进行 plan，而不是固定写死：

```text
短序列：split=1
中等序列：split=2/4
长序列：split=4/8
```

### 3.3 为什么优先级最高

现有代码尚未改动 Attention，而 16K 场景本身是 KV 读取主导。GEMM 已经优化后，继续调 GEMM 的边际收益会下降，GQA 数据复用更可能带来结构性收益。

### 3.4 推荐实现方式

先实现非 paged 的 microbenchmark，再接入 paged KV cache。需要单独测量：

- vendor kernel 是否已经共享 GQA 的 KV 读取；
- 实际 HBM bytes 和 L2 命中率；
- `BLOCK_N=64/128`；
- `split=1/2/4/8`；
- 8 个 Q head 复用带来的收益是否被寄存器压力抵消。

## 四、方向 2：FlashAttention-3 风格异步流水

### 4.1 理论来源

[FlashAttention-3](https://arxiv.org/abs/2407.08608) 的核心思想不是简单增加 tile，而是让数据搬运、矩阵乘和 softmax 计算异步重叠，并通过低精度路径提升 tensor core 利用率。

### 4.2 对 MetaX 的可迁移部分

不应直接照搬 CUDA/Hopper 指令，而应迁移设计模式：

```text
双缓冲 K/V tile
→ 当前 tile 做 QK/PV
→ 下一 tile 异步预取
→ softmax 状态保持在寄存器
```

如果 MetaX Triton 软件流水同样不可靠，可以采用与当前 GEMM 相同的手动预取方式，但要特别控制：

- LDS 占用；
- 寄存器数量；
- K/V tile 的加载粒度；
- softmax rescale 的同步次数。

### 4.3 建议落点

优先放在 GQA Attention 内部，不建议一开始重写整套 prefill attention。先做 decode 专用 kernel，验证 `16K / batch=64` 的 HBM 和 kernel 时间。

## 五、方向 3：Persistent Decode Kernel

### 5.1 社区思路

FlashInfer、FasterTransformer、TensorRT-LLM 和 GPU kernel 社区普遍使用 persistent kernel 或 CUDA Graph，将多个小 decode 操作放入更少的固定 launch 中。

适合当前问题的原因是：

- batch=64 时每步有大量小 kernel；
- GPU profile 中存在明显空闲间隙；
- CPU launch 和 event 同步开销可能大于部分小算子本身。

### 5.2 可行的渐进式实现

不要直接把整个 Transformer block 合成一个巨型 kernel，可以分三步：

1. 将 `RoPE + Q/K layout + KV append` 合并；
2. 将 `sampling 后处理` 合并成单个 persistent kernel；
3. 对固定 batch/shape 使用 CUDA Graph 或 MetaX 对应 graph 机制覆盖 decode 主路径。

重点是减少 launch 次数，而不是单纯追求单个 kernel 的峰值 TFLOPS。

## 六、方向 4：KV Cache 低比特压缩

### 6.1 论文来源

- [KIVI](https://arxiv.org/abs/2402.02750)：针对 KV cache 的非对称 2-bit 量化；
- [BitDecoding](https://arxiv.org/abs/2503.18773)：探索低比特 KV cache 与 tensor core 解码路径；
- 社区中还普遍采用 per-channel、per-token 或 per-block 的 INT8 KV cache。

### 6.2 推荐技术路线

先不要直接做 2-bit，而采用风险较低的 block-wise INT8：

```text
每个 KV block 保存 int8 values + scale
→ Attention 内核中寄存器反量化
→ QK/PV 使用 fp16/fp32 累加
```

建议比较：

- per-token scale；
- per-head scale；
- per-block scale；
- block size 16/32/64。

### 6.3 风险

KV cache 量化会改变每一步的 attention 输入，可能引起长 CoT 累积误差。必须同时验证：

- math_500 Level 3；
- 长序列困惑度或生成一致性；
- 4K 与 16K 的独立精度；
- 不同随机种子的最差结果。

如果精度余量不足，不要把它作为首版提交路径。

## 七、方向 5：动态稀疏 Prefill

### 7.1 理论来源

[MInference 1.0](https://arxiv.org/abs/2407.02490) 通过动态识别不同 attention head 的稀疏模式，减少长上下文 prefill 的注意力计算量。

### 7.2 对当前比赛的适配性

16K 场景有明显 prefill 成分，因此理论上有收益。但它改变了 attention 计算语义，与当前官方精度门槛存在冲突。

更稳妥的研究方式是：

1. 先离线统计 MiniCPM5-2B 各层各 head 的 attention 稀疏性；
2. 只对明显局部化的 head 使用 block-sparse；
3. 对关键层保留 dense fallback；
4. 用固定稀疏模式与动态稀疏模式分别评测。

### 7.3 结论

这是有论文创新性的方向，但不建议作为当前版本的第一优先级。除非能够证明 accuracy 几乎无损，否则更适合作为报告中的探索项。

## 八、方向 6：投机解码

### 8.1 理论来源

- [Speculative Decoding](https://arxiv.org/abs/2203.16487)：使用小模型提出候选，大模型批量验证；
- [Medusa](https://arxiv.org/abs/2401.10774)：在主模型上增加多个 decoding heads；
- [EAGLE](https://arxiv.org/abs/2401.15077)：利用 feature uncertainty 训练轻量 draft head。

### 8.2 为什么当前不适合作为首选

比赛当前提供的是固定 MiniCPM5-2B 推理路径，投机解码通常需要：

- 额外 draft model；或
- 重新训练/加载 Medusa/EAGLE heads；
- 改写 scheduler 和 verify batch；
- 额外占用显存。

这会改变提交包和模型运行方式，合规性、收益稳定性和工程量都较高。

如果后续能获得同系列小模型，可以作为独立创新分支，但不应阻塞算子优化主线。

## 九、方向 7：Async Scheduling 与计算/通信重叠

### 9.1 社区实现

vLLM、SGLang、TensorRT-LLM 都在持续推进：

```text
GPU 执行第 N 步
CPU 同时准备第 N+1 步
```

典型配套包括：

- pinned D2H；
- non-blocking event；
- overlap scheduler；
- CUDA Graph decode；
- 减少 detokenization 和 Python 调度；
- 预分配采样和 metadata buffer。

### 9.2 对当前项目的价值

这不是传统算子，但当前 4K 场景有明显 decode 抖动和 GPU 空闲时，系统级重叠可能比继续优化一个 GEMM 更有效。

建议把它作为“运行时创新”单独提交到方案中：

1. 检查 MetaX plugin 的 async scheduling 是否实际开启；
2. 检查 sampler token 是否经过 pageable D2H；
3. 将采样结果和 scheduler event 改为 pinned/non-blocking；
4. 确认 decode 主路径能被 graph 覆盖；
5. 分别对 cold start 和 warm start 测量。

## 十、方向 8：Prefix Cache / RadixAttention

### 10.1 社区思路

vLLM PagedAttention、SGLang RadixAttention 和 HiCache 都利用请求之间的共享前缀复用 KV cache。

参考：[SGLang](https://arxiv.org/abs/2312.07104)。

### 10.2 适配判断

如果官方 benchmark 的请求 prompt 没有共享前缀，该方向几乎不会提升单轮吞吐；如果评测请求有系统 prompt 或重复模板，则可能显著减少 prefill。

因此必须先统计：

- 请求前缀重复率；
- 可复用 token 数；
- cache 命中率；
- cache 管理开销。

没有前缀复用证据时，不建议投入核心开发资源。

## 十一、方向 9：vAttention 与连续 KV 内存

[vAttention](https://arxiv.org/abs/2405.04437) 探索不依赖 PagedAttention 的动态显存管理，减少页表和 block table 间接访问。

对当前单卡比赛环境，只有在 profile 证明以下问题严重时才值得尝试：

- block table 查找占用明显；
- KV cache block 太小导致访存不连续；
- paged layout 造成 L2 命中率显著下降。

否则重写 memory manager 的投入大于收益。更现实的实验是先只调 `block_size=16/32/64`，测量 attention kernel 的访存效率。

## 十二、推荐的创新组合

### 稳妥版本

```text
GQA head packing
+ 自适应 split-KV
+ RMSNorm/QKV/RoPE/Cache 融合
+ Gate/Up + SwiGLU epilogue
+ Top-P Gumbel sampling
```

### 冲刺版本

```text
稳妥版本
+ INT8 KV cache
+ persistent decode / graph 覆盖
+ async scheduling
```

### 研究展示版本

```text
动态稀疏 prefill
+ Prefix Cache/RadixAttention
+ 投机解码
```

研究展示版本创新性较强，但不应作为比赛稳定成绩的唯一依赖。

## 十三、建议的验证顺序

1. 先 profile Attention 的真实 HBM 流量和 GQA KV 重复读取比例；
2. 实现 GQA head packing，完成非 paged 与 paged 两套 microbenchmark；
3. 验证自适应 split-KV 在 4K/16K 的最优表；
4. 再做 QKV/RoPE/Cache 和 SwiGLU 融合；
5. 检查 async scheduling 和 graph 覆盖率；
6. 精度稳定后再试 INT8 KV cache；
7. 最后评估动态稀疏和投机解码。

## 十四、参考资料

- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)
- [Flash-Decoding++](https://arxiv.org/abs/2311.01282)
- [Lean Attention](https://arxiv.org/abs/2405.10480)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [FlashInfer](https://arxiv.org/abs/2501.01005)
- [vAttention](https://arxiv.org/abs/2405.04437)
- [KIVI](https://arxiv.org/abs/2402.02750)
- [BitDecoding](https://arxiv.org/abs/2503.18773)
- [MInference 1.0](https://arxiv.org/abs/2407.02490)
- [Sarathi-Serve](https://arxiv.org/abs/2403.02310)
- [Speculative Decoding](https://arxiv.org/abs/2203.16487)
- [Medusa](https://arxiv.org/abs/2401.10774)
- [EAGLE](https://arxiv.org/abs/2401.15077)
- [SGLang](https://arxiv.org/abs/2312.07104)
- [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
