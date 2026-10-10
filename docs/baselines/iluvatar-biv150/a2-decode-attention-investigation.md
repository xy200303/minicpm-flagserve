# 天数 A2 调研：decode attention（结论：维持 stock，负面结果归档）

日期：2026-10-10 ｜ 实例：BI-V150 32GB

## 三个方向，逐一排除

### 1. vendor op 做 decode（aten::_efficient_attention_forward）

q=1 decode 形状（B=64, kv=16384, GQA 16:2）实测 **15.7ms/层**
（0.07 TB/s）——ixAttnBkd 没有 flash-decoding 分段路径，q=1 时完全
不并行。死路。

### 2. GEMM 换 vendor cuBLAS（复刻 C500 opt8）

M=2048 四个 prefill 形状上 vendor ≈ FlagGems Triton（±3%，噪声内）。
天数的 FlagGems GEMM 已是 vendor 水平，无收益。死路。

### 3. 自研 Triton flash-decoding 内核（flash_dec_iluvatar.py）

起因：stock unified_attention（3D split-16）实测只有 0.17 TB/s，
而卡的大拷贝带宽 0.59 TB/s，看似有 3 倍空间。

自研内核：split-KV + online softmax + paged gather + GQA
（tl.dot bf16 走 TCU；双 kv-head 变体每 token 读 512B 连续行）。
踩坑记录：reduce 内核 M/L 用了 acc 的 stride（stride 混用 bug）；
空分段必须写中性值（-inf/0/0）；fp32 dot 无 TCU（换 bf16 提速 25%）；
tl.split 需要最后一维为 2（reshape+trans 调整）。

全空间扫描（splits × BLOCK_N × warps × stages × 双头）后：

| 配置 | 16k 耗时 | 说明 |
|---|---|---|
| stock（serve 实际配置） | 4812µs | 0.17 TB/s |
| 自研最优 (8,32,4,1,dual) | 5058µs | 0.16 TB/s（**0.95x，打平**） |
| stages=2 | 28674µs | 流水线对这卡的 gather 有害 |
| warps=8 | 14081µs | warp64 架构下 8 warp 有害 |

正确性两者一致（err ~3e-4）。打平原因：带宽上限是按大块连续读测的
（index_select 纯 gather 能到 0.56 TB/s），paged 小行 gather +
online-softmax 循环依赖把两类内核都压在 ~0.16 TB/s——**瓶颈在访存
模式不在内核调度**，Triton 层面无解。

最终判定：decode attention 维持 stock。16k decode 侧进一步提速需要
芯片级手段（如 cpasync 风格的预取，flagtree 不支持），不投入。

## 数据位置

内核与测试：`tools/flash_dec_iluvatar.py`（正确可用，打平 stock，
其它卡上可能有价值）、`tools/test_flash_dec.py`。
