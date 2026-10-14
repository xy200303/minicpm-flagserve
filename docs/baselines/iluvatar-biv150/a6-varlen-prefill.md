# 天数 BI-V150 A6：varlen vendor flash prefill + Triton dual-gather

日期：2026-10-14 ｜ 实例：FlagOS OnlineLab BI-V150（32GB 显存版）

## 起因

A5 之后 serve timing 日志（干净数据段）显示 16k 混合步 wall
~337ms，其中 vendor FA host 占 ~198ms/步（59%）：dec 段
2.95ms/call ×42 层，pre 段 1.77ms/call ×42 层。pre 段每请求
**4 次 kernel 发射**：2×index_select（K、V 分页收集）+ 1×aten
`_efficient_attention_forward` + 1×copy-out 回写输出。host 发射
开销远大于 GPU 实算，发射数可直接砍半。

## 探针验证（改造前先做，全部 bit-exact）

1. `ixformer.flash_attn_varlen_func` vs aten
   `_efficient_attention_forward`(cmt=2 bottom-right)：
   16k/4k/方阵/ragged/单 q/多请求 varlen 全形状 max_err 全
   0.0000；支持 `out=` 直写输出切片、varlen 多请求、GQA。
   签名 `(q[total_q,H,D], k[total_k,HKV,D], v, cu_seqlens_q
   int32, cu_seqlens_k int32, max_q, max_k, dropout_p,
   softmax_scale, causal, out)`，`causal=True` 即 bottom-right
   语义。aten 侧**没有 out 变体**（查过 schema），varlen 是
   消掉 copy-out 的唯一出路。
2. Triton `_dual_gather_kernel`：1 次发射完成 K+V 两页收集
   （kv_cache `[blocks,2,bs,hkv,d]` 页内 K/V 连续，每 program
   处理一页），与 2×index_select **bit-exact**，GPU 106→62µs、
   host 68→37µs，num_warps=4 最优（w8 反而更差）。

## 改造内容（vllm-plugin-FL commit 20d10aa）

`iluvatar/impl/vendor_flash_attn.py`：

1. builder split 新增 `_fl_cu_q`/`_fl_cu_k`（[n_pre,2] int32
   GPU 张量，rebase 到 0，每步 2 个小 HtoD，全 42 层共享）
2. pre 段重写为 varlen 分支：1×dual-gather（无 Triton 时兜底
   2×index_select 写同一池 `KVp[2, rows, hkv, d]`，K/V 各自
   连续）+ 1×`flash_attn_varlen_func(out=out[q0:q1])` 直写
   输出 = **2 次发射/层/请求**（原 4 次）
3. A/B 开关：`VLLM_FL_DISABLE_DUAL_GATHER=1`、
   `VLLM_FL_DISABLE_VARLEN_PREFILL=1`（连同原有
   `VLLM_FL_DISABLE_ILUVATAR_VENDOR_FA=1`）

**坑**：合并 gather（`index_select(kv_cache,0,pages)` 输出
`[n,2,bs,hkv,d]`）后 `sel[:,0].view(-1,hkv,d)` 因 dim0 stride
跳间隔报 "view size is not compatible"——两 gather 必须写同一
个池的两个独立连续区，不能指望 view 免费切分。

## 效果

timing 日志 pre 段 host：**1.77ms/call → ~0.13ms/call**（混合
负载均值，负载已排除 prefix-cache 干扰——自测用相同 prompt
会被 vLLM 前缀缓存全部命中，需逐请求不同前缀）。

端到端（官方 benchmark，4 跑弃首取后 3 均值）：

| 场景 | A5 | A6 | Δ |
|---|---|---|---|
| 4k total tok/s | 2587.84 | **2738.65** | **+5.83%** |
| 16k total tok/s | 2532.68 | **2558.37** | **+1.01%** |

16k 三次有效跑 2558.17/2558.42/2558.52，组内极差 0.01%。
TTFT 双场景均改善：4k mean 4736→4403ms、p99 27598→26150ms；
16k mean 188784→186421ms、p99 400562→396174ms。4k 提升大于
16k 符合预期：4k prefill 占比更高，host 发射削减收益更大；
16k 已高度受 KV 池排队限制（32GB 卡）。

精度实测 **98.1%**（MATH-500 L3，105 题，evalscope 同口径），
≥0.95 达标。两条新路径均 bit-exact，与 A3-A5 的 97.1% 之差
在评测噪声内。

## 后续候选

- dec 段 2.95ms/call 的构成：q gather(~0.5) + empty_like +
  stock `_orig_forward` 内部(~1.9ms，最大头) + index_copy_
  回写(~0.5)。可绕开 stock TritonAttentionImpl.forward 的
  Python 壳直调 decode 内核，但 A2 已证 decode 内核本身无
  提升空间，只能省 host 壳子，风险高于收益
- 16k host 侧 ~19% 调度空洞（vLLM scheduler，难动）
