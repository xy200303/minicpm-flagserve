# A25: `_build_split` 每步 81 ms 的真凶与它的真实价值

日期: 2026-10-22  卡: 天数 BI-V150 32GB（开发机）  栈: vLLM 0.24 + vllm-plugin-FL(flagos-2026-s2) b0cda5e（A20 交付态）

## 1. 背景

A13（2026-10-17）在官方 4k c64 剖面里测到 prefill 步 ≈200 ms，其中
`_build_attention_metadata` 占 74–82 ms，并归因为
"`common_attn_metadata.query_start_loc_cpu.tolist()` 每步触发一次 D2H 同步"。
当时实现了"runner 侧发布 CPU 侧 cu_l/seq_l"的补丁，但同日基线复测只有 2346 tok/s
（历史 3778），被判定为机器漂移，补丁回退、结论悬置。

## 2. 探针

在 `_build_patched` 与 `_build_split` 内插入分段计时（`VLLM_FL_BUILD_DBG=1`）：

```python
[BUILD] orig=0.0 cu_tolist=0.0 seq_tolist=0.0 split=81.8 | nreq=17 maxq=1374 cu=Tensor/cpu seq=Tensor/cpu
[SPLIT] total=82.3 | py_split=0.0 h2d_idx_t=0.1 h2d_dec_rows=0.1 h2d_seq_dec=0.0 \
        arange=0.1 block_table_sel=0.5 dec_meta=0.0 cu_qk=81.4 | npre=2 ndec=15
```

**A13 的归因是错的**：两处 `.tolist()` 实测 0.0 ms（本来就是 CPU tensor）。
81 ms 全部落在 `_build_split` 的最后两行。

## 3. 真凶

```python
dev32 = dict(dtype=torch.int32, device=dev)
m._fl_cu_q = torch.tensor([[0, cu_l[i+1]-cu_l[i]] for i in pre_idx], **dev32)
m._fl_cu_k = torch.tensor([[0, seq_l[i]] for i in pre_idx], **dev32)
```

`torch.tensor(python_list, device=cuda)` 在 BI-V150 上约 **40 ms/次**（2 次 = 81 ms），
而同一份数据"先在 CPU 建、再 `.to(dev)`"只要 0.1 ms——同函数里上面的 `idx_t` 就是这么写的。

## 4. 修复（A25）

```python
m._fl_cu_q = torch.tensor([[0, ...] for i in pre_idx], dtype=torch.int32).to(dev, non_blocking=True)
m._fl_cu_k = torch.tensor([[0, ...] for i in pre_idx], dtype=torch.int32).to(dev, non_blocking=True)
```

验证：

```
[SPLIT] total=30.1 | ... cu_qk=0.3 | npre=1 ndec=1
```

`cu_qk` 81.4 → **0.3 ms**。（同一行里 `block_table_sel` 首次 27.8 ms、稳态 0.5 ms。）

## 5. 端到端：不达门槛

官方 `[4096,1024,64,256]`，同日、同机、单次对照：

| | 基线 b0cda5e | A25 | 变化 |
|---|---|---|---|
| Total token throughput | 2571.12 tok/s | **2592.87 tok/s** | **+0.85%** |
| Mean TTFT | 5529.69 ms | 5507.54 ms | −0.4% |
| Mean TPOT | 96.58 ms | 95.79 ms | −0.8% |
| Successful requests | 256/256 | 256/256 | — |

**+0.85% 落在噪声内，不满足"当日交错 A/B 提升 ≥3%"的准入标准。**

## 6. 为什么省了 81 ms 却几乎不提速

因为那 81 ms 是**等待**，不是 CPU 浪费。

`torch.tensor(list, device=cuda)` 在这套后端的实现需要同步（数据要从主机逐元素搬到显存），
它把"GPU 上尚未完成的前向"一并等掉——于是 81 ms 被记在了这一行。
换成 `.to(dev, non_blocking=True)` 之后 CPU 不再阻塞，但**GPU 上该做的工作一点没少**，
时间只是转移到了流水线的其它位置。

这与 A23 在投机路径上看到的机制同源：`aten::index` 与这次的两行一样，都是**隐式同步点**。
差别在于数量：投机混合步每步有 4–6 个这样的同步点，串行化把 8 ms 的步拖成 1094 ms；
基线每步只有 1 个，且 GPU 本来就是瓶颈，所以去掉之后端到端几乎不变。

## 7. 结论与处置

- **不进提交栈**：未通过 3% 准入，且未做 ≥2 次交错 A/B。
- 留档为"对 A13 的第二次修正"：prefill metadata 的 74–82 ms 不是 D2H，是
  CPU-list→CUDA-tensor 的隐式同步；这一条应写入技术报告的"缺陷分析"。
- 补丁可随时重放：`docs/experiments/a25/a25.patch`（含上面所有计时埋点，
  埋点只在 `VLLM_FL_BUILD_DBG=1` 时生效）。

## 8. 复现

```bash
# 基线栈 + 计时埋点
VLLM_PLUGINS=fl VLLM_FL_BUILD_DBG=1 vllm serve /workspace/MiniCPM5-2B --port 9031 \
  --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
# 官方 4k
vllm bench serve --backend vllm --model minicpm --tokenizer /workspace/MiniCPM5-2B \
  --endpoint /v1/completions --host localhost --port 9031 --dataset-name random \
  --ignore-eos --random-input-len 4096 --random-output-len 1024 \
  --max-concurrency 64 --num-prompts 256
```

证据：`docs/experiments/a25/`（`split_before_a25.log`、`split_after_a25.log`、
`build_before_a25.log`、两份 4k result、`a25_throughput.log`、`a25.patch`）。
