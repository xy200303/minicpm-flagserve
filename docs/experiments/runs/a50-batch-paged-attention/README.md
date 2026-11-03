# A50：厂商 batch paged attention 探针（2026-11-04，停止）

当前 prefill 路径先用 dual-gather 将页式 KV 连续化，再调用 `flash_attn_varlen_func`。厂商库另外导出 `batch_paged_attention`，接受 `q_index/kv_index/block_tables`，因此调查能否省去 KV 连续化拷贝。该接口此前未用于 A6/A14/A20，不是重复调参。

## 已确认的接口约束

- KV 使用三维 `[total_entries, kv_heads, head_size]`，可由 vLLM 页缓存重塑，无需全局布局转换。
- `max_page_num × page_size` 必须等于整个 KV 缓存第一维。
- LSE 必须是二维张量或空张量。

## 结果

按上述约束构造单请求、非连续页的小形状，并与 bottom-right causal 的 FP32 参考实现准备对拍。内核在数值输出前返回无详情的 `RuntimeError`。以下组合均失败：

- `q_len=4` 与 `q_len=1`；
- BF16 与 FP16；
- causal 开启与关闭；
- 未使用页表项填 `-1` 与合法页号；
- 二维 LSE 与空 LSE。

因此当前镜像无法取得有效数值结果，不进行性能计时、不修改服务路径、不执行官方 benchmark。失败不是吞吐负收益结论，而是厂商接口可用性阻塞。完整错误见 `probe.log`，探针和机器版本见 `probe.py`、`machine-state.txt`。回滚无需操作：交付分支未改动。
