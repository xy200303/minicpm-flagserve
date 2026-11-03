# A51：MTP paged prefill attention（2026-11-04，停止）

`vllm_paged_attention_mtp` 支持多查询、GQA、causal 和 `cu_query_lens`，本是消除 prefill dual-gather 的另一条厂商接口路线，与失败的 A50 batch 接口不同。

接口要求 KV 的最后一维 stride 为 1，且 `stride(-2)=head_size`，即页内采用 `[kv_heads, block_size, head_size]`。当前 vLLM/A20 使用 `[block_size, kv_heads, head_size]`；零拷贝 `permute(0,2,1,3)` 后最后一维满足要求，但 `stride(-2)=1`，调用在计算前被明确拒绝。

全局改为 HND 会影响已调优的 A20 decode 和 KV 写入路径，而额外转置又会重新引入要消除的拷贝。dual-gather 在当前 vendor varlen 总路径中占比有限，无法支撑这项高风险布局改造的预期收益。因此不做性能计时、不修改服务路径、不执行官方 benchmark。原始错误和版本见 `probe.log`、`probe.py`、`machine-state.txt`；交付分支无改动。
