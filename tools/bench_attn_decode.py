#!/usr/bin/env python3
"""Standalone paged-KV decode attention microbench on MetaX C500.

Roofline per layer: KV bytes = B * ctx * 2(KV) * 2(kv_heads) * 128 * 2B.
If the vendor kernel shares KV across the 8 Q-heads of a GQA group,
measured time should approach bytes/HBM_BW; per-Q-head reads would cost 8x.
"""
import torch
from flash_attn import flash_attn_with_kvcache

dev = "cuda"
B, HQ, HKV, D = 64, 16, 2, 128
BLOCK = 16

def run_case(ctx, iters=100):
    blocks_per_seq = (ctx + BLOCK - 1) // BLOCK
    num_blocks = B * blocks_per_seq + 8
    k_cache = torch.randn(num_blocks, BLOCK, HKV, D, device=dev, dtype=torch.bfloat16)
    v_cache = torch.randn(num_blocks, BLOCK, HKV, D, device=dev, dtype=torch.bfloat16)
    q = torch.randn(B, 1, HQ, D, device=dev, dtype=torch.bfloat16)
    bt = torch.arange(num_blocks - 8, device=dev, dtype=torch.int32)
    bt = bt[torch.randperm(num_blocks - 8, device=dev)]
    block_table = bt[: B * blocks_per_seq].reshape(B, blocks_per_seq)
    seqlens = torch.full((B,), ctx, device=dev, dtype=torch.int32)
    scale = D ** -0.5

    fn = lambda: flash_attn_with_kvcache(
        q=q, k_cache=k_cache, v_cache=v_cache, block_table=block_table,
        cache_seqlens=seqlens, softmax_scale=scale, causal=True,
    )
    out = fn()
    assert out.shape == (B, 1, HQ, D), out.shape
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    t = s.elapsed_time(e) / iters * 1e3
    kv_bytes = B * ctx * 2 * HKV * D * 2
    roofline = kv_bytes / 1.6e12 * 1e6
    print(f"ctx={ctx:<6} {t:8.1f} us/layer   KV={kv_bytes/1e6:.0f}MB  "
          f"roofline@1.6TB/s={roofline:.0f}us  efficiency={roofline/t*100:.0f}%")

for ctx in (4096, 16384, 32768):
    run_case(ctx)
