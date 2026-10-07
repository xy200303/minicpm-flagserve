#!/usr/bin/env python3
"""Sweep metax flash-attn DECODE (flash_attn_with_kvcache) kernel choices.

Serve decode shape: batch=64, q=[64,1,16,128] bf16, paged KV, causal,
kv_len ~ context + generated.  Default runs with num_splits=0 (heuristic)
on whatever traits the scheduler picks (trace: 128,16,16,1).

Sweep: splitkv kernel candidates x alg x num_splits, per kv-length bucket.
"""
import torch
import flash_attn_2_cuda as fa2
from flash_attn.flash_attn_interface import flash_attn_with_kvcache

dev = "cuda"
HQ, HKV, HD, BLK = 16, 2, 128, 16
BATCH = 64

SPLITKV_KERNELS = [
    "fwd_split_hdimqk_128_hdimv_128_blockm_128_blockn_64_bfloat16_4_True_True",
    "fwd_split_hdimqk_128_hdimv_128_blockm_64_blockn_64_bfloat16_4_True_True",
    "fwd_split_hdimqk_128_hdimv_128_blockm_64_blockn_32_bfloat16_4_True_True",
    "fwd_split_hdimqk_128_hdimv_128_blockm_32_blockn_32_bfloat16_2_True_True",
    "fwd_split_hdimqk_128_hdimv_128_blockm_16_blockn_16_bfloat16_1_True_True",
]
SPLITS = [0, 1, 2, 4, 8, 16, 32]


def make_problem(kv_len):
    blocks_per_seq = (kv_len + BLK - 1) // BLK
    nblocks = BATCH * blocks_per_seq + 8
    kc = torch.randn(nblocks, BLK, HKV, HD, device=dev, dtype=torch.bfloat16)
    vc = torch.randn(nblocks, BLK, HKV, HD, device=dev, dtype=torch.bfloat16)
    q = torch.randn(BATCH, 1, HQ, HD, device=dev, dtype=torch.bfloat16)
    bt = torch.arange(1, BATCH * blocks_per_seq + 1, device=dev,
                      dtype=torch.int32).view(BATCH, blocks_per_seq)
    seqlens = torch.full((BATCH,), kv_len, device=dev, dtype=torch.int32)
    return q, kc, vc, bt, seqlens


def run_once(prob, num_splits=0):
    q, kc, vc, bt, seqlens = prob
    return flash_attn_with_kvcache(
        q, kc, vc, cache_seqlens=seqlens, block_table=bt,
        softmax_scale=HD ** -0.5, causal=True, num_splits=num_splits,
    )


def timeit(fn, iters=50):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3


BUCKETS = [2048, 4096, 8192, 16384]

print("=== DEFAULT (num_splits=0 heuristic, default traits) ===")
defaults = {}
for kv in BUCKETS:
    prob = make_problem(kv)
    t = timeit(lambda: run_once(prob))
    defaults[kv] = t
    print(f"kv={kv:<6} default {t:8.1f} us", flush=True)

print("=== sweep: num_splits via API (default traits) ===")
best = {kv: (defaults[kv], "default") for kv in BUCKETS}
for ns in SPLITS[1:]:
    row = []
    for kv in BUCKETS:
        prob = make_problem(kv)
        t = timeit(lambda: run_once(prob, num_splits=ns))
        row.append(t)
        if t < best[kv][0]:
            best[kv] = (t, f"default-traits ns{ns}")
    print(f"ns{ns:<3} " + " ".join(f"{t:7.1f}" for t in row), flush=True)

print("=== sweep: ks_set_solution(kernel, alg, splits) ===")
for kid in SPLITKV_KERNELS:
    for alg in (0, 2, 4):
        for ns in (1, 4, 8, 16):
            fa2.ks_set_solution(kid, 1, ns, alg)
            row = []
            for kv in BUCKETS:
                prob = make_problem(kv)
                try:
                    t = timeit(lambda: run_once(prob), iters=30)
                except Exception:
                    t = float("inf")
                row.append(t)
                if t < best[kv][0]:
                    best[kv] = (t, f"{kid} alg{alg} ns{ns}")
            print(f"{kid.split('blockm_')[1]:<36} alg{alg} ns{ns:<3} " +
                  " ".join(f"{t:7.1f}" for t in row), flush=True)

print("=== BEST per kv bucket ===")
for kv in BUCKETS:
    t, cfg = best[kv]
    d = defaults[kv]
    print(f"kv={kv:<6} default {d:8.1f}us -> best {t:8.1f}us ({d/t:4.2f}x)  {cfg}")
