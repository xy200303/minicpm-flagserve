#!/usr/bin/env python3
"""Sweep metax flash-attn fwd kernels/alg/splits at serve 16k-prefill shapes.

Prefill call in serve: flash_attn_varlen_func, paged KV, batch=1 (one 2048
chunk per step), q=[2048,16,128] bf16, kv_len = 2048..16384, causal.

For each (kernel_id, alg, num_splits) we force the choice via
flash_attn_2_cuda.ks_set_solution (the vendor tuner's own hook), then time.
Default heuristic is measured first, before any override is installed.
"""
import torch
import flash_attn_2_cuda as fa2
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward

dev = "cuda"
HQ, HKV, HD, BS = 16, 2, 128, 16  # q heads, kv heads, head dim, kv block size
Q_LEN = 2048
SM = torch.cuda.get_device_properties(0).multi_processor_count if hasattr(torch.cuda, "get_device_properties") else -1
print("SM count:", SM)

SPLITKV_KERNELS = [
    "fwd_split_hdimqk_128_hdimv_128_blockm_64_blockn_32_bfloat16_4_True_True",
    "fwd_split_hdimqk_128_hdimv_128_blockm_64_blockn_64_bfloat16_4_True_True",
    "fwd_split_hdimqk_128_hdimv_128_blockm_128_blockn_64_bfloat16_4_True_True",
    "fwd_split_hdimqk_128_hdimv_128_blockm_32_blockn_32_bfloat16_2_True_True",
]
FWD_KERNELS = [
    "fwd_hdimqk_128_hdimv_128_blockm_128_blockn_64_bfloat16_4_True_True",
    "fwd_hdimqk_128_hdimv_128_blockm_128_blockn_128_bfloat16_4_True_True",
    "fwd_hdimqk_128_hdimv_128_blockm_64_blockn_64_bfloat16_4_True_True",
]


def make_problem(kv_len, batch=1):
    nblocks = batch * ((kv_len + BS - 1) // BS) + 8
    k_cache = torch.randn(nblocks, BS, HKV, HD, device=dev, dtype=torch.bfloat16)
    v_cache = torch.randn(nblocks, BS, HKV, HD, device=dev, dtype=torch.bfloat16)
    q = torch.randn(batch * Q_LEN, HQ, HD, device=dev, dtype=torch.bfloat16)
    cuq = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * Q_LEN
    cuk = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * kv_len
    bt = torch.arange(1, batch * ((kv_len + BS - 1) // BS) + 1, device=dev,
                      dtype=torch.int32).view(batch, -1)
    return q, k_cache, v_cache, cuq, cuk, bt


def run_once(prob):
    q, k, v, cuq, cuk, bt = prob
    return _flash_attn_varlen_forward(
        q, k, v, cuq, cuk, Q_LEN, cuk[-1].item(), 0.0, HD ** -0.5,
        True, (-1, -1), 0.0, None, False, block_table=bt,
    )


def timeit(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3


BUCKETS = [2048, 4096, 8192, 12288, 16384]
SPLITS = [1, 2, 4, 8, 16]

print("=== DEFAULT dispatch (no override) ===")
defaults = {}
for kv in BUCKETS:
    prob = make_problem(kv)
    t = timeit(lambda: run_once(prob))
    defaults[kv] = t
    print(f"kv={kv:<6} default {t:8.1f} us", flush=True)

print("=== sweep splitkv kernels ===")
best = {kv: (defaults[kv], "default") for kv in BUCKETS}
for kid in SPLITKV_KERNELS:
    for alg in range(6):
        for ns in SPLITS:
            fa2.ks_set_solution(kid, 1, ns, alg)
            row = []
            for kv in BUCKETS:
                prob = make_problem(kv)
                try:
                    t = timeit(lambda: run_once(prob), iters=15)
                except Exception:
                    t = float("inf")
                row.append(t)
                if t < best[kv][0]:
                    best[kv] = (t, f"{kid} alg{alg} ns{ns}")
            print(f"{kid.split('blockm_')[1]:<28} alg{alg} ns{ns:<3} " +
                  " ".join(f"{t:7.1f}" for t in row), flush=True)

print("=== sweep plain fwd kernels (ns=1) ===")
for kid in FWD_KERNELS:
    for alg in range(0, 12, 3):
        fa2.ks_set_solution(kid, 0, 1, alg)
        row = []
        for kv in BUCKETS:
            prob = make_problem(kv)
            try:
                t = timeit(lambda: run_once(prob), iters=15)
            except Exception:
                t = float("inf")
            row.append(t)
            if t < best[kv][0]:
                best[kv] = (t, f"{kid} alg{alg} ns1")
        print(f"{kid.split('hdimv_128_')[1]:<40} alg{alg:<2} " +
              " ".join(f"{t:7.1f}" for t in row), flush=True)

print("=== BEST per kv bucket (vs default) ===")
for kv in BUCKETS:
    t, cfg = best[kv]
    d = defaults[kv]
    print(f"kv={kv:<6} default {d:8.1f}us -> best {t:8.1f}us ({d/t:4.2f}x)  {cfg}")
