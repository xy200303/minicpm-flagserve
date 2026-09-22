#!/usr/bin/env python3
"""Probe mm floor: vendor torch.mm vs FlagGems mm_kernel_nt config sweep."""
import sys
import torch
import triton

sys.path.insert(0, "/workspace/FlagGems/src")

dev = "cuda"
M = 64
SHAPES = [("qkv", 2048, 2560), ("gate_up", 2048, 12288), ("down", 6144, 2048)]

def timeit(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3

print("=== vendor torch.mm (no flag_gems override) ===")
for name, K, N in SHAPES:
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    t = timeit(lambda: torch.mm(a, w.t()))
    bw = K * N * 2 / (t * 1e-6) / 1e9
    print(f"{name:<9} {t:8.1f} us  weightBW={bw:7.1f} GB/s")

print("\n=== FlagGems mm_kernel_nt config sweep ===")
from flag_gems.runtime.backend._metax.ops.mm import mm_kernel_nt

for name, K, N in SHAPES:
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    c = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    best = (1e9, None)
    for BM in (16, 32, 64):
        for BN in (32, 64, 128, 256):
            for BK in (64, 128, 256):
                for ns in (2, 3, 4, 5):
                    for nw in (4, 8):
                        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
                        def run():
                            mm_kernel_nt[grid](
                                a, w.t().contiguous().t(), c, M, N, K,
                                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                                GROUP_M=8, EVEN_M=True, EVEN_N=(N % BN == 0),
                                EVEN_K=(K % BK == 0),
                                num_stages=ns, num_warps=nw,
                            )
                        try:
                            t = timeit(run, 50)
                        except Exception:
                            continue
                        if t < best[0]:
                            best = (t, (BM, BN, BK, ns, nw))
    bw = K * N * 2 / (best[0] * 1e-6) / 1e9
    print(f"{name:<9} best={best[0]:8.1f} us  cfg(BM,BN,BK,stages,warps)={best[1]}  weightBW={bw:7.1f} GB/s")
