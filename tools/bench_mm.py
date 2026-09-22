#!/usr/bin/env python3
"""Microbench FlagGems metax mm at MiniCPM5-2B decode shapes (M=64)."""
import os
import sys
import torch

sys.path.insert(0, "/workspace/FlagGems/src")
import flag_gems

flag_gems.enable(record=False)
dev = "cuda"

# (name, K, N): A[M=64,K] @ W[N,K]^T (mm_nt layout)
SHAPES = [
    ("qkv",     2048, 2560),
    ("o_proj",  2048, 2048),
    ("gate_up", 2048, 12288),
    ("down",    6144, 2048),
    ("lm_head", 2048, 130560),
]
M = 64

def bench(a, b, iters=200):
    for _ in range(20):
        torch.mm(a, b)
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        torch.mm(a, b)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us

total_t = 0.0
print(f"flagtune={'ON' if os.environ.get('USE_FLAGTUNE') else 'off'}")
for name, K, N in SHAPES:
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    b = w.t()  # [K, N] view, column-major = mm_nt
    t = bench(a, b)
    total_t += t
    bw = (K * N * 2) / (t * 1e-6) / 1e9  # GB/s counting weight traffic only
    flops = 2 * M * N * K / (t * 1e-6) / 1e12
    print(f"{name:<9} K={K:<6} N={N:<7} {t:8.1f} us  weightBW={bw:7.1f} GB/s  {flops:6.2f} TFLOPS")
print(f"TOTAL per step (1x each shape; model has 42x first four): {total_t:.1f} us")
est = 42 * sum(bench(torch.randn(M, K, device=dev, dtype=torch.bfloat16),
                     torch.randn(N, K, device=dev, dtype=torch.bfloat16).t(), 50)
               for _, K, N in SHAPES[:4])
print(f"EST model GEMM time/step (42 layers, excl lm_head): {est/1000:.2f} ms")
