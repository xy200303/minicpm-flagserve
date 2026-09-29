#!/usr/bin/env python3
"""Large-M (prefill) GEMM: vendor torch.mm vs FlagGems mm on C500."""
import sys
import torch

dev = "cuda"
K, N = 2048, 2560  # qkv shape; also try mlp-like

def timeit(fn, iters=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3

for M in (512, 1024, 2048, 4096, 8192):
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    t = timeit(lambda: torch.mm(a, w.t()))
    fl = 2 * M * N * K / (t * 1e-6) / 1e12
    print(f"vendor  M={M:<5} {t:8.1f} us  {fl:6.1f} TFLOPS")

sys.path.insert(0, "/workspace/FlagGems/src")
import flag_gems
flag_gems.enable(record=False)
for M in (512, 1024, 2048, 4096, 8192):
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    t = timeit(lambda: torch.mm(a, w.t()))
    fl = 2 * M * N * K / (t * 1e-6) / 1e12
    print(f"flaggems M={M:<5} {t:8.1f} us  {fl:6.1f} TFLOPS")
