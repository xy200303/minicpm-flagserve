#!/usr/bin/env python3
"""Probe: nn vs nt layout for large-M GEMM on metax FlagGems."""
import sys
import torch

dev = "cuda"
M, K, N = 2048, 2048, 2560

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

a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
w_nt = torch.randn(N, K, device=dev, dtype=torch.bfloat16)   # [N,K] -> nt path
w_nn = w_nt.t().contiguous()                                  # [K,N] -> nn path
t = timeit(lambda: torch.mm(a, w_nt.t()))
print(f"vendor nt  {t:7.1f} us  {2*M*N*K/(t*1e-6)/1e12:6.1f} TFLOPS", flush=True)
t = timeit(lambda: torch.mm(a, w_nn))
print(f"vendor nn  {t:7.1f} us  {2*M*N*K/(t*1e-6)/1e12:6.1f} TFLOPS", flush=True)

sys.path.insert(0, "/workspace/FlagGems/src")
import flag_gems
flag_gems.enable(record=False)
t = timeit(lambda: torch.mm(a, w_nt.t()))
print(f"gems   nt  {t:7.1f} us  {2*M*N*K/(t*1e-6)/1e12:6.1f} TFLOPS", flush=True)
t = timeit(lambda: torch.mm(a, w_nn))
print(f"gems   nn  {t:7.1f} us  {2*M*N*K/(t*1e-6)/1e12:6.1f} TFLOPS", flush=True)
print("done")
