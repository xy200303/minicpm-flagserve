#!/usr/bin/env python3
"""A/B: nt_db kernel (manual double-buffer + grouped raster) at large M."""
import sys
import torch

dev = "cuda"
K, N = 2048, 2560

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
    print(f"vendor  M={M:<5} {t:7.1f} us  {2*M*N*K/(t*1e-6)/1e12:6.1f} TFLOPS", flush=True)

sys.path.insert(0, "/workspace/FlagGems/src")
import flag_gems
import importlib
mxmm = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")
mxmm._NT_DB_MAX_M = 10**9  # test-only: widen scenario
flag_gems.enable(record=False)

for M in (512, 1024, 2048, 4096, 8192):
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    c = torch.mm(a, w.t())
    ref = a.float() @ w.t().float()
    rel = ((c.float() - ref).abs().max() / ref.abs().max()).item()
    t = timeit(lambda: torch.mm(a, w.t()))
    print(f"nt_db   M={M:<5} {t:7.1f} us  {2*M*N*K/(t*1e-6)/1e12:6.1f} TFLOPS  relerr={rel:.5f}", flush=True)
print("done")
