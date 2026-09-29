#!/usr/bin/env python3
"""Crossover probe: nt_db (skinny) vs nn (repacked) across M."""
import sys
import torch

sys.path.insert(0, "/workspace/FlagGems/src")
import flag_gems
import importlib
mxmm = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")

dev = "cuda"

def timeit(fn, iters=100):
    for _ in range(15):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3

flag_gems.enable(record=False)
for name, K, N in [("qkv", 2048, 2560), ("gate_up", 2048, 12288), ("down", 6144, 2048)]:
    for M in (64, 128, 256, 512, 2048):
        a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        wt = w.t().contiguous()
        mxmm._NT_DB_MAX_M = 10**9  # force nt_db for any M
        t_ntdb = timeit(lambda: torch.mm(a, w.t()))
        mxmm._NT_DB_MAX_M = 0      # disable nt_db -> nn path with wt
        t_nn = timeit(lambda: torch.mm(a, wt))
        print(f"{name:<8} M={M:<5} nt_db={t_ntdb:7.1f} us   nn={t_nn:7.1f} us", flush=True)
print("done")
