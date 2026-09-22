#!/usr/bin/env python3
"""Focused probe: big BLOCK_K (fewer K iters) for qkv shape on mm_kernel_nt."""
import sys
import torch
import triton

sys.path.insert(0, "/workspace/FlagGems/src")
import flag_gems
from flag_gems import runtime

dev = "cuda"
M, K, N = 64, 2048, 2560

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

cands = []
for BK in (256, 512, 1024, 2048):
    for BN in (32, 64, 128):
        for ns in (1, 2, 3):
            for nw in (4, 8):
                for pl in ("basic", "cpasync"):
                    cands.append(triton.Config(
                        {"BLOCK_M": 64, "BLOCK_N": BN, "BLOCK_K": BK,
                         "GROUP_M": 8, "pipeline": pl, "scenario": ""},
                        num_stages=ns, num_warps=nw))
print(f"candidates: {len(cands)}", flush=True)

orig_cfg = runtime.get_tuned_config
runtime.get_tuned_config = lambda op, **kw: cands if op == "mm_nt" else orig_cfg(op, **kw)
flag_gems.enable(record=False)

a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
t = timeit(lambda: torch.mm(a, w.t()), 100)
bw = K * N * 2 / (t * 1e-6) / 1e9
print(f"qkv best-of-{len(cands)}: {t:.1f} us ({bw:.0f} GB/s)")

# M=1024 scaling check: is the kernel good at big M (i.e. problem only at skinny M)?
runtime.get_tuned_config = orig_cfg
a2 = torch.randn(1024, K, device=dev, dtype=torch.bfloat16)
t2 = timeit(lambda: torch.mm(a2, w.t()), 100)
fl = 2 * 1024 * N * K / (t2 * 1e-6) / 1e12
print(f"M=1024 qkv: {t2:.1f} us ({fl:.1f} TFLOPS)  [M=64 was ~110us]")
