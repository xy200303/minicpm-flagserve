#!/usr/bin/env python3
"""Large-M sweep: FlagGems tuned space vs our nt_db kernel vs vendor."""
import sys
import torch
import triton

sys.path.insert(0, "/workspace/FlagGems/src")
import flag_gems
from flag_gems import runtime
import importlib
mxmm = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")

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

M = 2048
a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
ref = a.float() @ w.t().float()

# 1) vendor reference
t = timeit(lambda: torch.mm(a, w.t()))
print(f"vendor              {t:7.1f} us  {2*M*N*K/(t*1e-6)/1e12:6.1f} TFLOPS", flush=True)

# 2) FlagGems expanded config sweep (incl cpasync, big tiles)
cands = []
for BM in (64, 128, 256):
    for BN in (64, 128, 256):
        for BK in (32, 64, 128):
            for ns in (2, 3, 4, 5):
                for nw in (4, 8):
                    for pl in ("basic", "cpasync"):
                        cands.append(triton.Config(
                            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK,
                             "GROUP_M": 8, "pipeline": pl, "scenario": ""},
                            num_stages=ns, num_warps=nw))
print(f"flaggems sweep candidates: {len(cands)}", flush=True)
orig_cfg = runtime.get_tuned_config
runtime.get_tuned_config = lambda op, **kw: cands if op == "mm_nt" else orig_cfg(op, **kw)
flag_gems.enable(record=False)
t = timeit(lambda: torch.mm(a, w.t()))
print(f"flaggems sweep best {t:7.1f} us  {2*M*N*K/(t*1e-6)/1e12:6.1f} TFLOPS", flush=True)
runtime.get_tuned_config = orig_cfg
print("done")
