#!/usr/bin/env python3
"""Probe: which FlagGems path does torch.mm(x, w_nn) actually take at M=64?

verify_nn_db measured 343-921us for the nn path while the raw nn_db kernel
sweeps to 56-151us — suspect splitk_mm intercepts before _select_nn_db.
This script monkeypatches each candidate launcher to print when it fires.
"""
import sys
import torch

sys.path.insert(0, "/workspace/FlagGems/src")
import flag_gems
from flag_gems.runtime.backend._metax.ops import mm as mmmod

flag_gems.enable(record=False)
dev = "cuda"

for fname in ("splitk_mm", "splitk_mm_two_step", "nt_db_mm", "nt_db_splitk_mm",
              "nn_db_mm", "nn_db_splitk_mm", "general_mm_nn", "general_mm_nt",
              "general_mm", "gemv_mm"):
    orig = getattr(mmmod, fname, None)
    if orig is None:
        continue
    def make(name, fn):
        def wrapped(a, b, c, M, N, K, *rest):
            print(f"  -> path={name} M={M} N={N} K={K}", flush=True)
            return fn(a, b, c, M, N, K, *rest)
        return wrapped
    setattr(mmmod, fname, make(fname, orig))

for name, K, N in [("qkv", 2048, 2560), ("o_proj", 2048, 2048),
                   ("gate_up", 2048, 12288), ("down", 6144, 2048)]:
    a = torch.randn(64, K, device=dev, dtype=torch.bfloat16)
    w_nn = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
    print(f"[{name}] torch.mm nn:", flush=True)
    c = torch.mm(a, w_nn)
    torch.cuda.synchronize()
print("done")
