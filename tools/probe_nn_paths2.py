#!/usr/bin/env python3
"""Direct per-path timing at M=64: which launcher produced 343-921us?"""
import sys
import torch

sys.path.insert(0, "/workspace/FlagGems/src")
import flag_gems
from flag_gems.runtime.backend._metax.ops import mm as mmmod

flag_gems.enable(record=False)
dev = "cuda"


def timeit(fn, iters=100):
    for _ in range(15):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3


for name, K, N in [("qkv", 2048, 2560), ("o_proj", 2048, 2048),
                   ("gate_up", 2048, 12288), ("down", 6144, 2048)]:
    M = 64
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)  # 原始 (N,K) 布局
    w_nn = w.t().contiguous()                                # 重排 (K,N)
    w_nt = w.t()                                             # (K,N) nt 视图, strides (1,K)
    c = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    line = [f"{name:<9} dispatch_mm={timeit(lambda: torch.mm(a, w_nn)):7.1f}"]
    for fname, b in [("nt_db via mm(a,w.t())", None)]:
        pass
    line.append(f"nt_db(mm .t() view)={timeit(lambda: torch.mm(a, w_nt)):6.1f}")
    for fname in ("splitk_mm", "nn_db_mm", "general_mm_nn"):
        fn = getattr(mmmod, fname, None)
        if fn is None:
            line.append(f"{fname}=N/A")
            continue
        try:
            if fname == "splitk_mm":
                t = timeit(lambda: fn(a, w_nn, c, M, N, K), 50)
            else:
                t = timeit(lambda: fn(a, w_nn, c, M, N, K))
            line.append(f"{fname}={t:6.1f}")
        except Exception as ex:
            line.append(f"{fname}=ERR:{type(ex).__name__}")
    print(" ".join(line), flush=True)
print("done")
