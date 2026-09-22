#!/usr/bin/env python3
"""Verify nt_db routing: accuracy + speed through the aten path."""
import sys
import torch

sys.path.insert(0, "/workspace/FlagGems/src")
import flag_gems

flag_gems.enable(record=False)
dev = "cuda"

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

print("== decode shapes (expect nt_db path) ==")
tot = 0.0
for name, K, N in [("qkv", 2048, 2560), ("o_proj", 2048, 2048),
                   ("gate_up", 2048, 12288), ("down", 6144, 2048)]:
    a = torch.randn(64, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    c = torch.mm(a, w.t())
    ref = a.float() @ w.t().float()
    rel = ((c.float() - ref).abs().max() / ref.abs().max()).item()
    t = timeit(lambda: torch.mm(a, w.t()))
    tot += t
    ok = "OK" if rel < 0.01 else "BAD"
    print(f"{name:<9} {t:7.1f} us  relerr={rel:.5f} {ok}")
print(f"layer total (4 GEMMs): {tot:.1f} us  (was ~442 us) -> est step {42*tot/1000:.2f} ms (was ~18.5 ms)")

print("== regression: shapes that must NOT take nt_db ==")
for name, M, K, N in [("big-M", 1024, 2048, 2560), ("gemv", 2048, 2048, 1),
                      ("odd-K", 64, 2000, 2048), ("fp32", 64, 2048, 2048)]:
    dt = torch.float32 if name == "fp32" else torch.bfloat16
    a = torch.randn(M, K, device=dev, dtype=dt)
    w = torch.randn(N, K, device=dev, dtype=dt) if N > 1 else torch.randn(K, 1, device=dev, dtype=dt)
    b = w.t() if N > 1 else w
    c = torch.mm(a, b)
    ref = a.float() @ b.float()
    rel = ((c.float() - ref).abs().max() / ref.abs().max().clamp(min=1)).item()
    t = timeit(lambda: torch.mm(a, b), 50)
    print(f"{name:<9} M={M:<5}K={K:<5}N={N:<6} {t:7.1f} us  relerr={rel:.5f}")
print("done")
