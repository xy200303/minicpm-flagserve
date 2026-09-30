#!/usr/bin/env python3
"""Verify nn_db routing: correctness + head-to-head vs nt_db at decode shapes.

Single-layout goal: if nn_db ([K,N] row-major weight, double-buffered) matches
nt_db at M<=128, decode can run on the same nn-repacked weight copy that
prefill uses, eliminating the +4GB duplicate layout.

Checks:
  1) torch.mm(x, w_nn) at skinny M routes to nn_db and matches fp32 reference;
  2) edge M values (1, 17, 127) correct;
  3) perf table: nt_db path (F.linear, [N,K]) vs nn_db path (mm, [K,N]).
"""
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
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3


print("== correctness: nn path at skinny M ==")
all_ok = True
for M in (1, 17, 64, 127, 128):
    for name, K, N in [("qkv", 2048, 2560), ("gate_up", 2048, 12288),
                       ("down", 6144, 2048)]:
        a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w_nn = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
        c = torch.mm(a, w_nn)
        ref = a.float() @ w_nn.float()
        rel = ((c.float() - ref).abs().max() / ref.abs().max().clamp(min=1)).item()
        ok = rel < 0.01
        all_ok &= ok
        print(f"M={M:<4} {name:<9} relerr={rel:.5f} {'OK' if ok else 'BAD'}")

print("== perf: nt_db vs nn_db at decode shapes (M=64) ==")
tot_nt = tot_nn = 0.0
for name, K, N in [("qkv", 2048, 2560), ("o_proj", 2048, 2048),
                   ("gate_up", 2048, 12288), ("down", 6144, 2048)]:
    a = torch.randn(64, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)  # 原始 (N,K)
    w_nn = w.t().contiguous()                                # 重排 (K,N)
    t_nt = timeit(lambda: torch.mm(a, w.t()))
    t_nn = timeit(lambda: torch.mm(a, w_nn))
    tot_nt += t_nt
    tot_nn += t_nn
    ratio = t_nn / t_nt
    verdict = "WIN" if ratio <= 1.05 else ("tie" if ratio <= 1.15 else "LOSE")
    print(f"{name:<9} nt_db={t_nt:6.1f} us  nn_db={t_nn:6.1f} us  ratio={ratio:.3f} {verdict}")
print(f"layer total: nt_db={tot_nt:.1f} us  nn_db={tot_nn:.1f} us  "
      f"ratio={tot_nn / tot_nt:.3f}")

print(f"correctness: {'ALL PASS' if all_ok else 'FAIL'}")
print("done")
