#!/usr/bin/env python3
"""Unit test: mcblas_mm routing with FlagGems ENABLED (the serve condition)."""
import sys
import torch

sys.path.insert(0, "/workspace/FlagGems/src")
import flag_gems
flag_gems.enable(record=False)  # now aten::mm is Triton; vendor unreachable via aten

from vllm_fl.dispatch.backends.vendor.metax.patches import mcblas_mm as mb
from vllm_fl.dispatch.backends.vendor.metax.patches.linear_nn_repack import (
    _metax_unquantized_gemm,
)

dev = "cuda"
assert mb.AVAILABLE, "mcblas binding not available"

class FakeLayer:
    pass

def timeit(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3

ok = True
for (K, N) in [(2048, 2560), (2048, 2048), (2048, 12288), (6144, 2048)]:
    layer = FakeLayer()
    w_nt = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    layer._fl_w_nn = w_nt.t().contiguous()
    layer._fl_always_nn = False
    for M in (64, 2048):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        out = _metax_unquantized_gemm(layer, x, w_nt, None)
        ref = torch.nn.functional.linear(x, w_nt)  # stock gems path, fp32 accum
        rel = (out.float() - ref.float()).abs().max().item() / max(ref.float().abs().max().item(), 1e-9)
        t = timeit(lambda: _metax_unquantized_gemm(layer, x, w_nt, None))
        route = "vendor-nn" if M > 192 else "nt(triton)"
        good = rel < 2e-2
        ok &= good
        print(f"K={K} N={N} M={M}: route={route} rel_err={rel:.2e} {t:8.1f}us {'OK' if good else 'BAD'}")
print("ALL OK" if ok else "FAILURES")
