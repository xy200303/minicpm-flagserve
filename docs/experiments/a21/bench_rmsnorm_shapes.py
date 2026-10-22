"""A21 post-mortem: fused kernel at prefill-sized shapes vs the native chain."""
import time

import torch

from vllm.ir.ops.layernorm import fused_add_rms_norm as ir_native
from flag_gems.fused.fused_add_rms_norm_exact import fused_add_rms_norm_exact as fg


def bench(fn, n=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


H, eps = 2048, 1e-6
w = torch.randn(H, dtype=torch.bfloat16, device="cuda")
for N in (64, 512, 2048, 8192):
    x = torch.randn(N, H, dtype=torch.bfloat16, device="cuda")
    r = torch.randn(N, H, dtype=torch.bfloat16, device="cuda")
    o1, _ = ir_native(x.clone(), r.clone(), w, eps)
    o2 = fg(x.clone(), r.clone(), w, eps)
    torch.cuda.synchronize()
    d = (o1.float() - o2.float()).abs().max().item()
    t_nat = bench(lambda: ir_native(x, r, w, eps))
    t_fg = bench(lambda: fg(x, r, w, eps))
    gb = N * H * 2 * 3 / 1e9          # x + residual read, out+residual write (approx)
    print(f"N={N:5d}: native {t_nat*1e3:8.1f} us ({gb/(t_nat/1e3):5.0f} GB/s) | "
          f"fused {t_fg*1e3:8.1f} us ({gb/(t_fg/1e3):5.0f} GB/s) | "
          f"{t_nat/t_fg:4.2f}x | maxdiff {d:.5f}", flush=True)
