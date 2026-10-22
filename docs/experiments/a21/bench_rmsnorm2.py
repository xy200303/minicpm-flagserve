"""Prototype: Triton fused_add_rms_norm that reproduces vLLM's IR 'native' numerics.

native (fp32 path):
    s = x.float() + residual.float()
    residual_out = s.to(bf16)
    var = s.pow(2).mean(-1)
    out = ((s * rsqrt(var + eps)).to(bf16) * weight).to(bf16)
"""
import time

import torch
import triton
import triton.language as tl

from vllm.ir.ops.layernorm import fused_add_rms_norm as ir_native
from vllm.ir.ops.layernorm import rms_norm as ir_rms_native
import flag_gems


@triton.jit
def _fused_add_rms_norm_kernel(
    OUT, X, RES, W,
    stride_o, stride_x, stride_r,
    H: tl.constexpr, BLOCK: tl.constexpr, EPS,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(RES + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
    s = x + r

    var = tl.sum(tl.where(mask, s * s, 0.0), 0) / H
    xr = s * tl.rsqrt(var + EPS)

    # native casts to the weight dtype *before* multiplying by weight
    w = tl.load(W + cols, mask=mask, other=0.0)
    out = xr.to(w.dtype) * w

    tl.store(RES + row * stride_r + cols, s.to(RES.dtype.element_ty), mask=mask)
    tl.store(OUT + row * stride_o + cols, out.to(OUT.dtype.element_ty), mask=mask)


def fused_add_rms_norm(x, residual, weight, eps, num_warps=8):
    H = x.shape[-1]
    out = torch.empty_like(x)
    rows = x.numel() // H
    x2 = x.view(rows, H)
    r2 = residual.view(rows, H)
    o2 = out.view(rows, H)
    _fused_add_rms_norm_kernel[(rows,)](
        o2, x2, r2, weight, o2.stride(0), x2.stride(0), r2.stride(0),
        H=H, BLOCK=triton.next_power_of_2(H), EPS=eps,
        num_warps=num_warps,
    )
    return out


def bench(fn, n=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


N, H, eps = 64, 2048, 1e-6
torch.manual_seed(0)
x = torch.randn(N, H, dtype=torch.bfloat16, device="cuda")
r = torch.randn(N, H, dtype=torch.bfloat16, device="cuda")
w = torch.randn(H, dtype=torch.bfloat16, device="cuda")

o_nat, res_nat = ir_native(x.clone(), r.clone(), w, eps)
for nw in (4, 8, 16):
    o = fused_add_rms_norm(x.clone(), r.clone(), w, eps, num_warps=nw)
    torch.cuda.synchronize()
    d = (o.float() - o_nat.float()).abs().max().item()
    t = bench(lambda: fused_add_rms_norm(x, r, w, eps, num_warps=nw))
    print(f"mine w{nw:2d}: {t*1e3:6.1f} us  maxdiff={d:.6f}", flush=True)

r2 = r.clone()
o_fg, _ = flag_gems.fused_add_rms_norm(x.clone(), r2.clone(), (H,), w, eps)
t_fg = bench(lambda: flag_gems.fused_add_rms_norm(x, r2, (H,), w, eps))
t_nat = bench(lambda: ir_native(x, r, w, eps))
print(f"native: {t_nat*1e3:6.1f} us | flag_gems: {t_fg*1e3:6.1f} us", flush=True)
o_fg2, _ = flag_gems.fused_add_rms_norm(x.clone(), r.clone(), (H,), w, eps)
print("flag_gems maxdiff vs native: %.6f"
      % (o_fg2.float() - o_nat.float()).abs().max().item(), flush=True)

# residual must match too
print("residual maxdiff (mine vs native): %.6f"
      % ((res_nat.float() - r.clone().float().add_(0)).abs().max().item()), flush=True)
