"""Isolated check: vLLM's IR 'native' RMSNorm vs FlagGems' fused Triton kernel.

Shape is the decode-step one for MiniCPM5-2B: [num_tokens=64, hidden=2048] bf16.
"""
import time

import torch

import flag_gems
from vllm.ir.ops.layernorm import fused_add_rms_norm as ir_fused_native
from vllm.ir.ops.layernorm import rms_norm as ir_rms_native


def bench(fn, n=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


N, H = 64, 2048
eps = 1e-6
torch.manual_seed(0)
x = torch.randn(N, H, dtype=torch.bfloat16, device="cuda")
r = torch.randn(N, H, dtype=torch.bfloat16, device="cuda")
w = torch.randn(H, dtype=torch.bfloat16, device="cuda")

# --- numerics -------------------------------------------------------------
x1, r1 = x.clone(), r.clone()
out_nat, res_nat = ir_fused_native(x1, r1, w, eps)
x2, r2 = x.clone(), r.clone()
out_fg, res_fg = flag_gems.fused_add_rms_norm(x2, r2, (H,), w, eps)
torch.cuda.synchronize()
print("fused_add_rms_norm: maxdiff out = %.5f  res = %.5f"
      % ((out_nat.float() - out_fg.float()).abs().max().item(),
         (res_nat.float() - res_fg.float()).abs().max().item()),
      flush=True)

xn = x.clone()
o_nat = ir_rms_native(xn, w, eps)
xf = x.clone()
o_fg = flag_gems.rms_norm(xf, (H,), w, eps)
torch.cuda.synchronize()
print("rms_norm:           maxdiff out = %.5f"
      % (o_nat.float() - o_fg.float()).abs().max().item(), flush=True)

# --- speed ----------------------------------------------------------------
def run_native():
    xx, rr = x.clone(), r.clone()
    ir_fused_native(xx, rr, w, eps)


def run_fg_inplace():
    # in-place: no clones, like the engine would call it on fresh buffers
    flag_gems.fused_add_rms_norm(x, r, (H,), w, eps)


x_buf = x.clone()
r_buf = r.clone()

t_nat = bench(lambda: ir_fused_native(x_buf, r_buf, w, eps))
t_fg = bench(run_fg_inplace)
out_nat2 = ir_rms_native(x_buf, w, eps)
t_nat_r = bench(lambda: ir_rms_native(x_buf, w, eps))
t_fg_r = bench(lambda: flag_gems.rms_norm(x_buf, (H,), w, eps))
print(f"fused_add_rms_norm  native {t_nat*1e3:7.1f} us | flag_gems {t_fg*1e3:7.1f} us "
      f"| {t_nat/t_fg:.1f}x", flush=True)
print(f"rms_norm            native {t_nat_r*1e3:7.1f} us | flag_gems {t_fg_r*1e3:7.1f} us "
      f"| {t_nat_r/t_fg_r:.1f}x", flush=True)
per_step = (t_nat - t_fg) * 2 * 42       # two norms per layer, 42 layers
print(f"implied saving/step if both norms switch: {per_step:.2f} ms", flush=True)
