"""Clean check (fresh inputs per numeric comparison; in-place timing is fine)."""
import time

import torch
import triton
import triton.language as tl

from vllm.ir.ops.layernorm import fused_add_rms_norm as ir_native
import flag_gems
from bench_rmsnorm2 import fused_add_rms_norm as mine


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
x0 = torch.randn(N, H, dtype=torch.bfloat16, device="cuda")
r0 = torch.randn(N, H, dtype=torch.bfloat16, device="cuda")
w = torch.randn(H, dtype=torch.bfloat16, device="cuda")

o_nat, res_nat = ir_native(x0.clone(), r0.clone(), w, eps)
torch.cuda.synchronize()

for nw in (1, 2, 4, 8):
    o = mine(x0.clone(), r0.clone(), w, eps, num_warps=nw)
    torch.cuda.synchronize()
    print(f"mine w{nw}: maxdiff out = {(o.float()-o_nat.float()).abs().max().item():.6f}",
          flush=True)

o_fg, res_fg = flag_gems.fused_add_rms_norm(x0.clone(), r0.clone(), (H,), w, eps)
torch.cuda.synchronize()
print("flag_gems: maxdiff out = %.6f  residual = %.6f"
      % ((o_fg.float()-o_nat.float()).abs().max().item(),
         (res_fg.float()-res_nat.float()).abs().max().item()), flush=True)

# residual check for mine: it writes the sum back into the input buffer
xr, rr = x0.clone(), r0.clone()
mine(xr, rr, w, eps, num_warps=4)
torch.cuda.synchronize()
print("mine: residual maxdiff = %.6f"
      % (rr.float() - res_nat.float()).abs().max().item(), flush=True)

# timing (in-place is what the engine does per step)
xb, rb = x0.clone(), r0.clone()
t_nat = bench(lambda: ir_native(xb, rb, w, eps))
t_fg = bench(lambda: flag_gems.fused_add_rms_norm(xb, rb, (H,), w, eps))
tm = {nw: bench(lambda: mine(xb, rb, w, eps, num_warps=nw)) for nw in (1, 2, 4, 8)}
print(f"timing: native {t_nat*1e3:6.1f} us | flag_gems {t_fg*1e3:6.1f} us | "
      + " | ".join(f"mine w{nw} {t*1e3:5.1f} us" for nw, t in tm.items()), flush=True)
best = min(tm.values())
print(f"per-step saving (2 norms/layer x 42): {(t_nat-best)*2*42:.2f} ms "
      f"(flag_gems: {(t_nat-t_fg)*2*42:.2f} ms)", flush=True)
