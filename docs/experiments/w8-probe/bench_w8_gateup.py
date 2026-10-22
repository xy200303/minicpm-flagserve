"""W8 gate_up: int8 GEMM + fused (dequant + SwiGLU) epilogue vs bf16 GEMM + silu_and_mul.

gate_up output is [M, 2*I]; SwiGLU consumes it as silu(a) * b with a = [:I], b = [I:].
Fusing dequant into that epilogue removes the separate dequant pass entirely.
"""
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_silu_mul(ACC, SW, OUT, sx, I,
                      stride_am, stride_om,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < I
    a = tl.load(ACC + offs_m[:, None] * stride_am + offs_n[None, :],
                mask=mask[None, :]).to(tl.float32)
    b = tl.load(ACC + offs_m[:, None] * stride_am + (offs_n + I)[None, :],
                mask=mask[None, :]).to(tl.float32)
    sa = tl.load(SW + offs_n, mask=mask, other=1.0)
    sb = tl.load(SW + offs_n + I, mask=mask, other=1.0)
    a = a * (sx * sa)[None, :]
    b = b * (sx * sb)[None, :]
    out = (a * tl.sigmoid(a)) * b
    tl.store(OUT + offs_m[:, None] * stride_om + offs_n[None, :],
             out.to(OUT.dtype.element_ty), mask=mask[None, :])


def w8_gate_up(qx, qwt, sw, sx, I, bm=16, bn=512, warps=4):
    acc = torch._int_mm(qx, qwt)                      # [M, 2I] int32
    M = acc.shape[0]
    out = torch.empty(M, I, dtype=torch.bfloat16, device=acc.device)
    _dequant_silu_mul[(triton.cdiv(M, bm), triton.cdiv(I, bn))](
        acc, sw, out, sx, I, acc.stride(0), out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, num_warps=warps)
    return out


def bench(fn, n=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


M, K, I = 64, 2048, 6144
N = 2 * I
torch.manual_seed(0)
x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.02
wt = w.t().contiguous()

# bf16 path: GEMM + silu_and_mul (torch chain, what the plugin does today)
def bf16_path():
    y = x @ wt
    a, b = y[..., :I], y[..., I:]
    return (torch.nn.functional.silu(a) * b)

y_ref = bf16_path()

# int8 path
xi = x.abs().amax().item() / 127.0
wi = w.abs().amax(dim=1, keepdim=True).float() / 127.0
qx = (x.float() / xi).round().clamp(-127, 127).to(torch.int8)
qw = (w.float() / wi).round().clamp(-127, 127).to(torch.int8)
sw = wi.view(-1).contiguous()
qwt = qw.t().contiguous()          # [K, 2I] int8, prepared offline

t_bf16 = bench(bf16_path)
print(f"bf16 GEMM+silu: {t_bf16*1e3:7.1f} us", flush=True)
best = None
for bm in (8, 16, 32):
    for bn in (256, 512, 1024):
        for wp in (2, 4, 8):
            try:
                out = w8_gate_up(qx, qwt, sw, xi, I, bm, bn, wp)
                torch.cuda.synchronize()
                t = bench(lambda: w8_gate_up(qx, qwt, sw, xi, I, bm, bn, wp))
                if best is None or t < best[0]:
                    best = (t, bm, bn, wp, out.clone())
            except Exception:
                continue
t, bm, bn, wp, out = best
print(f"w8 GEMM+fused silu: {t*1e3:7.1f} us  x{t_bf16/t:.2f}  cfg=({bm},{bn},{wp})",
      flush=True)
d = (out.float() - y_ref.float()).abs()
rel = d.max().item() / y_ref.float().abs().max().item()
print(f"w8 maxdiff {d.max().item():.4f} (rel {rel:.4f})  mean {d.mean().item():.5f}",
      flush=True)
print(f"per layer saving {((t_bf16-t)*1e3):.1f} us -> x42 = {(t_bf16-t)*42:.2f} ms/step",
      flush=True)
