"""W8 prototype, step 1: a well-tuned int32 -> bf16 dequant epilogue.

out[m, n] = acc[m, n] * sx * sw[n]      (acc int32 from torch._int_mm)
Sweeps tile shapes so the epilogue is not the bottleneck any more.
"""
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant(ACC, SW, OUT, sx, N,
             stride_am, stride_om,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N
    acc = tl.load(ACC + offs_m[:, None] * stride_am + offs_n[None, :],
                  mask=mask[None, :])
    sw = tl.load(SW + offs_n, mask=mask, other=1.0)
    out = acc.to(tl.float32) * (sx * sw)[None, :]
    tl.store(OUT + offs_m[:, None] * stride_om + offs_n[None, :],
             out.to(OUT.dtype.element_ty), mask=mask[None, :])


def fused_dequant(acc, sw, sx, bm=16, bn=512, warps=4):
    M, N = acc.shape
    out = torch.empty(M, N, dtype=torch.bfloat16, device=acc.device)
    _dequant[(triton.cdiv(M, bm), triton.cdiv(N, bn))](
        acc, sw, out, sx, N, acc.stride(0), out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, num_warps=warps)
    return out


def bench(fn, n=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


configs = [(bm, bn, w) for bm in (8, 16, 32, 64) for bn in (256, 512, 1024)
           for w in (2, 4, 8)]
for M, N in ((64, 2048), (64, 12288), (64, 130560)):
    acc = torch.randint(-127, 127, (M, N), dtype=torch.int32, device="cuda")
    sw = torch.rand(N, dtype=torch.float32, device="cuda") * 1e-3
    ref = (acc.float() * sw).to(torch.bfloat16)
    t_torch = bench(lambda: (acc.float() * sw).to(torch.bfloat16))
    best = None
    for bm, bn, w in configs:
        try:
            out = fused_dequant(acc, sw, 0.01, bm, bn, w)
            if not torch.equal(out, (acc.float() * (0.01 * sw)).to(torch.bfloat16)):
                continue
            t = bench(lambda: fused_dequant(acc, sw, 0.01, bm, bn, w))
            if best is None or t < best[0]:
                best = (t, bm, bn, w)
        except Exception:
            continue
    gb = M * N * 6 / 1e9
    print(f"[{M}x{N}] torch {(acc.float()*sw).to(torch.bfloat16) is not None and t_torch*1e3:.1f} us"
          f" | fused best {best[0]*1e3:.1f} us ({gb/(best[0]/1e3):.0f} GB/s) "
          f"cfg={best[1:]}, {t_torch/best[0]:.1f}x", flush=True)
