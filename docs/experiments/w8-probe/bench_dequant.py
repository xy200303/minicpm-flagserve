"""How cheap can the int8->bf16 dequant epilogue be if fused into one Triton kernel?"""
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_kernel(ACC, SW, OUT, sx, N, stride_am, stride_om,
                    BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m + tl.arange(0, 1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N
    acc = tl.load(ACC + offs_m[:, None] * stride_am + offs_n[None, :], mask=mask[None, :])
    sw = tl.load(SW + offs_n, mask=mask, other=1.0)
    out = acc.to(tl.float32) * (sx * sw)[None, :]
    tl.store(OUT + offs_m[:, None] * stride_om + offs_n[None, :],
             out.to(OUT.dtype.element_ty), mask=mask[None, :])


def fused_dequant(acc, sw, sx, block_n=1024):
    M, N = acc.shape
    out = torch.empty(M, N, dtype=torch.bfloat16, device=acc.device)
    _dequant_kernel[(M, triton.cdiv(N, block_n))](
        acc, sw, out, sx, N, acc.stride(0), out.stride(0), BLOCK_N=block_n)
    return out


def torch_dequant(acc, sw, sx):
    return (acc.float() * (sx * sw)).to(torch.bfloat16)


def bench(fn, n=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


for M, N in ((64, 2048), (64, 12288), (64, 130560)):
    acc = torch.randint(-127, 127, (M, N), dtype=torch.int32, device="cuda")
    sw = torch.rand(N, dtype=torch.float32, device="cuda") * 1e-3
    sx = 0.01
    t_t = bench(lambda: torch_dequant(acc, sw, sx))
    t_f = bench(lambda: fused_dequant(acc, sw, sx))
    gb = M * N * (4 + 2) / 1e9
    print(f"[{M}x{N}] torch epilogue {t_t*1e3:7.1f} us | fused {t_f*1e3:6.1f} us "
          f"({gb/(t_f/1e3):5.0f} GB/s) | {t_t/t_f:4.1f}x", flush=True)
