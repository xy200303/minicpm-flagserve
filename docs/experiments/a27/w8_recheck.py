"""A27: re-check W8 gate_up with a single-kernel per-token activation quant.

A22 measured 157.5 us vs 120.9 us for bf16, but its activation quant was
    amax = x.abs().max().float().reshape(1)   # 3 kernels
    _quant_act_kernel[...]                    # +1
i.e. four tiny launches over a 64x2048 tensor.  This measures what is left
when the quant folds into one kernel.
"""
import time

import torch
import triton
import triton.language as tl

M, K = 64, 2048
I = 6144
N = 2 * I


def bench(fn, n=100, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


@triton.jit
def _quant_row(X, QX, SX, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < K
    x = tl.load(X + row * K + cols, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x))
    scale = tl.maximum(amax / 127.0, 1e-8)
    q = tl.clamp(tl.floor(x / scale + 0.5), -127.0, 127.0).to(tl.int8)
    tl.store(QX + row * K + cols, q, mask=mask)
    tl.store(SX + row, scale)


@triton.jit
def _quant_static(X, QX, INV_PTR, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    inv = tl.load(INV_PTR)
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    q = tl.clamp(tl.floor(x * inv + 0.5), -127.0, 127.0).to(tl.int8)
    tl.store(QX + offs, q, mask=mask)


@triton.jit
def _dequant_silu(ACC, SW, SX, OUT, I: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < I
    sx = tl.load(SX + row)
    a = tl.load(ACC + row * (2 * I) + offs, mask=mask, other=0).to(tl.float32)
    b = tl.load(ACC + row * (2 * I) + offs + I, mask=mask, other=0).to(tl.float32)
    sa = tl.load(SW + offs, mask=mask, other=1.0)
    sb = tl.load(SW + offs + I, mask=mask, other=1.0)
    a = a * (sx * sa)
    b = b * (sx * sb)
    out = (a * tl.sigmoid(a)) * b
    tl.store(OUT + row * I + offs, out.to(tl.bfloat16), mask=mask)


def main():
    torch.manual_seed(0)
    x = (torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.5)
    w = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.02)

    qmax = w.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-8)
    sw = (qmax / 127.0)
    qw = (w.float() / sw).round().clamp(-127, 127).to(torch.int8)
    qwt = qw.t().contiguous()

    out_bf = torch.empty(M, I, dtype=torch.bfloat16, device="cuda")
    qx = torch.empty(M, K, dtype=torch.int8, device="cuda")
    sx = torch.empty(M, dtype=torch.float32, device="cuda")
    inv = torch.ones(1, dtype=torch.float32, device="cuda") * (127.0 / 2.0)
    swv = sw.squeeze(1).contiguous()

    def bf16_path():
        y = x @ w.t()
        return torch.nn.functional.silu(y[:, :I]) * y[:, I:]

    def w8_naive():
        amax = x.abs().max().float().reshape(1)
        inv_ = 127.0 / torch.clamp(amax, min=1e-8)
        _quant_static[(triton.cdiv(M * K, 8192),)](x, qx, inv_, M * K, BLOCK=8192)
        acc_ = torch._int_mm(qx, qwt)
        sxs = (amax / 127.0).expand(M).contiguous()
        _dequant_silu[(M, triton.cdiv(I, 512))](acc_, swv, sxs, out_bf, I, 512)
        return out_bf

    def w8_row():
        _quant_row[(M,)](x, qx, sx, K, 2048)
        acc_ = torch._int_mm(qx, qwt)
        _dequant_silu[(M, triton.cdiv(I, 512))](acc_, swv, sx, out_bf, I, 512)
        return out_bf

    def w8_static():
        _quant_static[(triton.cdiv(M * K, 8192),)](x, qx, inv, M * K, BLOCK=8192)
        acc_ = torch._int_mm(qx, qwt)
        _dequant_silu[(M, triton.cdiv(I, 512))](acc_, swv, inv, out_bf, I, 512)
        return out_bf

    def int_mm_only():
        return torch._int_mm(qx, qwt)

    def bf16_gemm_only():
        return x @ w.t()

    def quant_row_only():
        _quant_row[(M,)](x, qx, sx, K, 2048)

    ref = bf16_path().float()
    rows = [
        ("bf16 gemm+silu", bf16_path),
        ("bf16 gemm alone", bf16_gemm_only),
        ("int_mm alone", int_mm_only),
        ("quant_row kernel only", quant_row_only),
        ("w8_naive (A22 impl)", w8_naive),
        ("w8_row (1-kernel quant)", w8_row),
        ("w8_static (no amax)", w8_static),
    ]
    for name, fn in rows:
        ms = bench(fn)
        note = ""
        if name.startswith("w8"):
            d = (fn().float() - ref).abs().max().item()
            note = "  maxdiff=%.3f" % d
        print("%-26s %8.1f us%s" % (name, ms, note), flush=True)


if __name__ == "__main__":
    main()
