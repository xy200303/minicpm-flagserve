#!/usr/bin/env python3
"""Expanded config sweep for metax mm_kernel_nn at prefill shapes (M=2048).

Kernel body copied verbatim from
FlagGems/src/flag_gems/runtime/backend/_metax/ops/mm.py::mm_kernel_nn.
torch.mm here is VENDOR (flag_gems NOT enabled in this process).
"""
import itertools
import torch
import triton
import triton.language as tl

dev = "cuda"
SHAPES = [
    ("qkv",     2048, 2560),
    ("o_proj",  2048, 2048),
    ("gate_up", 2048, 12288),
    ("down",    6144, 2048),
]
M = 2048

@triton.jit
def mm_kernel_nn(
    A, B, C, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if EVEN_M:
        ram = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
    else:
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    if EVEN_N:
        rbn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
    else:
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = A + ram[:, None] * K + rk[None, :]
    b_ptrs = B + rk[:, None] * N + rbn[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=rk[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=rk[None, :] < k_remaining, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    c_ptrs = C + rm[:, None] * N + rn[None, :]
    result = acc.to(C.dtype.element_ty)
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, result)
    else:
        tl.store(c_ptrs, result, mask=(rm < M)[:, None] & (rn < N)[None, :])


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


CONFIGS = []
for bm, bn, bk in itertools.product((64, 128, 256), (64, 128, 256), (32, 64, 128)):
    if bm * bn < 128 * 128:
        continue
    if bm * bn > 256 * 256:
        continue
    for st, wp in ((2, 4), (2, 8), (3, 8), (4, 8), (5, 8), (3, 16), (4, 16)):
        CONFIGS.append((bm, bn, bk, st, wp))
print(f"{len(CONFIGS)} configs per shape")

for name, K, N in SHAPES:
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
    ref = torch.mm(a, b)
    t_vendor = timeit(lambda: torch.mm(a, b))
    best = (1e9, None)
    for bm, bn, bk, st, wp in CONFIGS:
        c = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
        def run():
            mm_kernel_nn[grid](
                a, b, c, M, N, K, bm, bn, bk, 8,
                M % bm == 0, N % bn == 0, K % bk == 0,
                num_stages=st, num_warps=wp,
            )
        try:
            run()
            torch.cuda.synchronize()
            err = (c.float() - ref.float()).abs().max().item()
            rel = err / ref.float().abs().max().item()
            if rel > 0.02:
                print(f"  WRONG bm{bm} bn{bn} bk{bk} s{st} w{wp} rel={rel}")
                continue
            t = timeit(run)
            if t < best[0]:
                best = (t, (bm, bn, bk, st, wp))
        except Exception:
            continue
    tf = 2 * M * N * K / (best[0] * 1e-6) / 1e12
    tfv = 2 * M * N * K / (t_vendor * 1e-6) / 1e12
    print(f"{name:<8} K={K} N={N}: vendor {t_vendor:7.1f}us ({tfv:5.1f}TF) | "
          f"triton best {best[0]:7.1f}us ({tf:5.1f}TF) cfg={best[1]} | "
          f"ratio {best[0]/t_vendor:.2f}x", flush=True)
