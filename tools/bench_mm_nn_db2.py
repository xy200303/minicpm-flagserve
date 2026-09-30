#!/usr/bin/env python3
"""Round-2 sweep for nn_db at M=64 qkv/o_proj shapes: hunt <47us.

Adds BLOCK_M=32 (more CTAs on 104 SMs) and num_stages=3 to the space.
"""
import torch
import triton
import triton.language as tl

dev = "cuda"


@triton.jit
def mm_nn_db(A, B, C, M, N, K,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = A + rm[:, None] * K + rk[None, :]
    b_ptrs = B + rk[:, None] * N + rn[None, :]
    a = tl.load(a_ptrs, mask=rm[:, None] < M, other=0.0)
    b = tl.load(b_ptrs, mask=rn[None, :] < N, other=0.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(tl.cdiv(K, BLOCK_K) - 1):
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
        a_n = tl.load(a_ptrs, mask=rm[:, None] < M, other=0.0)
        b_n = tl.load(b_ptrs, mask=rn[None, :] < N, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a = a_n
        b = b_n
    acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    c_ptrs = C + rm[:, None] * N + rn[None, :]
    tl.store(c_ptrs, acc.to(C.dtype.element_ty),
             mask=(rm < M)[:, None] & (rn < N)[None, :])


def timeit(fn, iters=100):
    for _ in range(15):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3


M = 64
for name, K, N in [("qkv", 2048, 2560), ("o_proj", 2048, 2048)]:
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    wt = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
    c = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    ref = a.float() @ wt.float()
    best = (1e9, None)
    for BM in (32, 64):
        for BN in (32, 64, 128):
            for BK in (64, 128, 256):
                if K % BK:
                    continue
                for ns in (2, 3):
                    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
                    def run():
                        mm_nn_db[grid](a, wt, c, M, N, K, BLOCK_M=BM,
                                       BLOCK_N=BN, BLOCK_K=BK,
                                       num_warps=8, num_stages=ns)
                    try:
                        run()
                        rel = ((c.float() - ref).abs().max()
                               / ref.abs().max()).item()
                        if rel > 0.01:
                            continue
                        t = timeit(run, 100)
                    except Exception:
                        continue
                    if t < best[0]:
                        best = (t, (BM, BN, BK, ns))
    print(f"{name:<9} M=64 nn_db r2 best={best[0]:7.1f} us  "
          f"cfg(BM,BN,BK,ns)={best[1]}  [nt_db today ~47/46 us]", flush=True)
print("done")
