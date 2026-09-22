#!/usr/bin/env python3
"""Standalone test: manual double-buffered nt GEMM for skinny-M decode shapes."""
import torch
import triton
import triton.language as tl

dev = "cuda"

@triton.jit
def mm_nt_db(A, B, C, M, N, K,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = A + rm[:, None] * K + rk[None, :]
    b_ptrs = B + rk[:, None] + rn[None, :] * K  # B is [N,K] row-major

    a = tl.load(a_ptrs, mask=rm[:, None] < M, other=0.0)
    b = tl.load(b_ptrs)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    iters = tl.cdiv(K, BLOCK_K)
    for _ in range(iters - 1):
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
        a_n = tl.load(a_ptrs, mask=rm[:, None] < M, other=0.0)
        b_n = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a = a_n
        b = b_n
    acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    c_ptrs = C + rm[:, None] * N + rn[None, :]
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=rm[:, None] < M)


def timeit(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3


M = 64
for name, K, N in [("qkv", 2048, 2560), ("o_proj", 2048, 2048),
                   ("gate_up", 2048, 12288), ("down", 6144, 2048),
                   ("lm_head", 2048, 130560)]:
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    c = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    ref = a.float() @ w.t().float()

    best = (1e9, None)
    for BM in (64,):
        for BN in (64, 128, 256):
            for BK in (64, 128, 256):
                if K % BK:
                    continue
                for nw in (4, 8):
                    for ns in (1, 2, 3):
                        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
                        def run():
                            mm_nt_db[grid](a, w, c, M, N, K,
                                           BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                                           num_warps=nw, num_stages=ns)
                        try:
                            run()
                            err = ((c.float() - ref).abs().max() / ref.abs().max()).item()
                            if err > 0.02:
                                if best[1] is None and not hasattr(run, '_logged'):
                                    print(f"{name} cfg={(BM,BN,BK)} err={err}")
                                    run._logged = True
                                continue
                            t = timeit(run, 100)
                        except Exception as ex:
                            if best[1] is None and not hasattr(run, '_logged2'):
                                print(f"{name} cfg={(BM,BN,BK,nw,ns)} exc={type(ex).__name__}: {str(ex)[:150]}")
                                run._logged2 = True
                            continue
                        if t < best[0]:
                            best = (t, (BM, BN, BK, nw, ns))
    bw = K * N * 2 / (best[0] * 1e-6) / 1e9 if best[1] else 0
    print(f"{name:<9} best={best[0]:7.1f} us  cfg={best[1]}  weightBW={bw:6.0f} GB/s", flush=True)
