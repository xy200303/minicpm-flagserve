"""Split the FlagGems paged_decode_attention into its two launches and time each."""
import time

import torch
from flag_gems.fused import paged_decode_attention as op
from flag_gems.fused.paged_decode_attention import _segment_kernel, _combine_kernel, LOG2E


def bench(fn, n=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def run_case(KV, B, NSEG=32, BN=16, W=1):
    HQ, HKV, D, BS = 16, 2, 128, 16
    NB = KV // BS
    nb = B * NB
    torch.manual_seed(0)
    q = torch.randn(B, HQ, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(nb, BS, HKV, D, dtype=torch.bfloat16, device="cuda") * 0.1
    v = torch.randn(nb, BS, HKV, D, dtype=torch.bfloat16, device="cuda") * 0.1
    out = torch.empty_like(q)
    bt = torch.randperm(nb, dtype=torch.int32, device="cuda").view(B, NB).contiguous()
    seq = torch.full((B,), KV, dtype=torch.int32, device="cuda")
    gqa = HQ // HKV
    rows = B * HKV * NSEG
    sa = torch.empty((rows, gqa, D), dtype=torch.float32, device="cuda")
    sm = torch.empty((rows, gqa), dtype=torch.float32, device="cuda")
    ss = torch.empty((rows, gqa), dtype=torch.float32, device="cuda")
    kv_gb = B * KV * HKV * D * 2 * 2 / 1e9
    seg_gb = rows * gqa * D * 4 / 1e9

    def seg():
        _segment_kernel[(NSEG, HKV, B)](
            q, k, v, sa, sm, ss, bt, seq,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            bt.stride(0), bt.stride(1),
            HQ=HQ, HKV=HKV, D=D, BS=BS, GQA=gqa, BLOCK_N=BN, NSEG=NSEG,
            scale=D ** -0.5 * LOG2E, num_warps=W, num_stages=1)

    def comb():
        _combine_kernel[(B, HKV)](
            sa, sm, ss, out, out.stride(0), out.stride(1), out.stride(2),
            HQ=HQ, HKV=HKV, GQA=gqa, D=D, NSEG=NSEG, num_warps=4)

    seg(); comb()
    torch.cuda.synchronize()
    t_seg = bench(seg)
    t_comb = bench(comb)
    t_all = bench(lambda: op(q, k, v, out, bt, seq, D ** -0.5,
                             num_segments=NSEG, block_n=BN, num_warps=W,
                             seg_buffers=(sa, sm, ss)))
    print(f"[kv={KV} B={B}] segment {t_seg:.3f} ms ({kv_gb/(t_seg/1e3):4.0f} GB/s) | "
          f"combine {t_comb:.3f} ms ({seg_gb/(t_comb/1e3):4.0f} GB/s) | "
          f"total {t_all:.3f} ms | combine share {t_comb/t_all*100:.1f}%", flush=True)


run_case(4096, 64)
run_case(16384, 32)
