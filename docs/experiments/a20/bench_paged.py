"""Clean A/B: current FlagGems op vs a page-aligned segment kernel.

BLOCK_N == block_size means one loop iteration covers exactly one page:
`page` becomes a scalar load and `pos` is a constant arange, so the per-token
div/mod and the vector page gather disappear from the hot loop.
"""
import time

import torch
import triton
import triton.language as tl

from flag_gems.fused import paged_decode_attention as op
from flag_gems.fused.paged_decode_attention import _combine_kernel, LOG2E


@triton.jit
def _seg_paged(
    Q, K, V, SA, SM, SS, BT, SL,
    sqb, sqh, sqd, skb, sks, skh, skd, svb, svs, svh, svd, sbtb, sbtp,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
    GQA: tl.constexpr, NSEG: tl.constexpr, scale,
):
    g = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    od = tl.arange(0, D)
    og = tl.arange(0, GQA)
    kv = tl.load(SL + s)
    units = (kv + NSEG * BS - 1) // (NSEG * BS)
    start = g * units * BS
    end0 = start + units * BS
    # clamp to kvlen: an entirely-out-of-range segment must not iterate,
    # otherwise m stays -inf and exp2(-inf - -inf) turns the output into NaN
    end = tl.minimum(end0, kv)
    m_i = tl.full((GQA,), float("-inf"), tl.float32)
    l_i = tl.zeros((GQA,), tl.float32)
    acc = tl.zeros((GQA, D), tl.float32)
    qv = tl.load(Q + s * sqb + (h * GQA + og)[:, None] * sqh + od[None, :] * sqd)
    pos = tl.arange(0, BS)
    for t0 in range(start, end, BS):
        if t0 + BS <= kv:
            page = tl.load(BT + s * sbtb + (t0 // BS) * sbtp)
            kt = tl.load(K + page * skb + pos[None, :] * sks + h * skh
                         + od[:, None] * skd)
            vt = tl.load(V + page * svb + pos[:, None] * svs + h * svh
                         + od[None, :] * svd)
            sc = tl.dot(qv, kt) * scale
        else:
            mask_t = t0 + pos < kv
            page = tl.load(BT + s * sbtb + (t0 // BS) * sbtp)
            kt = tl.load(K + page * skb + pos[None, :] * sks + h * skh
                         + od[:, None] * skd, mask=mask_t[None, :], other=0.0)
            vt = tl.load(V + page * svb + pos[:, None] * svs + h * svh
                         + od[None, :] * svd, mask=mask_t[:, None], other=0.0)
            sc = tl.where(mask_t[None, :], tl.dot(qv, kt) * scale,
                          float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
        m_i = m_new
    base = (s * HKV + h) * NSEG + g
    tl.store(SM + base * GQA + og, m_i)
    tl.store(SS + base * GQA + og, l_i)
    tl.store(SA + (base * GQA + og)[:, None] * D + od[None, :], acc)


def paged_op(q, k, v, out, bt, seq, scale, nseg, bufs, warps=1):
    B = q.shape[0]
    HKV = k.shape[2]
    gqa = q.shape[1] // HKV
    _seg_paged[(nseg, HKV, B)](
        q, k, v, bufs[0], bufs[1], bufs[2], bt, seq,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        bt.stride(0), bt.stride(1),
        HQ=q.shape[1], HKV=HKV, D=q.shape[2], BS=k.shape[1], GQA=gqa,
        NSEG=nseg, scale=scale * LOG2E, num_warps=warps, num_stages=1)
    _combine_kernel[(B, HKV)](
        bufs[0], bufs[1], bufs[2], out,
        out.stride(0), out.stride(1), out.stride(2),
        HQ=q.shape[1], HKV=HKV, GQA=gqa, D=q.shape[2], NSEG=nseg, num_warps=4)


def bench(fn, n=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def case(KV, B, ragged=False, NSEG=32):
    HQ, HKV, D, BS = 16, 2, 128, 16
    NB = KV // BS
    nb = B * NB
    torch.manual_seed(0)
    q = torch.randn(B, HQ, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(nb, BS, HKV, D, dtype=torch.bfloat16, device="cuda") * 0.1
    v = torch.randn(nb, BS, HKV, D, dtype=torch.bfloat16, device="cuda") * 0.1
    bt = torch.randperm(nb, dtype=torch.int32, device="cuda").view(B, NB).contiguous()
    if ragged:
        seq = torch.tensor([max(BS, KV - (i * 137) % 1500) for i in range(B)],
                           dtype=torch.int32, device="cuda")
    else:
        seq = torch.full((B,), KV, dtype=torch.int32, device="cuda")
    gqa = HQ // HKV
    rows = B * HKV * NSEG
    bufs = (torch.empty((rows, gqa, D), dtype=torch.float32, device="cuda"),
            torch.empty((rows, gqa), dtype=torch.float32, device="cuda"),
            torch.empty((rows, gqa), dtype=torch.float32, device="cuda"))
    o1 = torch.empty_like(q)
    o2 = torch.empty_like(q)
    op(q, k, v, o1, bt, seq, D ** -0.5, num_segments=NSEG, block_n=BS,
       num_warps=1, seg_buffers=bufs)
    paged_op(q, k, v, o2, bt, seq, D ** -0.5, NSEG, bufs)
    torch.cuda.synchronize()
    d = (o1.float() - o2.float()).abs().max().item()
    t1 = bench(lambda: op(q, k, v, o1, bt, seq, D ** -0.5, num_segments=NSEG,
                          block_n=BS, num_warps=1, seg_buffers=bufs))
    best = None
    for w in (1, 2):
        t2 = bench(lambda: paged_op(q, k, v, o2, bt, seq, D ** -0.5, NSEG, bufs, w))
        d2 = (o1.float() - o2.float()).abs().max().item()
        print(f"    page-aligned w{w}: {t2:.3f} ms  delta {t2/t1-1:+.1%}  maxdiff {d2:.4f}",
              flush=True)
        if best is None or t2 < best[0]:
            best = (t2, w, d2)
    t2, wbest, d2 = best
    kv_gb = B * KV * HKV * D * 2 * 2 / 1e9
    tag = "ragged" if ragged else "equal"
    print(f"[kv={KV} B={B} {tag}] current {t1:.3f} ms ({kv_gb/t1*1e3/1e3:4.0f} GB/s) | "
          f"page-aligned {t2:.3f} ms ({kv_gb/t2*1e3/1e3:4.0f} GB/s) | "
          f"delta {t2/t1-1:+.1%} | maxdiff {d2:.4f} | warps={wbest}", flush=True)


case(4096, 64)
case(4096, 64, ragged=True)
case(16384, 32)
case(16384, 32, ragged=True)
case(4096, 8)
