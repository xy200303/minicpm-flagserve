"""Compare three ways to finish the decode attention:

  A. current: segment kernel + combine kernel (one big [GQA, NSEG, D] tile)
  B. chunked combine (small tiles, two passes over max/sum)
  C. single launch: last-CTA-per-(seq,head) does the combine inline

Prints per-layer timings and bitwise/abs differences against A.
"""
import time

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
CH = 8


@triton.jit
def _segment_kernel(
    Q, K, V, SEG_ACC, SEG_MAX, SEG_SUM, BT, SEQ_LEN,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_btb, stride_btp,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
    GQA: tl.constexpr, BLOCK_N: tl.constexpr,
    NSEG: tl.constexpr, scale,
):
    pid_g = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, GQA)
    kvlen = tl.load(SEQ_LEN + pid_s)
    seg_len = (kvlen + NSEG - 1) // NSEG
    start = pid_g * seg_len
    end = tl.minimum(start + seg_len, kvlen)
    m_i = tl.full((GQA,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((GQA,), dtype=tl.float32)
    acc = tl.zeros((GQA, D), dtype=tl.float32)
    qv = tl.load(Q + pid_s * stride_qb + (pid_h * GQA + offs_g)[:, None] * stride_qh
                 + offs_d[None, :] * stride_qd)
    n_full = (end - start) // BLOCK_N * BLOCK_N
    main_end = start + n_full
    for tok0 in range(start, main_end, BLOCK_N):
        offs_t = tok0 + tl.arange(0, BLOCK_N)
        page = tl.load(BT + pid_s * stride_btb + (offs_t // BS) * stride_btp)
        pos = offs_t % BS
        k_ptrs = (K + page[None, :] * stride_kb + pos[None, :] * stride_ks
                  + pid_h * stride_kh + offs_d[:, None] * stride_kd)
        vt_ptrs = (V + page[:, None] * stride_vb + pos[:, None] * stride_vs
                   + pid_h * stride_vh + offs_d[None, :] * stride_vd)
        s = tl.dot(qv, tl.load(k_ptrs)) * scale
        vt = tl.load(vt_ptrs)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
        m_i = m_new
    if main_end < end:
        offs_t = main_end + tl.arange(0, BLOCK_N)
        mask_t = offs_t < end
        page = tl.load(BT + pid_s * stride_btb + (offs_t // BS) * stride_btp,
                       mask=mask_t, other=0)
        pos = offs_t % BS
        k_ptrs = (K + page[None, :] * stride_kb + pos[None, :] * stride_ks
                  + pid_h * stride_kh + offs_d[:, None] * stride_kd)
        vt_ptrs = (V + page[:, None] * stride_vb + pos[:, None] * stride_vs
                   + pid_h * stride_vh + offs_d[None, :] * stride_vd)
        s = tl.dot(qv, tl.load(k_ptrs, mask=mask_t[None, :], other=0.0)) * scale
        s = tl.where(mask_t[None, :], s, float("-inf"))
        vt = tl.load(vt_ptrs, mask=mask_t[:, None], other=0.0)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
        m_i = m_new
    base = (pid_s * HKV + pid_h) * NSEG + pid_g
    tl.store(SEG_MAX + base * GQA + offs_g, m_i)
    tl.store(SEG_SUM + base * GQA + offs_g, l_i)
    tl.store(SEG_ACC + (base * GQA + offs_g)[:, None] * D + offs_d[None, :], acc)


@triton.jit
def _combine_big(SEG_ACC, SEG_MAX, SEG_SUM, OUT,
                 stride_ob, stride_oh, stride_od,
                 HQ: tl.constexpr, HKV: tl.constexpr, GQA: tl.constexpr,
                 D: tl.constexpr, NSEG: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_g = tl.arange(0, GQA)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, NSEG)
    base = (pid_s * HKV + pid_h) * NSEG
    mx = tl.load(SEG_MAX + (base + offs_n)[None, :] * GQA + offs_g[:, None])
    sm = tl.load(SEG_SUM + (base + offs_n)[None, :] * GQA + offs_g[:, None])
    m = tl.max(mx, 1)
    w = tl.exp2(mx - m[:, None])
    denom = tl.sum(w * sm, 1)
    accs = tl.load(SEG_ACC + ((base + offs_n)[None, :, None] * GQA
                              + offs_g[:, None, None]) * D + offs_d[None, None, :])
    o = tl.sum(accs * w[:, :, None], 1) / denom[:, None]
    tl.store(OUT + pid_s * stride_ob + (pid_h * GQA + offs_g)[:, None] * stride_oh
             + offs_d[None, :] * stride_od, o.to(OUT.dtype.element_ty))


@triton.jit
def _combine_chunked(SEG_ACC, SEG_MAX, SEG_SUM, OUT,
                     stride_ob, stride_oh, stride_od,
                     HQ: tl.constexpr, HKV: tl.constexpr, GQA: tl.constexpr,
                     D: tl.constexpr, NSEG: tl.constexpr, CHUNK: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_g = tl.arange(0, GQA)
    offs_d = tl.arange(0, D)
    base = (pid_s * HKV + pid_h) * NSEG
    # pass 1: global max over segments (cheap, only [GQA, NSEG])
    m = tl.full((GQA,), float("-inf"), dtype=tl.float32)
    for n0 in tl.static_range(0, NSEG, CHUNK):
        idx = n0 + tl.arange(0, CHUNK)
        mx = tl.load(SEG_MAX + (base + idx)[None, :] * GQA + offs_g[:, None])
        m = tl.maximum(m, tl.max(mx, 1))
    # pass 2: weights + denominator, then accumulate acc in chunks
    denom = tl.zeros((GQA,), dtype=tl.float32)
    o = tl.zeros((GQA, D), dtype=tl.float32)
    for n0 in tl.static_range(0, NSEG, CHUNK):
        idx = n0 + tl.arange(0, CHUNK)
        mx = tl.load(SEG_MAX + (base + idx)[None, :] * GQA + offs_g[:, None])
        sm = tl.load(SEG_SUM + (base + idx)[None, :] * GQA + offs_g[:, None])
        w = tl.exp2(mx - m[:, None])
        denom += tl.sum(w * sm, 1)
        accs = tl.load(SEG_ACC + ((base + idx)[None, :, None] * GQA
                                  + offs_g[:, None, None]) * D
                       + offs_d[None, None, :])
        o += tl.sum(accs * w[:, :, None], 1)
    o = o / denom[:, None]
    tl.store(OUT + pid_s * stride_ob + (pid_h * GQA + offs_g)[:, None] * stride_oh
             + offs_d[None, :] * stride_od, o.to(OUT.dtype.element_ty))


@triton.jit
def _segment_combine(SEG_ACC, SEG_MAX, SEG_SUM, OUT, CNT,
                     Q, K, V, BT, SEQ_LEN,
                     stride_qb, stride_qh, stride_qd,
                     stride_kb, stride_ks, stride_kh, stride_kd,
                     stride_vb, stride_vs, stride_vh, stride_vd,
                     stride_btb, stride_btp, stride_ob, stride_oh, stride_od,
                     HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                     BS: tl.constexpr, GQA: tl.constexpr, BLOCK_N: tl.constexpr,
                     NSEG: tl.constexpr, CHUNK: tl.constexpr, scale):
    pid_g = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, GQA)
    kvlen = tl.load(SEQ_LEN + pid_s)
    seg_len = (kvlen + NSEG - 1) // NSEG
    start = pid_g * seg_len
    end = tl.minimum(start + seg_len, kvlen)
    m_i = tl.full((GQA,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((GQA,), dtype=tl.float32)
    acc = tl.zeros((GQA, D), dtype=tl.float32)
    qv = tl.load(Q + pid_s * stride_qb + (pid_h * GQA + offs_g)[:, None] * stride_qh
                 + offs_d[None, :] * stride_qd)
    n_full = (end - start) // BLOCK_N * BLOCK_N
    main_end = start + n_full
    for tok0 in range(start, main_end, BLOCK_N):
        offs_t = tok0 + tl.arange(0, BLOCK_N)
        page = tl.load(BT + pid_s * stride_btb + (offs_t // BS) * stride_btp)
        pos = offs_t % BS
        k_ptrs = (K + page[None, :] * stride_kb + pos[None, :] * stride_ks
                  + pid_h * stride_kh + offs_d[:, None] * stride_kd)
        vt_ptrs = (V + page[:, None] * stride_vb + pos[:, None] * stride_vs
                   + pid_h * stride_vh + offs_d[None, :] * stride_vd)
        s = tl.dot(qv, tl.load(k_ptrs)) * scale
        vt = tl.load(vt_ptrs)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
        m_i = m_new
    if main_end < end:
        offs_t = main_end + tl.arange(0, BLOCK_N)
        mask_t = offs_t < end
        page = tl.load(BT + pid_s * stride_btb + (offs_t // BS) * stride_btp,
                       mask=mask_t, other=0)
        pos = offs_t % BS
        k_ptrs = (K + page[None, :] * stride_kb + pos[None, :] * stride_ks
                  + pid_h * stride_kh + offs_d[:, None] * stride_kd)
        vt_ptrs = (V + page[:, None] * stride_vb + pos[:, None] * stride_vs
                   + pid_h * stride_vh + offs_d[None, :] * stride_vd)
        s = tl.dot(qv, tl.load(k_ptrs, mask=mask_t[None, :], other=0.0)) * scale
        s = tl.where(mask_t[None, :], s, float("-inf"))
        vt = tl.load(vt_ptrs, mask=mask_t[:, None], other=0.0)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
        m_i = m_new
    base = (pid_s * HKV + pid_h) * NSEG + pid_g
    tl.store(SEG_MAX + base * GQA + offs_g, m_i)
    tl.store(SEG_SUM + base * GQA + offs_g, l_i)
    tl.store(SEG_ACC + (base * GQA + offs_g)[:, None] * D + offs_d[None, :], acc)

    # publish, then the last CTA of this (seq, head) finishes the reduction
    old = tl.atomic_add(CNT + pid_s * HKV + pid_h, 1, sem="acq_rel", scope="gpu")
    if (old % NSEG) == (NSEG - 1):
        base0 = (pid_s * HKV + pid_h) * NSEG
        m = tl.full((GQA,), float("-inf"), dtype=tl.float32)
        for n0 in tl.static_range(0, NSEG, CHUNK):
            idx = n0 + tl.arange(0, CHUNK)
            mx = tl.load(SEG_MAX + (base0 + idx)[None, :] * GQA + offs_g[:, None])
            m = tl.maximum(m, tl.max(mx, 1))
        denom = tl.zeros((GQA,), dtype=tl.float32)
        o = tl.zeros((GQA, D), dtype=tl.float32)
        for n0 in tl.static_range(0, NSEG, CHUNK):
            idx = n0 + tl.arange(0, CHUNK)
            mx = tl.load(SEG_MAX + (base0 + idx)[None, :] * GQA + offs_g[:, None])
            sm = tl.load(SEG_SUM + (base0 + idx)[None, :] * GQA + offs_g[:, None])
            w = tl.exp2(mx - m[:, None])
            denom += tl.sum(w * sm, 1)
            accs = tl.load(SEG_ACC + ((base0 + idx)[None, :, None] * GQA
                                      + offs_g[:, None, None]) * D
                           + offs_d[None, None, :])
            o += tl.sum(accs * w[:, :, None], 1)
        o = o / denom[:, None]
        tl.store(OUT + pid_s * stride_ob + (pid_h * GQA + offs_g)[:, None] * stride_oh
                 + offs_d[None, :] * stride_od, o.to(OUT.dtype.element_ty))


def bench(fn, n=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def make(KV, B, NSEG=32, BN=16, W=1):
    HQ, HKV, D, BS = 16, 2, 128, 16
    NB = KV // BS
    nb = B * NB
    torch.manual_seed(0)
    q = torch.randn(B, HQ, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(nb, BS, HKV, D, dtype=torch.bfloat16, device="cuda") * 0.1
    v = torch.randn(nb, BS, HKV, D, dtype=torch.bfloat16, device="cuda") * 0.1
    bt = torch.randperm(nb, dtype=torch.int32, device="cuda").view(B, NB).contiguous()
    seq = torch.full((B,), KV, dtype=torch.int32, device="cuda")
    gqa = HQ // HKV
    rows = B * HKV * NSEG
    bufs = (torch.empty((rows, gqa, D), dtype=torch.float32, device="cuda"),
            torch.empty((rows, gqa), dtype=torch.float32, device="cuda"),
            torch.empty((rows, gqa), dtype=torch.float32, device="cuda"))
    cnt = torch.zeros(B * HKV, dtype=torch.int32, device="cuda")
    out = torch.empty_like(q)
    sargs = (q, k, v, bufs[0], bufs[1], bufs[2], bt, seq,
             q.stride(0), q.stride(1), q.stride(2),
             k.stride(0), k.stride(1), k.stride(2), k.stride(3),
             v.stride(0), v.stride(1), v.stride(2), v.stride(3),
             bt.stride(0), bt.stride(1))
    kw = dict(HQ=HQ, HKV=HKV, D=D, BS=BS, GQA=gqa, BLOCK_N=BN, NSEG=NSEG,
              scale=D ** -0.5 * LOG2E, num_warps=W, num_stages=1)
    return q, k, v, out, bufs, cnt, sargs, kw, (B, HKV, NSEG, gqa, D)


def run_a(st):
    q, k, v, out, bufs, cnt, sargs, kw, dims = st
    B, HKV, NSEG, gqa, D = dims
    _segment_kernel[(NSEG, HKV, B)](*sargs, **kw)
    _combine_big[(B, HKV)](bufs[0], bufs[1], bufs[2], out,
                           out.stride(0), out.stride(1), out.stride(2),
                           HQ=kw["HQ"], HKV=HKV, GQA=gqa, D=D, NSEG=NSEG,
                           num_warps=4)


def run_b(st):
    q, k, v, out, bufs, cnt, sargs, kw, dims = st
    B, HKV, NSEG, gqa, D = dims
    _segment_kernel[(NSEG, HKV, B)](*sargs, **kw)
    _combine_chunked[(B, HKV)](bufs[0], bufs[1], bufs[2], out,
                               out.stride(0), out.stride(1), out.stride(2),
                               HQ=kw["HQ"], HKV=HKV, GQA=gqa, D=D, NSEG=NSEG,
                               CHUNK=CH, num_warps=4)


def run_c(st):
    q, k, v, out, bufs, cnt, sargs, kw, dims = st
    B, HKV, NSEG, gqa, D = dims
    _segment_combine[(NSEG, HKV, B)](
        bufs[0], bufs[1], bufs[2], out, cnt, q, k, v, sargs[6], sargs[7],
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        sargs[19], sargs[20], out.stride(0), out.stride(1), out.stride(2),
        HQ=kw["HQ"], HKV=HKV, D=D, BS=kw["BS"], GQA=gqa,
        BLOCK_N=kw["BLOCK_N"], NSEG=NSEG, CHUNK=CH, scale=kw["scale"],
        num_warps=kw["num_warps"], num_stages=kw["num_stages"])


def run_a_w(st, w):
    q, k, v, out, bufs, cnt, sargs, kw, dims = st
    B, HKV, NSEG, gqa, D = dims
    _segment_kernel[(NSEG, HKV, B)](*sargs, **kw)
    _combine_big[(B, HKV)](bufs[0], bufs[1], bufs[2], out,
                           out.stride(0), out.stride(1), out.stride(2),
                           HQ=kw["HQ"], HKV=HKV, GQA=gqa, D=D, NSEG=NSEG,
                           num_warps=w)


for KV, B in ((4096, 64), (16384, 32)):
    st = make(KV, B)
    out = st[3]
    run_a(st)
    torch.cuda.synchronize()
    ref = out.clone()
    ta = bench(lambda: run_a(st))
    run_b(st)
    torch.cuda.synchronize()
    db = (out.float() - ref.float()).abs().max().item()
    tb = bench(lambda: run_b(st))
    run_c(st)
    torch.cuda.synchronize()
    dc = (out.float() - ref.float()).abs().max().item()
    tc = bench(lambda: run_c(st))
    kv_gb = B * KV * 2 * 128 * 2 * 2 / 1e9
    ws = {}
    for w in (1, 2, 4, 8):
        run_a_w(st, w)
        torch.cuda.synchronize()
        ws[w] = bench(lambda: run_a_w(st, w))
    print(f"[kv={KV} B={B}] combine warps: " + " ".join(f"w{w}={t:.3f}" for w, t in ws.items()), flush=True)
    print(f"[kv={KV} B={B}] A two-kernel/big {ta:.3f} ms | "
          f"B chunked {tb:.3f} ms (diff {db:.2e}) | "
          f"C fused {tc:.3f} ms (diff {dc:.2e})  |  op GB/s A={kv_gb/(ta/1e3):.0f} "
          f"C={kv_gb/(tc/1e3):.0f}", flush=True)
