"""Feasibility: does an int8 KV cache (per-token-head scale) actually speed up
the A20 decode kernel?  Compares bf16-KV vs int8-KV at kv=4096 / kv=16384."""
import time
import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _seg_q8(
    Q, K, V, KS, VS, SEG_ACC, SEG_MAX, SEG_SUM, BT, SEQ_LEN,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_sb, stride_ss, stride_sh,
    stride_btb, stride_btp,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
    GQA: tl.constexpr, BLOCK_N: tl.constexpr, NSEG: tl.constexpr, scale,
    MODE: tl.constexpr = 0,
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
        kp = (K + page[None, :] * stride_kb + pos[None, :] * stride_ks
              + pid_h * stride_kh + offs_d[:, None] * stride_kd)
        vp = (V + page[:, None] * stride_vb + pos[:, None] * stride_vs
              + pid_h * stride_vh + offs_d[None, :] * stride_vd)
        sp = (KS + page[None, :] * stride_sb + pos[None, :] * stride_ss
              + pid_h * stride_sh)
        ki = tl.load(kp)
        kscale = tl.load(sp)
        if MODE == 0:
            kt = (ki.to(tl.float32) * kscale).to(tl.bfloat16)
        elif MODE == 1:
            kt = ki.to(tl.bfloat16)
        else:
            kt = ki.to(tl.bfloat16) * kscale.to(tl.bfloat16)
        sv = (VS + page[:, None] * stride_sb + pos[:, None] * stride_ss
              + pid_h * stride_sh)
        vi = tl.load(vp)
        vscale = tl.load(sv)
        if MODE == 0:
            vt = (vi.to(tl.float32) * vscale).to(tl.bfloat16)
        elif MODE == 1:
            vt = vi.to(tl.bfloat16)
        else:
            vt = vi.to(tl.bfloat16) * vscale.to(tl.bfloat16)
        s = tl.dot(qv, kt) * scale
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
def _comb(
    SEG_ACC, SEG_MAX, SEG_SUM, OUT,
    stride_ob, stride_oh, stride_od,
    HQ: tl.constexpr, HKV: tl.constexpr, GQA: tl.constexpr, D: tl.constexpr,
    NSEG: tl.constexpr,
):
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
    accs = tl.load(SEG_ACC + ((base + offs_n)[None, :, None] * GQA + offs_g[:, None, None]) * D
                   + offs_d[None, None, :])
    out = tl.sum(accs * w[:, :, None], 1) / denom[:, None]
    tl.store(OUT + pid_s * stride_ob + (pid_h * GQA + offs_g)[:, None] * stride_oh
             + offs_d[None, :] * stride_od, out.to(OUT.dtype.element_ty))


def run(q, k, v, ks, vs, out, bt, seq, scale, nseg=32, block_n=16, warps=1, mode=0):
    B, HQ, D = q.shape
    HKV = k.shape[2]
    GQA = HQ // HKV
    BS = k.shape[1]
    dev = q.device
    rows = B * HKV * nseg
    acc = torch.empty((rows, GQA, D), dtype=torch.float32, device=dev)
    mx = torch.empty((rows, GQA), dtype=torch.float32, device=dev)
    sm = torch.empty((rows, GQA), dtype=torch.float32, device=dev)
    _seg_q8[(nseg, HKV, B)](
        q, k, v, ks, vs, acc, mx, sm, bt, seq,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        ks.stride(0), ks.stride(1), ks.stride(2),
        bt.stride(0), bt.stride(1),
        HQ=HQ, HKV=HKV, D=D, BS=BS, GQA=GQA, BLOCK_N=block_n, NSEG=nseg,
        scale=scale * LOG2E, MODE=mode, num_warps=warps, num_stages=1)
    _comb[(B, HKV)](acc, mx, sm, out,
                    out.stride(0), out.stride(1), out.stride(2),
                    HQ=HQ, HKV=HKV, GQA=GQA, D=D, NSEG=nseg, num_warps=4)


def bench(fn, n=20):
    for _ in range(4):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


for KV, B in ((4096, 64), (16384, 64)):
    HQ, HKV, D, BS = 16, 2, 128, 16
    n_pages = KV // BS
    nb = B * n_pages
    torch.manual_seed(0)
    q = torch.randn(B, HQ, D, dtype=torch.bfloat16, device="cuda")
    kb = torch.randn(nb, BS, HKV, D, dtype=torch.bfloat16, device="cuda")
    vb = torch.randn(nb, BS, HKV, D, dtype=torch.bfloat16, device="cuda")
    kb /= 8
    vb /= 8
    # per-token-head int8 quantization
    kscale = kb.abs().amax(-1, keepdim=True).float() / 127.0
    vscale = vb.abs().amax(-1, keepdim=True).float() / 127.0
    ki = (kb.float() / kscale).round().clamp(-127, 127).to(torch.int8)
    vi = (vb.float() / vscale).round().clamp(-127, 127).to(torch.int8)
    ks = kscale.squeeze(-1).contiguous()
    vs = vscale.squeeze(-1).contiguous()
    out = torch.empty_like(q)
    bt = torch.randperm(nb, dtype=torch.int32, device="cuda").view(B, n_pages).contiguous()
    seq = torch.full((B,), KV, dtype=torch.int32, device="cuda")
    scale = D ** -0.5
    kv_gb_bf16 = B * KV * HKV * D * 2 * 2 / 1e9
    kv_gb_int8 = kv_gb_bf16 / 2
    best = None
    for bn, w, mode in ((16, 1, 0), (32, 2, 0), (16, 1, 1), (32, 2, 1),
                        (16, 1, 2), (32, 2, 2)):
        try:
            run(q, ki, vi, ks, vs, out, bt, seq, scale, block_n=bn, warps=w, mode=mode)
            torch.cuda.synchronize()
            t = bench(lambda: run(q, ki, vi, ks, vs, out, bt, seq, scale,
                                  block_n=bn, warps=w, mode=mode))
            print(f"[kv={KV}] int8 bn={bn:3d} w={w} mode={mode}: {t:.3f} ms  "
                  f"{kv_gb_int8/(t/1e3):4.0f} GB/s  "
                  f"bf16ref={0.72 if KV == 4096 else 2.41:.2f}", flush=True)
            if best is None or t < best[0]:
                best = (t, bn, w, mode)
        except Exception as e:
            print(f"[kv={KV}] int8 bn={bn} w={w} mode={mode} ERR {str(e)[:80]}", flush=True)
    print(f"[kv={KV}] BEST int8 {best}", flush=True)
