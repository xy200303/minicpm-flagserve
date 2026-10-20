"""Last cheap knobs on the segment kernel: eviction/cache hints, page-aligned
loop (one page per iteration, no per-token div/mod), and maxnreg."""
import time

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _seg(
    Q, K, V, SA, SM, SS, BT, SL,
    sqb, sqh, sqd, skb, sks, skh, skd, svb, svs, svh, svd, sbtb, sbtp,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
    GQA: tl.constexpr, BN: tl.constexpr, NSEG: tl.constexpr,
    HINT: tl.constexpr, PAGED: tl.constexpr, scale,
):
    g = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    od = tl.arange(0, D)
    og = tl.arange(0, GQA)
    kv = tl.load(SL + s)
    if PAGED:
        # round the segment up to whole pages so each iteration is one page
        units = (kv + NSEG * BS - 1) // (NSEG * BS)
        start = g * units * BS
        end = tl.minimum(start + units * BS, kv)
    else:
        seg = (kv + NSEG - 1) // NSEG
        start = g * seg
        end = tl.minimum(start + seg, kv)
    m_i = tl.full((GQA,), float("-inf"), tl.float32)
    l_i = tl.zeros((GQA,), tl.float32)
    acc = tl.zeros((GQA, D), tl.float32)
    qv = tl.load(Q + s * sqb + (h * GQA + og)[:, None] * sqh + od[None, :] * sqd)
    n_full = (end - start) // BN * BN
    main_end = start + n_full
    for tok0 in range(start, main_end, BN):
        if PAGED:
            page = tl.load(BT + s * sbtb + (tok0 // BS) * sbtp)
            page_v = page
            pos = tl.arange(0, BN)
        else:
            ot = tok0 + tl.arange(0, BN)
            page = tl.load(BT + s * sbtb + (ot // BS) * sbtp)
            page_v = page
            pos = ot % BS
        kp = (K + page[None, :] * skb + pos[None, :] * sks + h * skh
              + od[:, None] * skd)
        vp = (V + page_v[:, None] * svb + pos[:, None] * svs + h * svh
              + od[None, :] * svd)
        if HINT == 0:
            kt = tl.load(kp)
            vt = tl.load(vp)
        elif HINT == 1:
            kt = tl.load(kp, eviction_policy="evict_first")
            vt = tl.load(vp, eviction_policy="evict_first")
        else:
            kt = tl.load(kp, cache_modifier=".cg")
            vt = tl.load(vp, cache_modifier=".cg")
        sc = tl.dot(qv, kt) * scale
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
        m_i = m_new
    if main_end < end:
        ot = main_end + tl.arange(0, BN)
        mask_t = ot < end
        page = tl.load(BT + s * sbtb + (ot // BS) * sbtp, mask=mask_t, other=0)
        pos = ot % BS
        kp = (K + page[None, :] * skb + pos[None, :] * sks + h * skh
              + od[:, None] * skd)
        vp = (V + page[:, None] * svb + pos[:, None] * svs + h * svh
              + od[None, :] * svd)
        sc = tl.dot(qv, tl.load(kp, mask=mask_t[None, :], other=0.0)) * scale
        sc = tl.where(mask_t[None, :], sc, float("-inf"))
        vt = tl.load(vp, mask=mask_t[:, None], other=0.0)
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


B, KV, HQ, HKV, D, BS, NSEG, BN = 64, 4096, 16, 2, 128, 16, 32, 16
NB = KV // BS
nb = B * NB
torch.manual_seed(0)
q = torch.randn(B, HQ, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(nb, BS, HKV, D, dtype=torch.bfloat16, device="cuda") * 0.1
v = torch.randn(nb, BS, HKV, D, dtype=torch.bfloat16, device="cuda") * 0.1
bt = torch.randperm(nb, dtype=torch.int32, device="cuda").view(B, NB).contiguous()
sl = torch.full((B,), KV, dtype=torch.int32, device="cuda")
rows = B * HKV * NSEG
sa = torch.empty((rows, HQ // HKV, D), dtype=torch.float32, device="cuda")
sm = torch.empty((rows, HQ // HKV), dtype=torch.float32, device="cuda")
ss = torch.empty((rows, HQ // HKV), dtype=torch.float32, device="cuda")
ref_acc = None
kv_gb = B * KV * HKV * D * 2 * 2 / 1e9


def run(hint=0, paged=False, maxnreg=None, warps=1):
    kw = dict(HQ=HQ, HKV=HKV, D=D, BS=BS, GQA=HQ // HKV, BN=BN, NSEG=NSEG,
              HINT=hint, PAGED=paged, scale=D ** -0.5 * LOG2E,
              num_warps=warps, num_stages=1)
    if maxnreg:
        kw["maxnreg"] = maxnreg
    _seg[(NSEG, HKV, B)](q, k, v, sa, sm, ss, bt, sl,
                         q.stride(0), q.stride(1), q.stride(2),
                         k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                         v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                         bt.stride(0), bt.stride(1), **kw)


def bench(fn, n=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


run(0, False)
torch.cuda.synchronize()
base = sa.clone()
t0 = bench(lambda: run(0, False))
print(f"baseline                     {t0:.3f} ms  {kv_gb/(t0/1e3):4.0f} GB/s", flush=True)

for name, kwargs in (
    ("evict_first", dict(hint=1)),
    ("cache .cg", dict(hint=2)),
    ("page-aligned loop", dict(paged=True)),
    ("maxnreg=64", dict(maxnreg=64)),
    ("maxnreg=96", dict(maxnreg=96)),
    ("maxnreg=128", dict(maxnreg=128)),
    ("warps=2", dict(warps=2)),
):
    try:
        run(**kwargs)
        torch.cuda.synchronize()
        d = (sa - base).abs().max().item()
        t = bench(lambda: run(**kwargs))
        print(f"{name:28s} {t:.3f} ms  {kv_gb/(t/1e3):4.0f} GB/s  "
              f"delta={t/t0-1:+.1%}  maxdiff_acc={d:.2e}", flush=True)
    except Exception as e:
        print(f"{name:28s} ERR {type(e).__name__}: {str(e)[:100]}", flush=True)
