"""Ablate the shipped kernel config (NSEG=32, bn=16, w1) layer by layer."""
import time, torch, triton, triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _k(Q, K, V, SA, SM, SS, BT, SL,
       sqb, sqh, sqd, skb, sks, skh, skd, svb, svs, svh, svd, sbtb, sbtp,
       HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
       GQA: tl.constexpr, BN: tl.constexpr, NSEG: tl.constexpr, MODE: tl.constexpr,
       scale):
    g = tl.program_id(0); h = tl.program_id(1); s = tl.program_id(2)
    od = tl.arange(0, D); og = tl.arange(0, GQA)
    kv = tl.load(SL + s)
    seg = (kv + NSEG - 1) // NSEG
    start = g * seg; end = tl.minimum(start + seg, kv)
    m_i = tl.full((GQA,), float("-inf"), tl.float32)
    l_i = tl.zeros((GQA,), tl.float32)
    acc = tl.zeros((GQA, D), tl.float32)
    qv = tl.load(Q + s * sqb + (h * GQA + og)[:, None] * sqh + od[None, :] * sqd)
    for tok0 in range(start, end, BN):
        ot = tok0 + tl.arange(0, BN)
        page = tl.load(BT + s * sbtb + (ot // BS) * sbtp)
        pos = ot % BS
        kp = (K + page[None, :] * skb + pos[None, :] * sks + h * skh
              + od[:, None] * skd)
        vp = (V + page[:, None] * svb + pos[:, None] * svs + h * svh
              + od[None, :] * svd)
        kt = tl.load(kp)
        vt = tl.load(vp)
        if MODE == 0:  # loads only
            acc += (tl.sum(kt.to(tl.float32), 1)
                    + tl.sum(vt.to(tl.float32), 0))[None, :]
        else:
            sc = tl.dot(qv, kt) * scale
            if MODE == 1:  # + QK
                m_i = tl.maximum(m_i, tl.max(sc, 1))
                l_i += tl.sum(sc, 1)
            else:
                m_new = tl.maximum(m_i, tl.max(sc, 1))
                alpha = tl.exp2(m_i - m_new)
                p = tl.exp2(sc - m_new[:, None])
                l_i = l_i * alpha + tl.sum(p, 1)
                if MODE == 2:  # + softmax, no PV
                    acc = acc * alpha[:, None]
                else:
                    acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
                m_i = m_new
    base = (s * HKV + h) * NSEG + g
    tl.store(SM + base * GQA + og, m_i)
    tl.store(SS + base * GQA + og, l_i)
    tl.store(SA + (base * GQA + og)[:, None] * D + od[None, :], acc)


B, KV, HQ, HKV, D, BS = 64, 4096, 16, 2, 128, 16
NB = KV // BS; nb = B * NB
torch.manual_seed(0)
q = torch.randn(B, HQ, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(nb, BS, HKV, D, dtype=torch.bfloat16, device="cuda") * 0.1
v = torch.randn(nb, BS, HKV, D, dtype=torch.bfloat16, device="cuda") * 0.1
bt = torch.randperm(nb, dtype=torch.int32, device="cuda").view(B, NB).contiguous()
sl = torch.full((B,), KV, dtype=torch.int32, device="cuda")
NSEG, BN, W = 32, 16, 1
rows = B * HKV * NSEG
sa = torch.empty((rows, HQ // HKV, D), dtype=torch.float32, device="cuda")
sm = torch.empty((rows, HQ // HKV), dtype=torch.float32, device="cuda")
ss = torch.empty((rows, HQ // HKV), dtype=torch.float32, device="cuda")
kv_gb = B * KV * HKV * D * 2 * 2 / 1e9


def run(mode):
    _k[(NSEG, HKV, B)](q, k, v, sa, sm, ss, bt, sl,
                       q.stride(0), q.stride(1), q.stride(2),
                       k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                       v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                       bt.stride(0), bt.stride(1),
                       HQ=HQ, HKV=HKV, D=D, BS=BS, GQA=HQ // HKV, BN=BN,
                       NSEG=NSEG, MODE=mode, scale=D ** -0.5 * LOG2E,
                       num_warps=W, num_stages=1)


def bench(fn, n=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3


for mode, name in ((0, "loads only"), (1, "+QK"), (2, "+softmax"), (3, "+PV = full")):
    run(mode)
    torch.cuda.synchronize()
    t = bench(lambda: run(mode))
    print(f"{name:12s} {t:.3f} ms/layer  {kv_gb/(t/1e3):4.0f} GB/s", flush=True)
