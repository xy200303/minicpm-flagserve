"""Probe ixformer.batch_paged_attention for prefill-length queries (no de-page gather)."""
import time
import torch
import ixformer as ixf

dev = "cuda"
torch.manual_seed(0)

B, KV, HQ, HKV, D, PAGE = 4, 512, 16, 2, 128, 16
n_pages = (KV + PAGE - 1) // PAGE
scale = D ** -0.5

q = torch.randn(B * KV, HQ, D, dtype=torch.bfloat16, device=dev) * 0.1
k = torch.randn(B * n_pages * PAGE, HKV, D, dtype=torch.bfloat16, device=dev)
v = torch.randn(B * n_pages * PAGE, HKV, D, dtype=torch.bfloat16, device=dev)
kflat = k.view(-1, HKV, D)
vflat = v.view(-1, HKV, D)
kvc = (kflat, vflat)
total_pages = B * n_pages
bt = torch.zeros(B, total_pages, dtype=torch.int32, device=dev)
for i in range(B):
    bt[i, :n_pages] = torch.arange(i * n_pages, (i + 1) * n_pages, dtype=torch.int32, device=dev)
bt = bt.contiguous()
out = torch.empty_like(q)


def ref_single(i, qlen):
    qs = q[i * KV:i * KV + qlen].float()
    pages = bt[i, :n_pages].long()
    kk = torch.cat([kflat[p * PAGE:(p + 1) * PAGE] for p in pages])[:KV].float()
    vv = torch.cat([vflat[p * PAGE:(p + 1) * PAGE] for p in pages])[:KV].float()
    qg = qs.view(qlen, HKV, HQ // HKV, D)
    s = torch.einsum("qghd,khd->ghqk", qg, kk) * scale
    mask = torch.tril(torch.ones(qlen, KV, device=dev), diagonal=KV - qlen)
    s = s.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
    p = torch.softmax(s, dim=-1)
    o = torch.einsum("ghqk,khd->qghd", p, vv).reshape(qlen, HQ, D)
    return o


for qlen in (1, 128, KV):
    cu_q = torch.arange(0, (B + 1) * qlen, qlen, dtype=torch.int32, device=dev)
    cu_k = torch.arange(0, (B + 1) * KV, KV, dtype=torch.int32, device=dev)
    lse = torch.empty(B * qlen, HQ, dtype=torch.float32, device=dev)
    try:
        ixf.batch_paged_attention(
            out[: B * qlen], q[: B * qlen], kvc, lse,
            cu_q, cu_k, bt, cu_q.cpu(), cu_k.cpu(), bt.cpu(),
            B, total_pages, PAGE, HKV, scale, True)
        torch.cuda.synchronize()
        errs = []
        for i in range(B):
            r = ref_single(i, qlen)
            got = out[i * KV:i * KV + qlen].float()
            errs.append((got - r).abs().max().item())
        print(f"qlen={qlen:4d}: ok, maxdiff per req = "
              f"{['%.4f' % e for e in errs]}", flush=True)
    except Exception as e:
        print(f"qlen={qlen:4d}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
