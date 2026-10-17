import time, torch, json
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
import vllm.v1.attention.ops.triton_unified_attention as tua
import dattn4

dev = "cuda"; B, KV, HQ, HKV, D, BS = 64, 4096, 16, 2, 128, 16
NB = KV // BS; nb = B * NB
torch.manual_seed(0)
q = torch.randn(B, HQ, D, dtype=torch.bfloat16, device=dev)
k = torch.randn(nb, BS, HKV, D, dtype=torch.bfloat16, device=dev)
v = torch.randn(nb, BS, HKV, D, dtype=torch.bfloat16, device=dev)
ref = torch.empty_like(q); mine = torch.empty_like(q)
seqused = torch.full((B,), KV, dtype=torch.int32, device=dev)
cu = torch.arange(0, B + 1, dtype=torch.int32, device=dev)
scale = D ** -0.5; kv_gb = B * KV * HKV * D * 2 * 2 / 1e9
so = torch.empty((64, HQ, 16, 128), dtype=torch.float32, device=dev)
sm = torch.empty((64, HQ, 16), dtype=torch.float32, device=dev)
se = torch.empty((64, HQ, 16), dtype=torch.float32, device=dev)

kern = tua.kernel_unified_attention
class Pin:
    def __getitem__(self, grid):
        fn = kern[grid]
        def call(**kw):
            if kw.get("IS_3D"):
                kw["num_warps"] = 1; kw["num_stages"] = 1
            return fn(**kw)
        return call
tua.kernel_unified_attention = Pin()

bt = torch.randperm(nb, dtype=torch.int32, device=dev).view(B, NB).contiguous()

def ref_call():
    unified_attention(q=q, k=k, v=v, out=ref, cu_seqlens_q=cu, max_seqlen_q=1,
        seqused_k=seqused, max_seqlen_k=KV, softmax_scale=scale, causal=True,
        window_size=(-1, -1), block_table=bt, softcap=0.0, q_descale=None,
        k_descale=None, v_descale=None, seq_threshold_3D=64,
        num_par_softmax_segments=16, softmax_segm_output=so,
        softmax_segm_max=sm, softmax_segm_expsum=se)

def bench(fn, n=20):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3

ref_call(); torch.cuda.synchronize()
t = bench(ref_call)
print(f"stock pinned {t:.3f} ms {kv_gb/(t/1e3):4.0f} GB/s", flush=True)

rows = []
bufs = None
cfgs = [(256,16,1,1,False),(256,16,1,1,True),(128,16,1,1,False),(256,16,1,2,False),
        (256,16,2,1,False),(512,16,1,1,True),(256,32,1,1,True),(1024,16,1,1,True)]
for seg, bn, w, st, gf in cfgs:
    try:
        bufs = dattn4.decode_attention(q, k, v, mine, bt, seqused, scale, seg_tok=seg,
                                       block_n=bn, num_warps=w, num_stages=st,
                                       grid_seg_first=gf, buffers=bufs)
        torch.cuda.synchronize()
        d = (mine.float() - ref.float()).abs().max().item()
        t2 = bench(lambda: dattn4.decode_attention(q, k, v, mine, bt, seqused, scale,
                    seg_tok=seg, block_n=bn, num_warps=w, num_stages=st,
                    grid_seg_first=gf, buffers=bufs))
        print(f"seg{seg:4d} bn{bn:2d} w{w} s{st} gf{int(gf)}: {t2:6.3f} ms "
              f"{kv_gb/(t2/1e3):4.0f} GB/s diff={d:.4f}", flush=True)
        rows.append((t2, seg, bn, w, st, gf, d))
    except Exception as e:
        print(f"seg{seg} bn{bn} w{w} s{st} gf{int(gf)} ERR {str(e)[:70]}", flush=True)
json.dump(rows, open("/workspace/experiments/bench_dattn4.json", "w"))
