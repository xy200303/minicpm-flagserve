import torch, statistics, json

def bench(fn, iters=30, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    ts=[]
    for _ in range(iters):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)*1000  # us

shapes=[(2048,2560),(2048,2048),(2048,12288),(6144,2048)]
out={}
for M in (64,2048):
    out[M]=[]
    for K,N in shapes:
        a=torch.randn(M,K,device='cuda',dtype=torch.bfloat16)
        b=torch.randn(K,N,device='cuda',dtype=torch.bfloat16)
        t_bf16=bench(lambda: a@b)
        # naive per-token quant (upper bound; a fused kernel would be ~1 pass)
        def q():
            amax=a.abs().amax(dim=1,keepdim=True)
            scale=(amax/127.0).clamp_min(1e-8)
            ai=(a/scale).round().clamp(-127,127).to(torch.int8)
            return ai,scale
        ai,scale=q()
        bi=torch.randint(-127,127,(K,N),device='cuda',dtype=torch.int8)
        t_q=bench(q)
        t_int8=bench(lambda: torch._int_mm(ai,bi))
        c=torch._int_mm(ai,bi)
        t_deq=bench(lambda: c.to(torch.bfloat16))
        out[M].append(dict(K=K,N=N,bf16=round(t_bf16,1),quant=round(t_q,1),
                           int8=round(t_int8,1),deq=round(t_deq,1),
                           total8=round(t_q+t_int8+t_deq,1),
                           speedup=round(t_bf16/(t_q+t_int8+t_deq),2)))
print(json.dumps(out, indent=1))
