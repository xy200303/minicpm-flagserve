import time, torch, json, itertools
from dattn_abl2 import run
dev="cuda"; B,KV,HQ,HKV,D,BS=64,4096,16,2,128,16
NB=KV//BS; num_blocks=B*NB
torch.manual_seed(0)
q=torch.randn(B,HQ,D,dtype=torch.bfloat16,device=dev)
k=torch.randn(num_blocks,BS,HKV,D,dtype=torch.bfloat16,device=dev)
v=torch.randn(num_blocks,BS,HKV,D,dtype=torch.bfloat16,device=dev)
bt=torch.randperm(num_blocks,dtype=torch.int32,device=dev).view(B,NB).contiguous()
seqused=torch.full((B,),KV,dtype=torch.int32,device=dev)
scale=D**-0.5; kv_gb=B*KV*HKV*D*2*2/1e9
def bench(fn,n=20):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e3
out={}
for mode,label in [(2,"softmax-noPV"),(3,"PV-mma"),(4,"PV-fma")]:
    rows=[]
    for seg,bn,w,st in itertools.product([256,512,1024],[32,64],[2,4],[1,2]):
        try:
            run(q,k,v,bt,seqused,scale,mode,seg_tok=seg,block_n=bn,num_warps=w,num_stages=st)
            torch.cuda.synchronize()
            t=bench(lambda: run(q,k,v,bt,seqused,scale,mode,seg_tok=seg,block_n=bn,num_warps=w,num_stages=st))
            rows.append((t,seg,bn,w,st))
        except Exception as e:
            print(label,seg,bn,w,st,"ERR",str(e)[:50], flush=True)
    rows.sort()
    for t,seg,bn,w,st in rows[:6]:
        print(f"{label:14s} seg{seg:4d} bn{bn:2d} w{w} s{st}: {t:6.3f} ms {kv_gb/(t/1e3):4.0f} GB/s", flush=True)
    out[label]=rows[:6]
json.dump(out, open("/workspace/experiments/bench_abl2.json","w"))
