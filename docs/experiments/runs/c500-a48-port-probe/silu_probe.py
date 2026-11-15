import torch, statistics
def bench(fn, iters=30, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(iters):
        s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)*1000
import vllm
for M in (2048, 64):
    x=torch.randn(M, 24576, device='cuda', dtype=torch.bfloat16)
    out=torch.empty(M, 12288, device='cuda', dtype=torch.bfloat16)
    t=bench(lambda: torch.ops._C.silu_and_mul(out, x))
    b=M*(24576+12288)*2
    print(f'M={M}: silu_and_mul {t:.1f} us, {b/1e6:.1f} MB, {b/t/1e3:.0f} GB/s', flush=True)
