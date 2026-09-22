#!/usr/bin/env python3
"""Correctness + perf test for flag_gems.fused.top_k_top_p on MetaX."""
import sys
import torch

sys.path.insert(0, "/workspace/FlagGems/src")
from flag_gems.fused.top_k_top_p import apply_top_k_top_p
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

torch.manual_seed(0)
dev = "cuda"


def make_logits(b, v, dtype):
    # realistic-ish sampler logits: a sharp peak over diffuse noise
    x = torch.randn(b, v, device=dev, dtype=torch.float32) * 2.0
    hot = torch.randint(0, v, (b, 50), device=dev)
    x.scatter_add_(1, hot, torch.rand(b, 50, device=dev) * 8.0)
    return x.to(dtype)


def compare(name, logits, k, p):
    ref = apply_top_k_top_p_pytorch(logits.clone().float(), k, p)
    got = apply_top_k_top_p(logits.clone(), k, p).float()
    kept_ref = torch.isfinite(ref)
    kept_got = torch.isfinite(got)
    inter = (kept_ref & kept_got).sum(1).float()
    union = (kept_ref | kept_got).sum(1).float().clamp(min=1)
    iou = (inter / union).mean().item()
    pr = ref.softmax(-1).nan_to_num()
    pg = got.softmax(-1).nan_to_num()
    tv = 0.5 * (pr - pg).abs().sum(1).mean().item()
    print(f"{name:<28} kept_ref={kept_ref.sum(1).float().mean():7.1f} "
          f"kept_got={kept_got.sum(1).float().mean():7.1f} IoU={iou:.4f} TV={tv:.5f}")
    return iou, tv


B, V = 64, 130560
results = []
x = make_logits(B, V, torch.float32)
p95 = torch.full((B,), 0.95, device=dev)
p90 = torch.full((B,), 0.90, device=dev)
p100 = torch.full((B,), 1.0, device=dev)
k50 = torch.full((B,), 50, device=dev, dtype=torch.int32)
k_big = torch.full((B,), V, device=dev, dtype=torch.int32)

results.append(compare("p=0.95 fp32", x, None, p95))
results.append(compare("p=0.90 fp32", x, None, p90))
results.append(compare("k=50 fp32", x, k50, None))
results.append(compare("k=50+p=0.90 fp32", x, k50, p90))
results.append(compare("p=1.0 fp32", x, None, p100))
results.append(compare("k=vocab fp32", x, k_big, None))
xb = make_logits(B, V, torch.bfloat16)
results.append(compare("p=0.95 bf16", xb, None, p95))
results.append(compare("none", x, None, None))

# ---- perf ----
def bench(fn, *args, iters=50):
    for _ in range(5):
        fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn(*args)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


xl = make_logits(B, V, torch.float32)
t_ref = bench(lambda: apply_top_k_top_p_pytorch(xl.clone(), None, p95))
t_got = bench(lambda: apply_top_k_top_p(xl.clone(), None, p95))
print(f"\nperf [64x130560 p=0.95 fp32]: pytorch={t_ref:.3f} ms  ours={t_got:.3f} ms  speedup={t_ref/t_got:.1f}x")

t_ref2 = bench(lambda: apply_top_k_top_p_pytorch(xl, None, p95))
t_got2 = bench(lambda: apply_top_k_top_p(xl, None, p95))
print(f"perf in-place reuse          : pytorch={t_ref2:.3f} ms  ours={t_got2:.3f} ms  speedup={t_ref2/t_got2:.1f}x")

bad = [n for n, (iou, tv) in zip(["p95","p90","k50","k50p90","p100","kbig","bf16","none"], results) if iou < 0.99 or tv > 0.01]
print("\nRESULT:", "FAIL " + str(bad) if bad else "ALL PASS")
