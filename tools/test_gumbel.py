#!/usr/bin/env python3
"""Statistical + perf validation for gumbel_max_sample."""
import sys
import torch

sys.path.insert(0, "/workspace/FlagGems/src")
from flag_gems.fused.gumbel_max_sample import gumbel_max_sample

dev = "cuda"
torch.manual_seed(0)
B, V = 64, 130560

# --- statistical test: sampled histogram vs softmax probs ---
logits = torch.randn(B, V, device=dev) * 2.0
hot = torch.randint(0, V, (B, 50), device=dev)
logits.scatter_add_(1, hot, torch.rand(B, 50, device=dev) * 6.0)
# mask like top-p does: keep top ~500 per row
thr = logits.topk(500, dim=-1).values[:, -1:]
masked = logits.masked_fill(logits < thr, float("-inf"))

N_SAMPLE = 2000
counts = torch.zeros(B, V, device=dev)
for i in range(N_SAMPLE):
    tok = gumbel_max_sample(masked.clone(), seed=1234)
    counts.scatter_add_(1, tok.unsqueeze(1), torch.ones(B, 1, device=dev))
freq = counts / N_SAMPLE
probs = masked.softmax(-1)
# per-row total variation distance between empirical freq and true probs
tv = 0.5 * (freq - probs).abs().sum(1)

# calibrate: same-N empirical TV of a known-correct sampler (torch.multinomial)
counts_ref = torch.zeros(B, V, device=dev)
g = torch.Generator(device=dev).manual_seed(999)
for i in range(N_SAMPLE):
    tok_ref = torch.multinomial(probs, 1, generator=g).squeeze(1)
    counts_ref.scatter_add_(1, tok_ref.unsqueeze(1), torch.ones(B, 1, device=dev))
tv_ref = 0.5 * (counts_ref / N_SAMPLE - probs).abs().sum(1)
print(f"statistical: ours TV mean={tv.mean().item():.4f} max={tv.max().item():.4f} | "
      f"torch.multinomial same-N TV mean={tv_ref.mean().item():.4f} max={tv_ref.max().item():.4f}")
print(f"calibrated verdict: {'OK (ours <= ref*1.2)' if tv.mean() <= tv_ref.mean() * 1.2 else 'SUSPECT'}")
# sanity: every sampled token must be in the kept set
bad = (~torch.isfinite(masked.gather(1, tok.unsqueeze(1)))).sum().item()
print(f"masked-set violation count: {bad} (must be 0)")

# --- determinism: same step/seed twice gives same tokens ---
a1 = gumbel_max_sample(masked.clone(), seed=7)
# reset counter to force same step
import importlib
gm = importlib.import_module("flag_gems.fused.gumbel_max_sample")
gm._STEP_COUNTER["value"] -= 1
a2 = gumbel_max_sample(masked.clone(), seed=7)
print(f"determinism: identical={torch.equal(a1, a2)}")
a3 = gumbel_max_sample(masked.clone(), seed=7)  # next step
print(f"next step differs: {(a1 != a3).any().item()} (should be True)")

# --- edge: single unmasked token ---
edge = torch.full((4, V), float("-inf"), device=dev)
edge[:, 42] = 1.0
tok_edge = gumbel_max_sample(edge)
print(f"single-token edge: {(tok_edge == 42).all().item()} (must be True)")

# --- perf: fused vs eager tail ---
def timeit(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3

lg = torch.randn(B, V, device=dev, dtype=torch.float32)
def eager_tail():
    probs = lg.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty_like(probs)
    q.exponential_()
    return probs.div(q).argmax(dim=-1)

t_eager = timeit(eager_tail)
t_fused = timeit(lambda: gumbel_max_sample(lg))
print(f"perf [64x130560 fp32]: eager_tail={t_eager:.0f} us  fused={t_fused:.0f} us  speedup={t_eager/t_fused:.1f}x")
