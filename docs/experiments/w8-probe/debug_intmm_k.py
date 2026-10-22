"""Is torch._int_mm numerically correct at K=2048 (the real MLP K)?"""
import torch

torch.manual_seed(0)
for M, K, N in ((64, 256, 64), (64, 2048, 64), (64, 2048, 256), (64, 2048, 12288)):
    a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-127, 127, (N, K), dtype=torch.int8, device="cuda")
    ref = a.float() @ b.float().t()
    out = torch._int_mm(a, b.t().contiguous())
    torch.cuda.synchronize()
    d = (out.float() - ref).abs().max().item()
    scale = ref.abs().max().item()
    print(f"M={M} K={K} N={N}: maxdiff {d:.0f} (ref max {scale:.0f}) "
          f"{'OK' if d == 0 else 'WRONG'}", flush=True)
