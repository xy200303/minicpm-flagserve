"""W8 estimate: torch._int_mm + a dequant epilogue vs bf16, at real layer shapes."""
import time

import torch


def bench(fn, n=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


# MiniCPM5-2B (hidden 2048, inter 6144, 16 q / 2 kv heads, d 128, vocab 130560)
shapes = [
    ("qkv_proj", 64, 2048, 2560),
    ("o_proj", 64, 2048, 2048),
    ("gate_up", 64, 2048, 12288),
    ("down_proj", 64, 6144, 2048),
    ("lm_head", 64, 2048, 130560),
]

tot_bf16 = 0.0
tot_w8 = 0.0
print(f"{'matmul':10s} {'M':>4s} {'K':>6s} {'N':>7s} | {'bf16 us':>8s} {'w8 us':>8s} | "
      f"{'x':>5s} | {'bf16 GB/s':>9s} {'w8 GB/s':>8s}")
for name, M, K, N in shapes:
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.02
    wt = w.t().contiguous()

    xi = x.abs().amax().item() / 127.0
    wi = w.abs().amax(dim=1, keepdim=True).float() / 127.0     # per-output-channel
    qx = (x.float() / xi).round().clamp(-127, 127).to(torch.int8)
    qw = (w.float() / wi).round().clamp(-127, 127).to(torch.int8)
    qwt = qw.t().contiguous()
    sx = torch.tensor(xi, dtype=torch.float32, device="cuda")
    sw = wi.view(1, N).contiguous()

    def w8():
        acc = torch._int_mm(qx, qwt)                          # int32 [M, N]
        return (acc.float() * (sx * sw)).to(torch.bfloat16)

    t_bf16 = bench(lambda: x @ wt)
    t_w8 = bench(w8)
    gw = N * K * 2 / 1e9
    tot_bf16 += t_bf16
    tot_w8 += t_w8
    print(f"{name:10s} {M:4d} {K:6d} {N:7d} | {t_bf16*1e3:8.1f} {t_w8*1e3:8.1f} | "
          f"{t_bf16/t_w8:5.2f} | {gw/(t_bf16/1e3):9.0f} {gw/2/(t_w8/1e3):8.0f}", flush=True)

print(f"\nper layer (qkv+o+gate_up+down): bf16 {tot_bf16*1e3:.1f} us | "
      f"w8 {tot_w8*1e3:.1f} us -> x{tot_bf16/tot_w8:.2f}")
print("x42 layers + lm_head, per-step GEMM saving: "
      f"{(tot_bf16 - (shapes and 0)):.0f}")
