"""bnb_qgemm (int8 GEMM + fused dequant) on BI-V150: speed and accuracy."""
import time

import torch
import ixformer


def bench(fn, n=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


M, K = 64, 2048
for N in (2048, 6144):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.02
    wt = w.t().contiguous()
    y_ref = x @ wt
    t_bf16 = bench(lambda: x @ wt)
    w_gb = N * K * 2 / 1e9

    # per-tensor scales, fp16 as the kernel requires
    xi = x.abs().amax().item() / 127.0
    wi = w.abs().amax().item() / 127.0
    qx = (x.float() / xi).round().clamp(-127, 127).to(torch.int8)
    qw = (w.float() / wi).round().clamp(-127, 127).to(torch.int8)
    SA = torch.tensor([xi], dtype=torch.float16, device="cuda")
    SW = torch.tensor([wi], dtype=torch.float16, device="cuda")

    variants = {}
    for name, fn in (
        ("bnb_qgemm", lambda: ixformer.bnb_qgemm(qx, qw, SA, SW)),
    ):
        try:
            y = fn()
            torch.cuda.synchronize()
            d = (y.float() - y_ref.float()).abs().max().item()
            rel = d / y_ref.float().abs().max().item()
            t = bench(fn)
            variants[name] = (t, d, rel, tuple(y.shape), y.dtype)
        except Exception as e:
            print(f"{name} failed: {str(e)[:120]}")

    print(f"N={N:5d}: bf16 {t_bf16*1e3:7.1f} us ({w_gb/(t_bf16/1e3):4.0f} GB/s) "
          f"[{w_gb/2:.2f} GB weights]", flush=True)
    for name, (t, d, rel, shp, dt) in variants.items():
        print(f"        {name}: {t*1e3:7.1f} us ({w_gb/2/(t/1e3):4.0f} GB/s int8) "
              f"x{t_bf16/t:.2f} | out {shp} {dt} | maxdiff {d:.4f} (rel {rel:.4f})",
              flush=True)
