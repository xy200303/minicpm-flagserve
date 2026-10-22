"""Is int8 GEMM actually faster than bf16 on BI-V150 at decode shapes?"""
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

    # int8: per-tensor dynamic activation scale (bitsandbytes style)
    xi = x.abs().amax().item() / 127.0
    wi = w.abs().amax().item() / 127.0
    qx = (x.float() / xi).round().clamp(-127, 127).to(torch.int8)
    qw = (w.float() / wi).round().clamp(-127, 127).to(torch.int8)
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    try:
        t_int_mm = bench(lambda: torch._int_mm(qx, qw.t()))
        ok_int = True
    except Exception as e:
        ok_int = False
        print("int_mm failed:", str(e)[:120])

    res = {}
    for name, fn in (
        ("bnb_qgemm", lambda: ixformer.bnb_qgemm(
            qx, qw, torch.tensor([xi], device="cuda"),
            torch.tensor([wi], device="cuda"))),
    ):
        try:
            y = fn()
            torch.cuda.synchronize()
            t = bench(fn)
            d = (y.float() * xi * wi - y_ref.float()).abs().max().item()
            res[name] = (t, d)
        except Exception as e:
            print(name, "failed:", str(e)[:140])

    print(f"N={N:5d} K={K}: bf16 {t_bf16*1e3:7.1f} us ({w_gb/(t_bf16/1e3):4.0f} GB/s)"
          + (f" | int_mm {t_int_mm*1e3:7.1f} us ({w_gb/2/(t_int_mm/1e3):4.0f} GB/s int8)"
             if ok_int else ""), flush=True)
    for name, (t, d) in res.items():
        print(f"           {name}: {t*1e3:7.1f} us ({w_gb/2/(t/1e3):4.0f} GB/s int8) "
              f"maxdiff vs bf16 {d:.4f}", flush=True)
