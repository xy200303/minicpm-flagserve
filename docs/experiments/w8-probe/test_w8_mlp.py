"""Validate the W8 gate_up path against the stock bf16 linear, then time both."""
import time

import torch

from vllm.model_executor.layers.linear import MergedColumnParallelLinear
import vllm_fl.dispatch.backends.vendor.iluvatar.impl.w8_mlp as w8


def bench(fn, n=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


torch.manual_seed(0)
K, N = 2048, 12288
layer = torch.nn.Module()
layer.weight = torch.nn.Parameter(
    torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.02)
layer.bias = None
entry = w8._prepare(layer)

x = torch.randn(64, K, dtype=torch.bfloat16, device="cuda")
ref = x @ layer.weight.t()

qx, sx = w8._quant_act(x)  # sx is a device tensor
acc = torch._int_mm(qx, entry["qwt"])
out = w8._dequant(acc, entry["sw"], sx)
torch.cuda.synchronize()
d = (out.float() - ref.float()).abs()
rel = (d / ref.float().abs().clamp(min=1e-6)).mean().item()
print(f"maxdiff {d.max().item():.4f}  mean {d.mean().item():.5f}  "
      f"mean-rel {rel:.4f}", flush=True)


def stock():
    return x @ layer.weight.t()


def w8_path():
    q, s = w8._quant_act(x)
    return w8._dequant(torch._int_mm(q, entry["qwt"]), entry["sw"], s)


t_s = bench(stock)
t_w = bench(w8_path)
print(f"stock bf16 {t_s*1e3:7.1f} us | w8 {t_w*1e3:7.1f} us | x{t_s/t_w:.2f} | "
      f"per-step (x42) saving {(t_s-t_w)*42:.2f} ms", flush=True)
