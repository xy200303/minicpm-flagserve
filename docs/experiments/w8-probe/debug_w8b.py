"""Locate the 4.14 discrepancy: quantize/dequantize in torch, compare step by step."""
import torch

torch.manual_seed(0)
K, N, M = 2048, 4096, 64      # smaller N for fast float math, same K
w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.02
x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

y = x.float() @ w.float().t()
print(f"y: std {y.std().item():.4f} absmax {y.abs().max().item():.4f}")

# per-tensor activation quant / per-channel weight quant, done in torch
sx = x.abs().max().float() / 127.0
sw = w.abs().amax(dim=1).float().clamp(min=1e-8) / 127.0
qx = (x.float() / sx).round().clamp(-127, 127)
qw = (w.float() / sw[:, None]).round().clamp(-127, 127)
x_dq = qx * sx
w_dq = qw * sw[:, None]
print("x quant err max:", (x_dq - x.float()).abs().max().item(),
      " scale:", sx.item())
print("w quant err max:", (w_dq - w.float()).abs().max().item(),
      " scale:", sw.mean().item())

y_dq = x_dq @ w_dq.t()
print("y_dq vs y maxdiff:", (y_dq - y).abs().max().item())
print("y_dq vs y mean abs:", (y_dq - y).abs().mean().item(), " (y abs mean",
      y.abs().mean().item(), ")")

acc = torch._int_mm(qx.to(torch.int8), (qw.to(torch.int8)).t().contiguous())
y_acc = acc.float() * sx * sw[None, :]
print("y_acc vs y_dq maxdiff:", (y_acc - y_dq).abs().max().item())
print("y_acc vs y maxdiff:", (y_acc - y).abs().max().item())
