"""Step-by-step debug of the plugin's W8 path against a torch reference."""
import torch

import vllm_fl.dispatch.backends.vendor.iluvatar.impl.w8_mlp as w8

torch.manual_seed(0)
K, N = 2048, 12288
M = 64
w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.02
x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

qmax = w.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-8)
sw_ref = (qmax / 127.0).view(-1)
qw_ref = (w.float() / qmax).round().clamp(-127, 127)

layer = torch.nn.Module()
layer.weight = torch.nn.Parameter(w)
entry = w8._prepare(layer)

print("sw match:", torch.equal(entry["sw"], sw_ref.contiguous()))
print("qw match:", torch.equal(entry["qwt"], qw_ref.t().contiguous().to(torch.int8)))

qx, amax = w8._quant_act(x)
torch.cuda.synchronize()
print("amax:", amax.item(), "ref amax:", x.abs().max().item())
qref = (x.float() / (x.abs().max().float() / 127.0)).round().clamp(-127, 127).to(torch.int8)
print("qx match:", torch.equal(qx, qref), "q diff max:",
      (qx.float() - qref.float()).abs().max().item())

acc = torch._int_mm(qx, entry["qwt"])
acc_ref = torch._int_mm(qref, qw_ref.t().contiguous().to(torch.int8))
print("acc match:", torch.equal(acc, acc_ref))

out = w8._dequant(acc, entry["sw"], amax)
torch.cuda.synchronize()
ref = (acc.float() * (amax / 127.0) * sw_ref[None, :]).to(torch.bfloat16)
print("out vs formula maxdiff:", (out.float() - ref.float()).abs().max().item())

y = x @ w.t()
print("formula vs exact matmul maxdiff:",
      (ref.float() - y.float()).abs().max().item())
# per-row check of the worst row
d = (out.float() - y.float()).abs()
i = d.argmax().item() // N
print("worst row", i, "row maxdiff", d[i].max().item(),
      "row |y| max", y[i].float().abs().max().item())
