# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused residual-add + RMSNorm that matches a fp32 reference bit-for-bit.

Why this exists: vLLM's IR ``native`` implementation runs the whole chain in
eager torch (cast to fp32, add, cast back, pow, mean, rsqrt, mul, cast, mul --
~8-10 kernels per call).  For a decode step of a 42-layer model that is ~84
calls, i.e. several hundred kernels and a fp32 intermediate per call.

This kernel does the same arithmetic in one pass and reproduces the reference
exactly (bit-identical output and residual measured on BI-V150):

    s            = x.float() + residual.float()
    residual_out = s.to(dtype)                      # written in place
    var          = mean(s * s)                      # fp32 accumulation
    out          = (s * rsqrt(var + eps)).to(dtype) * weight

Note that the reference casts to the weight dtype *before* multiplying by the
weight; that detail is what keeps the result bit-exact, and it is why the
generic low-precision ``flag_gems.fused_add_rms_norm`` (which accumulates the
variance in low precision) differs by ~8 ulp and is not used here.

Measured on BI-V150, [64, 2048] bf16, per call: 181.5 us -> 47.4 us.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rms_norm_kernel(
    OUT, X, RES, W,
    stride_o, stride_x, stride_r,
    H: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(RES + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
    s = x + r

    var = tl.sum(tl.where(mask, s * s, 0.0), 0) / H
    xr = s * tl.rsqrt(var + EPS)

    w = tl.load(W + cols, mask=mask, other=0.0)
    out = xr.to(w.dtype) * w

    tl.store(RES + row * stride_r + cols, s.to(RES.dtype.element_ty), mask=mask)
    tl.store(OUT + row * stride_o + cols, out.to(OUT.dtype.element_ty), mask=mask)


def fused_add_rms_norm_exact(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``out`` = rms_norm(x + residual) * weight, residual updated in place.

    Args:
        x: ``[..., hidden]`` contiguous activation.
        residual: same shape/dtype as ``x``; overwritten with ``x + residual``.
        weight: ``[hidden]`` in the activation dtype.
        eps: variance epsilon.
        out: optional destination for the normalized output.

    Returns:
        The normalized tensor (``out``).
    """
    assert x.shape == residual.shape and x.stride(-1) == 1
    hidden = x.shape[-1]
    rows = x.numel() // hidden
    if out is None:
        out = torch.empty_like(x)
    x2 = x.view(rows, hidden)
    r2 = residual.view(rows, hidden)
    o2 = out.view(rows, hidden)
    _fused_add_rms_norm_kernel[(rows,)](
        o2, x2, r2, weight,
        o2.stride(0), x2.stride(0), r2.stride(0),
        H=hidden, BLOCK=triton.next_power_of_2(hidden), EPS=eps,
        num_warps=4,
    )
    return out
