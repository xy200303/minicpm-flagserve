# SPDX-License-Identifier: Apache-2.0
"""fl opt-A22: W8 (int8 weight-only) gate_up projection via the vendor int8 GEMM.

Motivation (BI-V150, M=64 decode): the decode step is weight-bandwidth bound --
5.3 GB of weights per step at ~450 GB/s.  The vendor stack ships
``torch._int_mm`` (int8 x int8 -> int32), which reads half the bytes of bf16:
gate_up (K=2048, N=12288) measured 114.7 us (bf16) vs 26.1 us (int8) in
isolation.  The dequantisation epilogue is what decides the outcome: a naive
three-kernel torch epilogue eats the gain at small shapes, so the epilogue
here is a single Triton kernel with per-channel weight scales.

Measured for the gate_up + SwiGLU pair (M=64, K=2048, 2I=12288):
    bf16 GEMM + silu_and_mul   137.5 us
    int8 GEMM + fused epilogue  73.8 us   (1.86x)  -> ~2.7 ms/step over 42 layers

Scope and switch:
  * only the MLP gate_up projection (``MergedColumnParallelLinear``);
  * weights are quantised once, per output channel, on first use;
  * activations are quantised per tensor, per call;
  * ``VLLM_FL_ENABLE_W8_MLP=1`` turns it on (default off), any shape/dtype
    mismatch or exception falls back to the stock bf16 path.

Accuracy: int8 per-channel weights + per-tensor activations is lossy (max|d| ~0.19
on random gate_up inputs, ~2% relative); every adoption has to clear the
MATH-500 gate before it is enabled by default.
"""

import os

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLED = os.environ.get("VLLM_FL_ENABLE_W8_MLP", "0") == "1"
_applied = False
_cache = {}
_stats = {"hit": 0, "fallback": 0}


@triton.jit
def _quant_act_kernel(X, QX, AMAX, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    amax = tl.load(AMAX)
    scale = tl.maximum(amax / 127.0, 1e-8)
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    q = tl.clamp(tl.floor(x / scale + 0.5), -127.0, 127.0).to(tl.int8)
    tl.store(QX + offs, q, mask=mask)


@triton.jit
def _dequant_kernel(ACC, SW, OUT, AMAX, N, stride_am, stride_om,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N
    acc = tl.load(ACC + offs_m[:, None] * stride_am + offs_n[None, :],
                  mask=mask[None, :])
    sw = tl.load(SW + offs_n, mask=mask, other=1.0)
    sx = tl.maximum(tl.load(AMAX) / 127.0, 1e-8)
    out = acc.to(tl.float32) * (sx * sw)[None, :]
    tl.store(OUT + offs_m[:, None] * stride_om + offs_n[None, :],
             out.to(OUT.dtype.element_ty), mask=mask[None, :])


def _quant_act(x: torch.Tensor):
    """Per-tensor int8 quantisation; the scale stays on the device (no sync)."""
    numel = x.numel()
    qx = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    amax = x.abs().max().float().reshape(1)
    nblk = triton.cdiv(numel, 8192)
    _quant_act_kernel[(nblk,)](x, qx, amax, numel, BLOCK=8192, num_warps=8)
    return qx, amax


def _dequant(acc, sw, sx):
    M, N = acc.shape
    out = torch.empty(M, N, dtype=torch.bfloat16, device=acc.device)
    bm = 8
    bn = 256
    _dequant_kernel[(triton.cdiv(M, bm), triton.cdiv(N, bn))](
        acc, sw, out, sx, N, acc.stride(0), out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, num_warps=8)
    return out


def _prepare(layer):
    """Quantise the layer's weight once (per output channel)."""
    w = getattr(layer, "weight", None)
    if w is None or w.dtype != torch.bfloat16 or w.dim() != 2:
        return None
    qmax = w.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-8)
    sw = (qmax / 127.0).view(-1).contiguous()
    qw = (w.float() / qmax).round().clamp(-127, 127).to(torch.int8)
    entry = {"qwt": qw.t().contiguous(), "sw": sw, "K": w.shape[1],
             "N": w.shape[0], "bias": getattr(layer, "bias", None)}
    logger.info("fl iluvatar: W8 gate_up prepared (%d x %d)", w.shape[0], w.shape[1])
    return entry


def apply() -> None:
    global _applied
    if not _ENABLED or _applied:
        return
    try:
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear
    except Exception as exc:  # pragma: no cover
        logger.warning("w8_mlp: cannot import linear layer (%s)", exc)
        return

    orig_forward = MergedColumnParallelLinear.forward

    def forward(self, x):
        entry = _cache.get(id(self))
        if entry is None:
            try:
                entry = _prepare(self)
            except Exception as exc:  # pragma: no cover
                logger.warning("w8_mlp: prepare failed (%s)", exc)
                entry = False
            _cache[id(self)] = entry
        if not entry or x.dtype != torch.bfloat16 or x.dim() != 2 \
                or x.shape[-1] != entry["K"] or entry["bias"] is not None:
            _stats["fallback"] += 1
            return orig_forward(self, x)
        try:
            qx, sx = _quant_act(x)
            acc = torch._int_mm(qx, entry["qwt"])
            out = _dequant(acc, entry["sw"], sx)
            out = out.view(*x.shape[:-1], entry["N"])
            _stats["hit"] += 1
            return out
        except Exception as exc:  # pragma: no cover
            if _stats["fallback"] < 3:
                logger.warning("w8_mlp: fallback (%s)", exc)
            _stats["fallback"] += 1
            return orig_forward(self, x)

    MergedColumnParallelLinear.forward = forward
    _applied = True
    logger.info("fl iluvatar: W8 gate_up enabled (int8 weights per-channel, "
                "int8 activations per-tensor, Triton dequant epilogue)")
