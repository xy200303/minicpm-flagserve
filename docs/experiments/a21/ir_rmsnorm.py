# SPDX-License-Identifier: Apache-2.0
"""fl opt-A21: route RMSNorm to a fused Triton kernel via vLLM's IR op registry.

vLLM 0.24 dispatches layer norm through ``vllm.ir.op``; the engine config we
run shows ``ir_op_priority: rms_norm=['native'], fused_add_rms_norm=['native']``,
and the "native" implementation is an eager torch chain:

    x = x.float(); x = x + residual.float(); residual = x.to(bf16);
    var = x.pow(2).mean(-1); x = x * rsqrt(var + eps); x = x.to(bf16) * weight

i.e. ~8-10 kernels plus a fp32 intermediate per call, twice per layer.  For a
42-layer decode step that is ~84 calls.

This patch registers ``flag_gems.fused.fused_add_rms_norm_exact`` as an IR op
provider and puts it ahead of "native".  The kernel reproduces the reference
bit-for-bit (maxdiff 0.0 on both the output and the residual, measured on
BI-V150) and costs 47.4 us vs 181.5 us per [64, 2048] bf16 call.

A/B switch: VLLM_FL_DISABLE_IR_RMSNORM=1 keeps the native implementation.
"""

import os

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_DISABLED = os.environ.get("VLLM_FL_DISABLE_IR_RMSNORM", "0") == "1"
_PROVIDER = "flag_gems_iluvatar"
_applied = False


def _supported(
    x: torch.Tensor,
    x_residual: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    if x_residual is None or weight is None or variance_size is not None:
        return False
    if x.dtype not in (torch.bfloat16, torch.float16):
        return False
    if x.shape != x_residual.shape or x.dtype != x_residual.dtype:
        return False
    if x.stride(-1) != 1 or weight.ndim != 1:
        return False
    return True


def register() -> None:
    """Register the fused kernel as an IR op provider (idempotent)."""
    global _applied
    if _DISABLED or _applied:
        return
    try:
        from flag_gems.fused.fused_add_rms_norm_exact import (
            fused_add_rms_norm_exact,
        )
        from vllm.ir.ops.layernorm import fused_add_rms_norm as _ir_fused
    except Exception as exc:  # pragma: no cover
        logger.warning("ir_rmsnorm: unavailable (%s)", exc)
        return

    # The kernel must be opaque to torch.compile: vLLM's IR lowering inlines the
    # provider body into the FX graph, and a bare Triton launch would then be
    # traced (data_ptr on FakeTensor).  Wrap it exactly like vLLM's own impls,
    # which call torch.ops._C.* from inside the provider.
    @torch.library.custom_op(
        "vllm_fl_iluvatar::fused_add_rms_norm",
        mutates_args={"x", "x_residual"},
    )
    def _fused_add_rms_norm_op(
        x: torch.Tensor,
        x_residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        return fused_add_rms_norm_exact(x, x_residual, weight, epsilon)

    @_fused_add_rms_norm_op.register_fake
    def _fake(x, x_residual, weight, epsilon):
        return torch.empty_like(x)

    def _impl(
        x: torch.Tensor,
        x_residual: torch.Tensor,
        weight: torch.Tensor | None,
        epsilon: float,
        variance_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if variance_size is not None:  # not used by this model
            raise NotImplementedError
        assert weight is not None
        out = _fused_add_rms_norm_op(x, x_residual, weight, epsilon)
        return out, x_residual

    try:
        _ir_fused.register_impl(
            _PROVIDER,
            supported=True,
            supports_args=_supported,
            inplace=True,
        )(_impl)
    except Exception as exc:  # pragma: no cover
        logger.warning("ir_rmsnorm: registration failed (%s)", exc)
        return

    _applied = True
    logger.info("fl iluvatar: RMSNorm provider registered (%s, bit-exact vs "
                "native); priority is set by PlatformFL.get_default_ir_op_priority",
                _PROVIDER)
