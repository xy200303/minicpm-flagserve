# SPDX-License-Identifier: Apache-2.0
# 2026 - MiniCPM-FlagServe: MetaX large-M GEMM layout co-optimization.

# FlagGems' metax mm_kernel_nn ([K,N] row-major weight) is ~1.35-2x faster
# than the nt path at prefill shapes (M >= ~256), because the MACA Triton
# async-load pipeline works for nn but not nt.  This patch repacks linear
# weights once at load time into nn layout and routes large-M calls through
# torch.mm(x, W_nn); skinny-M (decode) keeps the original F.linear path,
# which lands on the FlagGems nt_db kernel.  Math and dtypes are unchanged;
# both layouts coexist (weights are read-only after load).

import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
import vllm.model_executor.layers.linear as linear_mod
import vllm.model_executor.layers.utils as layer_utils

_FL_NN_MIN_M = 192
_FL_NN_MAX_N = 16384  # huge-N (lm_head) keeps its existing path
_FL_NN_MIN_N = 512

_orig_pw = UnquantizedLinearMethod.process_weights_after_loading


def _process_weights_after_loading_metax(self, layer):
    _orig_pw(self, layer)
    w = getattr(layer, "weight", None)
    if (
        w is not None
        and w.dim() == 2
        and w.dtype in (torch.bfloat16, torch.float16)
        and w.is_cuda
        and w.stride(1) == 1
        and w.stride(0) == w.shape[1]
        and _FL_NN_MIN_N <= w.shape[0] <= _FL_NN_MAX_N
        and w.shape[1] % 128 == 0
    ):
        layer._fl_w_nn = w.data.t().contiguous()


def _metax_unquantized_gemm(layer, x, weight, bias):
    wn = getattr(layer, "_fl_w_nn", None)
    if wn is not None and bias is None and x.dim() >= 2:
        m = x.numel() // x.shape[-1]
        if m > _FL_NN_MIN_M:
            out = torch.mm(x.reshape(-1, x.shape[-1]), wn)
            return out.view(*x.shape[:-1], wn.shape[1])
    return torch.nn.functional.linear(x, weight, bias)


def _dispatch_unquantized_gemm_metax():
    return _metax_unquantized_gemm


UnquantizedLinearMethod.process_weights_after_loading = (
    _process_weights_after_loading_metax
)
layer_utils.dispatch_unquantized_gemm = _dispatch_unquantized_gemm_metax
linear_mod.dispatch_unquantized_gemm = _dispatch_unquantized_gemm_metax
