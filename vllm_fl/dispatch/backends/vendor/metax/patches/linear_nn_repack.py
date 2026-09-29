# SPDX-License-Identifier: Apache-2.0
# 2026 - MiniCPM-FlagServe: MetaX single nn-layout weights + skinny-M nn_db.

# FlagGems' metax mm_kernel_nn ([K,N] row-major weight) is ~1.35-2x faster
# than the nt path at prefill shapes (M >= ~256), because the MACA Triton
# async-load pipeline works for nn but not nt.  At decode shapes (M <= 128)
# the hand-written double-buffered mm_kernel_nn_db matches or beats nt_db
# (measured layer total ratio 0.982), so decode can use the same nn copy.
# This patch therefore repacks linear weights ONCE into nn layout, routes all
# M through torch.mm(x, W_nn), and releases the original (N,K) storage —
# no duplicate weight copy (~4GB returned to the KV pool on MiniCPM5-2B).
# Math and dtypes are unchanged.

import os

import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
import vllm.model_executor.layers.linear as linear_mod
import vllm.model_executor.layers.utils as layer_utils

_FL_NN_MAX_N = 16384  # huge-N (lm_head) keeps its existing path
_FL_NN_MIN_N = 512
_FL_FREE_NT = os.getenv("VLLM_FL_FREE_NT_WEIGHTS", "1") == "1"

_orig_pw = UnquantizedLinearMethod.process_weights_after_loading


def _process_weights_after_loading_metax(self, layer):
    _orig_pw(self, layer)
    w = getattr(layer, "weight", None)
    if (
        w is not None
        and w.dim() == 2
        and w.dtype in (torch.bfloat16, torch.float16)
        and w.is_cuda
        and w.numel() > 0
        and w.stride(1) == 1
        and w.stride(0) == w.shape[1]
        and _FL_NN_MIN_N <= w.shape[0] <= _FL_NN_MAX_N
        and w.shape[1] % 128 == 0
    ):
        layer._fl_w_nn = w.data.t().contiguous()
        if _FL_FREE_NT:
            # decode no longer needs the (N,K) copy; keep the Parameter shell
            # (shape metadata) but release the storage
            layer.weight.data = torch.empty(
                0, dtype=w.dtype, device=w.device
            )


def _metax_unquantized_gemm(layer, x, weight, bias):
    wn = getattr(layer, "_fl_w_nn", None)
    if wn is not None and bias is None and x.dim() >= 2:
        m = x.numel() // x.shape[-1]
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
