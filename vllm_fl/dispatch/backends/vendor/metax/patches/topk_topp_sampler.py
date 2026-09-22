# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# vLLM's Triton topk_topp kernel fails to compile on MetaX
# (PassManager::run failed in ttgir stage). FlagGems now provides a
# MetaX-compatible sort-free implementation (flag_gems.fused.top_k_top_p);
# route apply_top_k_top_p to it, keeping the PyTorch path as a safety net.

import torch
import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler

try:
    from flag_gems.fused.top_k_top_p import (
        apply_top_k_top_p as _apply_top_k_top_p_flaggems,
    )
except Exception:  # FlagGems without the fused kernel: stay on PyTorch
    _apply_top_k_top_p_flaggems = None


def _apply_top_k_top_p_metax(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits
    if _apply_top_k_top_p_flaggems is not None:
        try:
            return _apply_top_k_top_p_flaggems(logits, k, p)
        except Exception:
            pass
    return topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)


# Replace the dispatch function with the MetaX-compatible kernel
topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p_metax
