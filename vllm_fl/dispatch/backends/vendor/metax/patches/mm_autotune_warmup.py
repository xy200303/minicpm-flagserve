# SPDX-License-Identifier: Apache-2.0
# 2026 - MiniCPM-FlagServe: MetaX GEMM autotune warmup at model load.

# FlagGems' LibTuner benchmarks every pruned config the first time an
# (M, N, K) autotune bucket is seen.  On a cold cache that cost lands inside
# benchmark-critical execute_model calls (observed: single 23.8 s stalls and
# degraded first benchmark rounds).  This patch replays the exact GEMM call
# paths once at load time so all runtime buckets are tuned before the server
# starts accepting traffic.  Startup takes the one-time tuning cost instead;
# scored request path stays tuning-free.  Disable with VLLM_FL_MM_WARMUP=0.

import os
import time

import torch
from vllm_fl.worker.model_runner import ModelRunnerFL

_FL_WARMUP = os.getenv("VLLM_FL_MM_WARMUP", "1") == "1"
# Small-M buckets exercised through F.linear (nt / nt_db decode path).
_FL_WARMUP_SMALL_M = (1, 2, 4, 8, 16, 32, 64, 128, 1024)
# Prefill buckets: multiples of 128 up to max_num_batched_tokens + one bucket
# of slack for decode-mixed chunks (align128 rounding in FlagGems libtuner).
_FL_WARMUP_PREFILL_STEP = 128
_FL_WARMUP_PREFILL_MIN = 256
_FL_WARMUP_FALLBACK_MAX_B = 2048

_orig_load_model = ModelRunnerFL.load_model


def _unique_linear_weights(model):
    seen = {}
    for module in model.modules():
        w_nn = getattr(module, "_fl_w_nn", None)
        if w_nn is None or w_nn.dim() != 2:
            continue
        # layer.weight may be an empty shell after the nt copy is released
        w = getattr(module, "weight", None)
        key = (tuple(w_nn.shape), str(w_nn.dtype))
        if key not in seen:
            seen[key] = (w, w_nn)
    return list(seen.values())


def _warmup_mm_autotune(runner):
    weights = _unique_linear_weights(runner.model)
    if not weights:
        print("[fl-warmup] no nn-repacked linear weights found; skip", flush=True)
        return
    try:
        max_b = int(runner.vllm_config.scheduler_config.max_num_batched_tokens)
    except Exception:
        max_b = _FL_WARMUP_FALLBACK_MAX_B
    prefill_m = list(
        range(_FL_WARMUP_PREFILL_MIN, max_b + _FL_WARMUP_PREFILL_STEP,
              _FL_WARMUP_PREFILL_STEP)
    )
    n_calls = len(weights) * (len(_FL_WARMUP_SMALL_M) + len(prefill_m))
    print(
        f"[fl-warmup] mm autotune warmup start: {len(weights)} unique weight "
        f"shapes, small M={list(_FL_WARMUP_SMALL_M)}, "
        f"prefill M {prefill_m[0]}..{prefill_m[-1]} step "
        f"{_FL_WARMUP_PREFILL_STEP} ({n_calls} calls)",
        flush=True,
    )
    t0 = time.time()
    torch.cuda.empty_cache()  # make released nt-layout storage visible to mem_get_info
    with torch.no_grad():
        for w, w_nn in weights:
            k, n = w_nn.shape  # w_nn is [K, N]
            for m in _FL_WARMUP_SMALL_M:
                x = torch.empty(m, k, dtype=w_nn.dtype, device=w_nn.device)
                if w is not None and w.numel() > 0:
                    torch.nn.functional.linear(x, w)
                torch.mm(x, w_nn)  # nn_db skinny-M path
                del x
            for m in prefill_m:
                x = torch.empty(m, k, dtype=w_nn.dtype, device=w_nn.device)
                torch.mm(x, w_nn)
                del x
    torch.cuda.synchronize()
    print(f"[fl-warmup] mm autotune warmup done in {time.time() - t0:.1f}s", flush=True)


def _load_model_metax(self, load_dummy_weights: bool = False):
    _orig_load_model(self, load_dummy_weights)
    if _FL_WARMUP and not load_dummy_weights:
        try:
            _warmup_mm_autotune(self)
        except Exception as exc:  # never let warmup break serving
            print(f"[fl-warmup] warmup skipped due to error: {exc}", flush=True)


ModelRunnerFL.load_model = _load_model_metax
