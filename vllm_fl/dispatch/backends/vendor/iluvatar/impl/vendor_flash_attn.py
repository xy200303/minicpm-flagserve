# SPDX-License-Identifier: Apache-2.0
"""Iluvatar: route prefill attention to the vendor ixAttnBkd flash kernel.

Stock path on BI-V150: vLLM's generic TRITON_ATTN backend whose
kernel_unified_attention costs ~36.8 ms per 2048-token chunk per layer at
16k context (measured: 91.8% of prefill GPU time).  CoreX torch ships
aten::_efficient_attention_forward, backed by Iluvatar's ixAttnBkd flash
library; with custom_mask_type=2 (bottom-right causal, the FA2 chunked-
prefill semantics) the same shape costs ~3.5 ms (10.5x faster).

Patch structure (two hooks, both in apply()):
  * TritonAttentionMetadataBuilder.build: compute the per-step
    decode/prefill request split ONCE per step from CPU-side metadata
    (query_start_loc_cpu, seq_lens_cpu_upper_bound) -- doing it per layer
    from GPU tensors would need 2 DtoH syncs x 42 layers per step
    (measured: ~270s of host stall over 83 steps).
  * TritonAttentionImpl.forward: pure-decode steps take the stock path
    untouched; steps with prefill rows run decode rows through the stock
    unified_attention on a sliced metadata and prefill rows through the
    vendor op on de-paged KV.  Gather targets persistent module-level
    pools (per-layer allocation on a nearly-full card triggered allocator
    sync storms, ~43ms host per layer).

Fallbacks: alibi / real sliding window / softcap / sinks / fp8 KV /
output scale / cascade / encoder / any exception -> stock forward.
VLLM_FL_DISABLE_ILUVATAR_VENDOR_FA=1: A/B switch back to stock.
"""

import dataclasses
import os
import time as _time

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_CMT_BOTTOM_RIGHT = 2  # xFormers CausalFromBottomRight (FA2 chunked semantics)
_applied = False
_orig_forward = None
_orig_build = None
_AttentionType = None
_engaged = {"n": 0, "fb": 0}
_T = {"calls": 0, "dec": 0.0, "pre": 0.0}

# VLLM_FL_DISABLE_ILUVATAR_VENDOR_FA=1: stock path (A/B switch)
_DISABLED = os.getenv("VLLM_FL_DISABLE_ILUVATAR_VENDOR_FA", "0") == "1"

# persistent de-paged KV gather pools (shared across layers and steps)
_POOL = {}


def _pool(name, rows, hkv, d, dtype, dev):
    key = (name, dtype)
    buf = _POOL.get(key)
    if buf is None or buf.shape[0] < rows:
        rows = max(rows, 2048)
        _POOL[key] = torch.empty(rows, hkv, d, dtype=dtype, device=dev)
        buf = _POOL[key]
    return buf


def _build_split(m, cu_l, seq_l):
    """Compute the decode/prefill request split from CPU lists, attach GPU
    tensors for the decode subset.  Runs once per step in the builder."""
    pre_idx, dec_idx = [], []
    for i in range(len(seq_l)):
        (dec_idx if cu_l[i + 1] - cu_l[i] == 1 else pre_idx).append(i)
    dev = m.query_start_loc.device
    if dec_idx:
        dec_rows = torch.cat([
            torch.arange(cu_l[i], cu_l[i + 1], device=dev, dtype=torch.long)
            for i in dec_idx
        ])
        dec_meta = dataclasses.replace(
            m,
            num_actual_tokens=len(dec_idx),
            max_query_len=1,
            query_start_loc=torch.arange(
                0, len(dec_idx) + 1, device=dev,
                dtype=m.query_start_loc.dtype),
            seq_lens=m.seq_lens[dec_idx],
            block_table=m.block_table[dec_idx],
        )
    else:
        dec_rows = None
        dec_meta = None
    m._fl_split = (cu_l, seq_l, pre_idx, dec_rows, dec_meta)


_STEP = {"t0": 0.0, "n": 0}

def _build_patched(self, common_prefix_len, common_attn_metadata,
                   fast_build=False, **kwargs):
    _t = _time.perf_counter()
    if _STEP["t0"] > 0:
        _STEP["n"] += 1
        dt = _t - _STEP["t0"]
        if _STEP["n"] % 20 == 0:
            logger.info("fl iluvatar pace: step=%d dt=%.1fms", _STEP["n"], dt * 1e3)
    _STEP["t0"] = _t
    m = _orig_build(self, common_prefix_len, common_attn_metadata,
                    fast_build, **kwargs)
    try:
        cu_cpu = common_attn_metadata.query_start_loc_cpu
        seq_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        if (cu_cpu is not None and seq_cpu is not None
                and common_attn_metadata.max_query_len > 1):
            _build_split(m, cu_cpu.tolist(), seq_cpu.tolist())
    except Exception as e:
        logger.warning("fl iluvatar: split build failed (stock path): %s", e)
    return m


def _eligible(self, attn_metadata, kv_cache) -> bool:
    if attn_metadata is None or attn_metadata.max_query_len <= 1:
        return False
    # Crossover measured on BI-V150 (4k scenario pace A/B): the vendor op's
    # slow path costs ~0.22us/token/layer vs stock unified ~2.3us/token at
    # 16k but the patch host overhead (~100ms/mixed step) loses below ~8k.
    if attn_metadata.max_seq_len < 8192:
        return False
    if getattr(attn_metadata, "_fl_split", None) is None:
        return False
    if _AttentionType is not None and self.attn_type != _AttentionType.DECODER:
        return False
    if self.alibi_slopes is not None:
        return False
    sw = self.sliding_window
    if sw is not None and tuple(sw) not in ((-1, -1), (None, None)):
        return False
    if self.logits_soft_cap != 0.0 or getattr(self, "sinks", None) is not None:
        return False
    if getattr(self, "kv_cache_dtype", "auto") != "auto":
        return False
    if kv_cache.dtype not in (torch.bfloat16, torch.float16):
        return False
    causal = attn_metadata.causal
    if causal is not True and not (isinstance(causal, torch.Tensor) and bool(causal.all())):
        return False
    if attn_metadata.use_cascade:
        return False
    return True


def _vendor_forward(self, layer, query, key, value, kv_cache,
                    attn_metadata, output):
    key_cache, value_cache = kv_cache.unbind(1)  # [blocks, bs, hkv, d]
    bs = key_cache.shape[1]
    hkv, d = key_cache.shape[2], key_cache.shape[3]
    n_tok = attn_metadata.num_actual_tokens
    q = query[:n_tok]
    out = output[:n_tok]
    cu_l, seq_l, pre_idx, dec_rows, dec_meta = attn_metadata._fl_split

    # decode rows: stock unified attention on the sliced metadata
    _t1 = _time.perf_counter()
    if dec_meta is not None:
        q_dec = q.index_select(0, dec_rows)
        out_dec = torch.empty_like(q_dec)
        _orig_forward(self, layer, q_dec, key[:n_tok], value[:n_tok], kv_cache,
                      dec_meta, out_dec)
        out.index_copy_(0, dec_rows, out_dec)
    _T["dec"] += _time.perf_counter() - _t1
    _t2 = _time.perf_counter()

    # prefill rows: vendor flash per request on de-paged KV (pooled buffers)
    for i in pre_idx:
        q0, q1 = cu_l[i], cu_l[i + 1]
        kv = seq_l[i]
        n_pages = (kv + bs - 1) // bs
        pages = attn_metadata.block_table[i, :n_pages].long()
        Kp = _pool("K", n_pages * bs, hkv, d, key_cache.dtype, key_cache.device)
        Vp = _pool("V", n_pages * bs, hkv, d, value_cache.dtype, value_cache.device)
        torch.index_select(key_cache, 0, pages,
                           out=Kp[: n_pages * bs].view(-1, hkv, d))
        torch.index_select(value_cache, 0, pages,
                           out=Vp[: n_pages * bs].view(-1, hkv, d))
        K = Kp[:kv].unsqueeze(0)
        V = Vp[:kv].unsqueeze(0)
        oi = torch.ops.aten._efficient_attention_forward(
            q[q0:q1].unsqueeze(0), K, V,
            None, None, None, None, None,
            0.0, _CMT_BOTTOM_RIGHT, False,
            scale=self.scale,
        )[0]
        out[q0:q1] = oi.squeeze(0)
    _T["pre"] += _time.perf_counter() - _t2
    _T["calls"] += 1
    if _T["calls"] % 200 == 0:
        logger.info(
            "fl iluvatar timing: calls=%d dec=%.1fms pre=%.1fms (host, cumulative)",
            _T["calls"], _T["dec"] * 1e3, _T["pre"] * 1e3,
        )
    return output


def _forward_iluvatar(self, layer, query, key, value, kv_cache,
                      attn_metadata, output, output_scale=None,
                      output_block_scale=None):
    if output_scale is None and output_block_scale is None and not _DISABLED \
            and _eligible(self, attn_metadata, kv_cache):
        try:
            if _engaged["n"] == 0:
                logger.info("fl iluvatar: vendor flash prefill ENGAGED")
            _engaged["n"] += 1
            return _vendor_forward(self, layer, query, key, value, kv_cache,
                                   attn_metadata, output)
        except Exception:
            _engaged["fb"] += 1
            if _engaged["fb"] <= 3:
                logger.exception("fl iluvatar: vendor flash failed, fallback")
    return _orig_forward(self, layer, query, key, value, kv_cache,
                         attn_metadata, output, output_scale,
                         output_block_scale)


def apply() -> bool:
    global _applied, _orig_forward, _orig_build, _AttentionType
    if _applied:
        return True
    try:
        from vllm.v1.attention.backends.triton_attn import (
            TritonAttentionImpl,
            TritonAttentionMetadataBuilder,
        )
        from vllm.v1.attention.backend import AttentionType
    except Exception as e:
        logger.warning("fl iluvatar: vendor flash patch unavailable: %s", e)
        return False
    _AttentionType = AttentionType
    _orig_forward = TritonAttentionImpl.forward
    TritonAttentionImpl.forward = _forward_iluvatar
    _orig_build = TritonAttentionMetadataBuilder.build
    TritonAttentionMetadataBuilder.build = _build_patched
    _applied = True
    logger.info("fl iluvatar: vendor flash prefill patch armed "
                "(ixAttnBkd bottom-right causal, builder-side split)")
    return True
