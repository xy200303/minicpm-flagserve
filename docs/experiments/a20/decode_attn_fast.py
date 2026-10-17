# SPDX-License-Identifier: Apache-2.0
"""fl opt-A20: self-developed paged decode-attention kernel (BI-V150).

Stock TRITON_ATTN decode (3D flash-decoding path, A7-pinned w1s1) costs
1.04 ms per layer at the official 4k shape (B=64, kv=4096, 16 q / 2 kv
heads, d=128, block=16, bf16) -- 258 GB/s, well under the 587 GB/s this
card sustains on a plain copy.  Ablating that kernel showed loads+QK
already run at 509 GB/s (0.54 ms) and softmax is free (0.02 ms); the
remaining 0.36 ms is the PV `tl.dot`, which must convert `p` from the MMA
accumulator layout into an A-operand.

Changes here:
  * one warp per (seq, kv_head, 256-token segment).  With num_warps=1 the
    accumulator->operand conversion stays inside a warp and needs no
    shared-memory round trip (num_warps=2 was 16% slower, 4/8 worse);
  * BLOCK_N=16 so the p tile never exceeds one MMA shape (32/64 lost
    12-30%);
  * base-2 online softmax (scale folded with log2(e)); exp2 hits the MUFU
    path and plain exp() measured 10% slower;
  * the segment loop runs unmasked and only the ragged tail block is
    masked -- masking every block cost +0.25 ms (0.72 -> 0.98).

Measured (kv=4096, random block tables, 20 reps): 0.72 ms/layer
(372 GB/s) vs 1.04 ms stock, max|diff| 2e-4 vs the stock kernel, and
0.66 ms vs 0.90 ms at ragged kv lengths.

CUDA-graph safety: the launch grid uses a fixed segment count (like stock's
num_par_softmax_segments) and the segment length is derived from seq_lens
on the device, so nothing about the launch depends on the host-side kv
length.  Scratch lives in one slab allocated when the patch is installed,
i.e. before graph capture, so no allocation happens inside a graph.

A/B switch:
  VLLM_FL_DISABLE_DECODE_FAST=1        -> stock unified attention
  VLLM_FL_DECODE_FAST_SEGMENTS/_BLOCK/_WARPS/_STAGES -> kernel shape knobs
  VLLM_FL_DECODE_FAST_DEBUG=1          -> log eligibility once per reason
"""

import os

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger

logger = init_logger(__name__)

LOG2E = 1.4426950408889634

_DISABLED = os.environ.get("VLLM_FL_DISABLE_DECODE_FAST", "0") == "1"
_DEBUG = os.environ.get("VLLM_FL_DECODE_FAST_DEBUG", "0") == "1"
_NSEG = int(os.environ.get("VLLM_FL_DECODE_FAST_SEGMENTS", "32"))
_BLOCK_N = int(os.environ.get("VLLM_FL_DECODE_FAST_BLOCK", "16"))
_NUM_WARPS = int(os.environ.get("VLLM_FL_DECODE_FAST_WARPS", "1"))
_NUM_STAGES = int(os.environ.get("VLLM_FL_DECODE_FAST_STAGES", "1"))

_HEAD_SIZE = 128
_NUM_Q_HEADS = 16
_NUM_KV_HEADS = 2
_GQA = _NUM_Q_HEADS // _NUM_KV_HEADS
# Scratch slab: max sequences per decode step.
_BMAX = int(os.environ.get("VLLM_FL_DECODE_FAST_BMAX", "256"))

_applied = False
_orig_forward = None
_slab = {}
_stats = {"fast": 0, "fallback": 0}
_seen = set()


@triton.jit
def _segment_kernel(
    Q, K, V, SEG_ACC, SEG_MAX, SEG_SUM, BT, SEQ_LEN,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_btb, stride_btp,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
    GQA: tl.constexpr, BLOCK_N: tl.constexpr,
    NSEG: tl.constexpr, scale,
):
    pid_g = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, GQA)
    # Segment count is a compile-time constant so the launch grid (and the
    # scratch slab) never depend on host-side kv length: this is what makes
    # the kernel capturable into a CUDA graph.  The per-sequence segment
    # length is derived from seq_lens on the device, like stock's
    # num_par_softmax_segments.
    kvlen = tl.load(SEQ_LEN + pid_s)
    seg_len = (kvlen + NSEG - 1) // NSEG
    start = pid_g * seg_len
    end = tl.minimum(start + seg_len, kvlen)

    m_i = tl.full((GQA,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((GQA,), dtype=tl.float32)
    acc = tl.zeros((GQA, D), dtype=tl.float32)

    qv = tl.load(Q + pid_s * stride_qb + (pid_h * GQA + offs_g)[:, None] * stride_qh
                 + offs_d[None, :] * stride_qd)

    n_full = (end - start) // BLOCK_N * BLOCK_N
    main_end = start + n_full

    for tok0 in range(start, main_end, BLOCK_N):
        offs_t = tok0 + tl.arange(0, BLOCK_N)
        page = tl.load(BT + pid_s * stride_btb + (offs_t // BS) * stride_btp)
        pos = offs_t % BS
        k_ptrs = (K + page[None, :] * stride_kb + pos[None, :] * stride_ks
                  + pid_h * stride_kh + offs_d[:, None] * stride_kd)
        vt_ptrs = (V + page[:, None] * stride_vb + pos[:, None] * stride_vs
                   + pid_h * stride_vh + offs_d[None, :] * stride_vd)
        s = tl.dot(qv, tl.load(k_ptrs)) * scale
        vt = tl.load(vt_ptrs)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
        m_i = m_new

    if main_end < end:
        offs_t = main_end + tl.arange(0, BLOCK_N)
        mask_t = offs_t < end
        page = tl.load(BT + pid_s * stride_btb + (offs_t // BS) * stride_btp,
                       mask=mask_t, other=0)
        pos = offs_t % BS
        k_ptrs = (K + page[None, :] * stride_kb + pos[None, :] * stride_ks
                  + pid_h * stride_kh + offs_d[:, None] * stride_kd)
        vt_ptrs = (V + page[:, None] * stride_vb + pos[:, None] * stride_vs
                   + pid_h * stride_vh + offs_d[None, :] * stride_vd)
        s = tl.dot(qv, tl.load(k_ptrs, mask=mask_t[None, :], other=0.0)) * scale
        s = tl.where(mask_t[None, :], s, float("-inf"))
        vt = tl.load(vt_ptrs, mask=mask_t[:, None], other=0.0)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
        m_i = m_new

    base = (pid_s * HKV + pid_h) * NSEG + pid_g
    tl.store(SEG_MAX + base * GQA + offs_g, m_i)
    tl.store(SEG_SUM + base * GQA + offs_g, l_i)
    tl.store(SEG_ACC + (base * GQA + offs_g)[:, None] * D + offs_d[None, :], acc)


@triton.jit
def _combine_kernel(
    SEG_ACC, SEG_MAX, SEG_SUM, OUT,
    stride_ob, stride_oh, stride_od,
    HQ: tl.constexpr, HKV: tl.constexpr, GQA: tl.constexpr, D: tl.constexpr,
    NSEG: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_g = tl.arange(0, GQA)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, NSEG)

    base = (pid_s * HKV + pid_h) * NSEG
    mx = tl.load(SEG_MAX + (base + offs_n)[None, :] * GQA + offs_g[:, None])
    sm = tl.load(SEG_SUM + (base + offs_n)[None, :] * GQA + offs_g[:, None])
    m = tl.max(mx, 1)
    w = tl.exp2(mx - m[:, None])
    denom = tl.sum(w * sm, 1)
    accs = tl.load(SEG_ACC + ((base + offs_n)[None, :, None] * GQA + offs_g[:, None, None]) * D
                   + offs_d[None, None, :])
    out = tl.sum(accs * w[:, :, None], 1) / denom[:, None]
    tl.store(OUT + pid_s * stride_ob + (pid_h * GQA + offs_g)[:, None] * stride_oh
             + offs_d[None, :] * stride_od, out.to(OUT.dtype.element_ty))


def _make_slab(dev):
    rows = _BMAX * _NUM_KV_HEADS * _NSEG
    return (
        torch.empty((rows, _GQA, _HEAD_SIZE), dtype=torch.float32, device=dev),
        torch.empty((rows, _GQA), dtype=torch.float32, device=dev),
        torch.empty((rows, _GQA), dtype=torch.float32, device=dev),
    )


def _get_slab(dev):
    idx = dev.index
    slab = _slab.get(idx)
    if slab is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        slab = _make_slab(dev)
        _slab[idx] = slab
        logger.info("fl iluvatar: decode-attn scratch slab %.1f MiB",
                    (slab[0].numel() * 4 + slab[1].numel() * 8) / 2**20)
    return slab


def decode_attention(q, k, v, out, block_table, seq_lens, scale, slab,
                     nseg=_NSEG, block_n=_BLOCK_N, num_warps=_NUM_WARPS,
                     num_stages=_NUM_STAGES):
    B, HQ, D = q.shape
    HKV = k.shape[2]
    GQA = HQ // HKV
    BS = k.shape[1]
    seg_acc, seg_max, seg_sum = slab
    _segment_kernel[(nseg, HKV, B)](
        q, k, v, seg_acc, seg_max, seg_sum, block_table, seq_lens,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        block_table.stride(0), block_table.stride(1),
        HQ=HQ, HKV=HKV, D=D, BS=BS, GQA=GQA,
        BLOCK_N=block_n, NSEG=nseg, scale=scale * LOG2E,
        num_warps=num_warps, num_stages=num_stages,
    )
    _combine_kernel[(B, HKV)](
        seg_acc, seg_max, seg_sum, out,
        out.stride(0), out.stride(1), out.stride(2),
        HQ=HQ, HKV=HKV, GQA=GQA, D=D, NSEG=nseg,
        num_warps=4,
    )


def _reject(reason):
    if _DEBUG and reason not in _seen:
        _seen.add(reason)
        logger.info("decode_attn_fast: not eligible (%s)", reason)
    return False


def _eligible(impl, query, kv_cache, md, output_scale):
    if output_scale is not None:
        return _reject("output_scale")
    if impl.alibi_slopes is not None or impl.sinks is not None:
        return _reject("alibi/sinks")
    if getattr(impl, "logits_soft_cap", None):
        return _reject("softcap")
    win = getattr(impl, "sliding_window", None)
    if win is not None and win[0] > -1:
        return _reject("sliding_window")
    if getattr(impl, "kv_cache_dtype", "auto") not in ("auto", "bfloat16"):
        return _reject("kv_cache_dtype")
    mode = getattr(impl, "_kv_quant_mode", None)
    if mode is not None and int(mode) != 0:
        return _reject("kv_quant_mode")
    if query.dtype != torch.bfloat16 or kv_cache.dtype != torch.bfloat16:
        return _reject("dtype")
    if query.shape[1] != _NUM_Q_HEADS or query.shape[2] != _HEAD_SIZE:
        return _reject("q_shape")
    # kv_cache is [num_blocks, 2 (K/V), block_size, num_kv_heads, head_size]
    if (kv_cache.dim() != 5 or kv_cache.shape[1] != 2
            or kv_cache.shape[3] != _NUM_KV_HEADS
            or kv_cache.shape[4] != _HEAD_SIZE):
        return _reject("kv_shape")
    if md.max_query_len != 1 or md.block_table is None:
        return _reject("not_decode")
    if md.causal is not True:
        return _reject("causal")
    if getattr(md, "mm_prefix_range_tensor", None) is not None:
        return _reject("mm_prefix")
    n = md.num_actual_tokens
    if n != md.seq_lens.shape[0] or n > _BMAX:
        return _reject("batch")
    return True


def _fast_forward(impl, query, kv_cache, md, output):
    n = md.num_actual_tokens
    slab = _get_slab(query.device)
    if slab is None:
        return None
    key_cache, value_cache = kv_cache.unbind(1)
    decode_attention(query[:n], key_cache, value_cache, output[:n],
                     md.block_table, md.seq_lens, impl.scale, slab)
    return output


def apply() -> None:
    global _applied, _orig_forward
    if _DISABLED or _applied:
        return
    try:
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
    except Exception as exc:  # pragma: no cover
        logger.warning("decode_attn_fast: cannot patch (%s)", exc)
        return

    _orig_forward = TritonAttentionImpl.forward

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale=None, output_block_scale=None):
        if attn_metadata is not None and _eligible(self, query, kv_cache,
                                                   attn_metadata, output_scale):
            try:
                res = _fast_forward(self, query, kv_cache, attn_metadata, output)
                if res is not None:
                    _stats["fast"] += 1
                    if _DEBUG and _stats["fast"] <= 2:
                        logger.info("decode_attn_fast: engaged n=%d kv=%d",
                                    attn_metadata.num_actual_tokens,
                                    attn_metadata.max_seq_len)
                    return res
            except Exception as exc:
                logger.warning("decode_attn_fast: fallback (%s)", exc)
            _stats["fallback"] += 1
        return _orig_forward(self, layer, query, key, value, kv_cache,
                             attn_metadata, output, output_scale,
                             output_block_scale)

    TritonAttentionImpl.forward = forward
    _applied = True
    # Allocate the scratch slab now: never inside a captured graph.
    try:
        _get_slab(torch.device("cuda", 0))
    except Exception as exc:  # pragma: no cover
        logger.warning("decode_attn_fast: slab alloc failed (%s)", exc)
    logger.info("fl iluvatar: decode-attention fast kernel enabled "
                "(segments=%d block=%d warps=%d stages=%d bmax=%d)",
                _NSEG, _BLOCK_N, _NUM_WARPS, _NUM_STAGES, _BMAX)
