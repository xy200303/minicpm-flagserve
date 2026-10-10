#!/usr/bin/env python3
"""Custom split-KV flash-decoding for Iluvatar BI-V150 (paged, GQA).

Stock vLLM unified_attention (3D split-16) gets 6.25ms/layer at
B=64/kv=16384 (0.17 TB/s vs 0.6 TB/s card ceiling).  This kernel:
  * grid (B, HKV, SPLITS); each program online-softmax over its KV segment
  * KV loaded in BLOCK_N=128 token tiles via block-table gather
  * Q group [HQ_PER_KV,128] padded to BLOCK_M=16 for tl.dot
  * fp32 partials per split, tiny reduce kernel merges with LSE rescale
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _dec_partial(
    Q, BT, KC, VC, SEQ_LENS,
    M_OUT, L_OUT, ACC_OUT,
    scale,
    num_seqs,
    seg_len,
    stride_q_b, stride_q_h,
    stride_bt_b,
    stride_k_blk, stride_k_slot, stride_k_h,
    stride_v_blk, stride_v_slot, stride_v_h,
    stride_ml_b, stride_ml_h, stride_ml_s,
    stride_acc_b, stride_acc_h, stride_acc_s,
    H_Q: tl.constexpr,
    HQKV: tl.constexpr,      # query heads per kv head
    BLOCK_M: tl.constexpr,   # padded q rows (>= HQKV)
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    seq_len = tl.load(SEQ_LENS + b)
    seg_start = s * seg_len
    # programs whose segment lies entirely past the sequence still must write
    # neutral partials (-inf/0/0) -- the reduce kernel reads all splits.
    empty = seg_start >= seq_len
    seg_end = tl.minimum(seg_start + seg_len, seq_len)
    if empty:
        seg_end = seg_start  # skip the tile loop entirely

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q_mask = offs_m < HQKV
    q = tl.load(
        Q + b * stride_q_b + (h * HQKV + offs_m)[:, None] * stride_q_h + offs_d[None, :],
        mask=q_mask[:, None], other=0.0,
    )  # bf16, dot on TCU

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    for tile in range(seg_start, seg_end, BLOCK_N):
        offs_n = tile + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seg_end
        pages = tl.load(BT + b * stride_bt_b + offs_n // PAGE, mask=n_mask, other=0)
        slots = offs_n % PAGE
        kaddr = pages * stride_k_blk + slots * stride_k_slot + h * stride_k_h
        kt = tl.load(KC + kaddr[:, None] + offs_d[None, :],
                     mask=n_mask[:, None], other=0.0)  # bf16
        s_ = tl.dot(q, tl.trans(kt), out_dtype=tl.float32) * scale
        s_ = tl.where(n_mask[None, :], s_, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s_, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s_ - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        vaddr = pages * stride_v_blk + slots * stride_v_slot + h * stride_v_h
        vt = tl.load(VC + vaddr[:, None] + offs_d[None, :],
                     mask=n_mask[:, None], other=0.0)  # bf16
        acc += tl.dot(p.to(kt.dtype), vt, out_dtype=tl.float32)
        m_i = m_new

    offs_h = h * HQKV + offs_m  # global q-head index per padded row
    tl.store(M_OUT + b * stride_ml_b + offs_h * stride_ml_h + s * stride_ml_s,
             m_i, mask=q_mask)
    tl.store(L_OUT + b * stride_ml_b + offs_h * stride_ml_h + s * stride_ml_s,
             l_i, mask=q_mask)
    tl.store(ACC_OUT + b * stride_acc_b + offs_h[:, None] * stride_acc_h
             + s * stride_acc_s + offs_d[None, :],
             acc, mask=q_mask[:, None])


@triton.jit
def _dec_partial2(
    Q, BT, KC, VC, SEQ_LENS,
    M_OUT, L_OUT, ACC_OUT,
    scale,
    num_seqs,
    seg_len,
    stride_q_b, stride_q_h,
    stride_bt_b,
    stride_k_blk, stride_k_slot,
    stride_v_blk, stride_v_slot,
    stride_ml_b, stride_ml_h, stride_ml_s,
    stride_acc_b, stride_acc_h, stride_acc_s,
    HQKV: tl.constexpr,      # query heads per kv head
    H_Q: tl.constexpr,       # total q heads (= HKV * HQKV)
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGE: tl.constexpr,
):
    """One program covers ALL kv heads of one (seq, split): per-token cache
    row [HKV, D] is contiguous (stride_k_h == D), so each gather row is
    HKV*D*2 bytes instead of D*2 -- halved row count, better coalescing."""
    b = tl.program_id(0)
    s = tl.program_id(1)

    seq_len = tl.load(SEQ_LENS + b)
    seg_start = s * seg_len
    empty = seg_start >= seq_len
    seg_end = tl.minimum(seg_start + seg_len, seq_len)
    if empty:
        seg_end = seg_start

    offs_m = tl.arange(0, H_Q)
    offs_d = tl.arange(0, D)
    offs_hd = tl.arange(0, 2 * D)  # both kv heads, contiguous per token
    q = tl.load(
        Q + b * stride_q_b + offs_m[:, None] * stride_q_h + offs_d[None, :]
    )  # [H_Q, D] bf16

    m_i = tl.full([H_Q], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([H_Q], dtype=tl.float32)
    acc = tl.zeros([H_Q, D], dtype=tl.float32)

    for tile in range(seg_start, seg_end, BLOCK_N):
        offs_n = tile + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seg_end
        pages = tl.load(BT + b * stride_bt_b + offs_n // PAGE, mask=n_mask, other=0)
        slots = offs_n % PAGE
        kaddr = pages * stride_k_blk + slots * stride_k_slot
        kt2 = tl.load(KC + kaddr[:, None] + offs_hd[None, :],
                      mask=n_mask[:, None], other=0.0)  # [BLOCK_N, 2D]
        k0, k1 = tl.split(tl.trans(tl.reshape(kt2, (BLOCK_N, 2, D)), 0, 2, 1))
        s0 = tl.dot(q, tl.trans(k0), out_dtype=tl.float32)
        s1 = tl.dot(q, tl.trans(k1), out_dtype=tl.float32)
        # row i belongs to kv head i // HQKV
        use0 = (offs_m // HQKV) == 0
        s_ = tl.where(use0[:, None], s0, s1) * scale
        s_ = tl.where(n_mask[None, :], s_, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s_, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s_ - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        vaddr = pages * stride_v_blk + slots * stride_v_slot
        vt2 = tl.load(VC + vaddr[:, None] + offs_hd[None, :],
                      mask=n_mask[:, None], other=0.0)
        v0, v1 = tl.split(tl.trans(tl.reshape(vt2, (BLOCK_N, 2, D)), 0, 2, 1))
        acc += tl.dot(tl.where(use0[:, None], p, 0.0).to(v0.dtype), v0,
                      out_dtype=tl.float32)
        acc += tl.dot(tl.where(use0[:, None], 0.0, p).to(v1.dtype), v1,
                      out_dtype=tl.float32)
        m_i = m_new

    offs_h = offs_m  # global q-head == row here
    tl.store(M_OUT + b * stride_ml_b + offs_h * stride_ml_h + s * stride_ml_s,
             m_i)
    tl.store(L_OUT + b * stride_ml_b + offs_h * stride_ml_h + s * stride_ml_s,
             l_i)
    tl.store(ACC_OUT + b * stride_acc_b + offs_h[:, None] * stride_acc_h
             + s * stride_acc_s + offs_d[None, :],
             acc)


@triton.jit
def _dec_reduce(
    M_IN, L_IN, ACC_IN, OUT,
    stride_ml_b, stride_ml_h, stride_ml_s,
    stride_acc_b, stride_acc_h, stride_acc_s,
    stride_o_b, stride_o_h,
    SPLITS: tl.constexpr,
    D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, SPLITS)
    offs_d = tl.arange(0, D)
    base_ml = b * stride_ml_b + h * stride_ml_h
    m = tl.load(M_IN + base_ml + offs_s * stride_ml_s)
    l = tl.load(L_IN + base_ml + offs_s * stride_ml_s)
    m_star = tl.max(m, axis=0)
    m_star = tl.where(m_star == float("-inf"), 0.0, m_star)
    scale = tl.exp(m - m_star)
    scale = tl.where(m == float("-inf"), 0.0, scale)
    l_star = tl.sum(l * scale, axis=0)
    l_star = tl.where(l_star == 0.0, 1.0, l_star)
    acc = tl.load(ACC_IN + b * stride_acc_b + h * stride_acc_h
                  + offs_s[:, None] * stride_acc_s + offs_d[None, :])
    acc = tl.sum(acc * scale[:, None], axis=0) / l_star
    tl.store(OUT + b * stride_o_b + h * stride_o_h + offs_d,
             acc.to(OUT.dtype.element_ty))


def decode_attention(q, k_cache, v_cache, block_table, seq_lens, scale,
                     max_seq_len, splits=8, block_n=128, num_warps=4,
                     num_stages=2, dual=False):
    """q [B,H,D] bf16; kc/vc [nb,PAGE,HKV,D] bf16; bt [B,max_pages] int32;
    seq_lens [B] int32.  Returns [B,H,D] bf16.

    dual=True: one program covers all HKV heads of a (seq, split) — the
    per-token cache row [HKV,D] is contiguous, so gathers read HKV*D*2
    bytes per token instead of D*2 (needs stride_k_h == D)."""
    B, H, D = q.shape
    HKV = k_cache.shape[2]
    PAGE = k_cache.shape[1]
    HQKV = H // HKV
    BLOCK_M = max(16, triton.next_power_of_2(HQKV))
    seg_len = triton.cdiv(max_seq_len, splits)
    dev = q.device
    acc = torch.empty(B, H, splits, D, device=dev, dtype=torch.float32)
    m_out = torch.empty(B, H, splits, device=dev, dtype=torch.float32)
    l_out = torch.empty(B, H, splits, device=dev, dtype=torch.float32)
    out = torch.empty(B, H, D, device=dev, dtype=torch.bfloat16)

    if dual and HKV == 2 and k_cache.stride(2) == D:
        _dec_partial2[(B, splits)](
            q, block_table, k_cache, v_cache, seq_lens,
            m_out, l_out, acc, scale, B, seg_len,
            q.stride(0), q.stride(1),
            block_table.stride(0),
            k_cache.stride(0), k_cache.stride(1),
            v_cache.stride(0), v_cache.stride(1),
            m_out.stride(0), m_out.stride(1), m_out.stride(2),
            acc.stride(0), acc.stride(1), acc.stride(2),
            HQKV=HQKV, H_Q=H, D=D,
            BLOCK_N=block_n, PAGE=PAGE,
            num_warps=num_warps, num_stages=num_stages,
        )
    else:
        _dec_partial[(B, HKV, splits)](
            q, block_table, k_cache, v_cache, seq_lens,
            m_out, l_out, acc, scale, B, seg_len,
            q.stride(0), q.stride(1),
            block_table.stride(0),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            m_out.stride(0), m_out.stride(1), m_out.stride(2),
            acc.stride(0), acc.stride(1), acc.stride(2),
            H_Q=H, HQKV=HQKV, BLOCK_M=BLOCK_M, D=D,
            BLOCK_N=block_n, PAGE=PAGE,
            num_warps=num_warps, num_stages=num_stages,
        )
    _dec_reduce[(B, H)](
        m_out, l_out, acc, out,
        m_out.stride(0), m_out.stride(1), m_out.stride(2),
        acc.stride(0), acc.stride(1), acc.stride(2),
        out.stride(0), out.stride(1),
        SPLITS=splits, D=D, num_warps=1,
    )
    return out
