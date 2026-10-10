#!/usr/bin/env python3
"""Test custom flash-decoding kernel vs stock unified_attention (BI-V150)."""
import sys
import torch

sys.path.insert(0, "/workspace")
from flash_dec_iluvatar import decode_attention
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

dev = "cuda"
torch.manual_seed(0)
B, H, HKV, D, BLK = 64, 16, 2, 128, 16

def timeit(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3

for KV in (4096, 16384):
    nblocks = B * (KV // BLK) + 8
    kc = torch.randn(nblocks, BLK, HKV, D, device=dev, dtype=torch.bfloat16)
    vc = torch.randn(nblocks, BLK, HKV, D, device=dev, dtype=torch.bfloat16)
    q = torch.randn(B, H, D, device=dev, dtype=torch.bfloat16)
    out = torch.empty(B, H, D, device=dev, dtype=torch.bfloat16)
    cuq = torch.arange(0, B + 1, device=dev, dtype=torch.int32)
    # mixed lengths: half at KV, half at KV//2 (realistic spread)
    seqlens = torch.where(
        torch.arange(B, device=dev) % 2 == 0, KV, KV // 2
    ).to(torch.int32)
    bt = torch.arange(0, B * (KV // BLK), device=dev, dtype=torch.int32).view(B, -1)

    # fp32 reference (per row): softmax(q @ K^T * scale) @ V, GQA expand
    kk = kc[bt].view(B, KV, HKV, D)
    vv = vc[bt].view(B, KV, HKV, D)
    ref = torch.empty(B, H, D, device=dev, dtype=torch.float32)
    for b in range(B):
        L = seqlens[b].item()
        kh = kk[b, :L].repeat_interleave(H // HKV, dim=1).float()
        vh = vv[b, :L].repeat_interleave(H // HKV, dim=1).float()
        s = torch.einsum("hd,lhd->hl", q[b].float(), kh) * (D ** -0.5)
        ref[b] = torch.einsum("hl,lhd->hd", s.softmax(-1), vh)

    # stock 3D
    SEG = 16
    so = torch.empty(B, H, SEG, D, device=dev, dtype=torch.float32)
    sm = torch.empty(B, H, SEG, device=dev, dtype=torch.float32)
    se = torch.empty(B, H, SEG, device=dev, dtype=torch.float32)
    unified_attention(
        q=q, k=kc, v=vc, out=out, cu_seqlens_q=cuq, max_seqlen_q=1,
        seqused_k=seqlens, max_seqlen_k=KV, softmax_scale=D ** -0.5,
        causal=True, window_size=(-1, -1), block_table=bt, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
        seq_threshold_3D=128, num_par_softmax_segments=SEG,
        softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se)
    torch.cuda.synchronize()
    err_stock = (out.float() - ref).abs().max().item()

    gb = int(seqlens.sum()) * HKV * D * 2 * 2 / 1e9
    t_stock = timeit(lambda: unified_attention(
        q=q, k=kc, v=vc, out=out, cu_seqlens_q=cuq, max_seqlen_q=1,
        seqused_k=seqlens, max_seqlen_k=KV, softmax_scale=D ** -0.5,
        causal=True, window_size=(-1, -1), block_table=bt, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
        seq_threshold_3D=128, num_par_softmax_segments=SEG,
        softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se))

    mine = decode_attention(q, kc, vc, bt, seqlens, D ** -0.5, KV)
    torch.cuda.synchronize()
    err_mine = (mine.float() - ref).abs().max().item()
    mine2 = decode_attention(q, kc, vc, bt, seqlens, D ** -0.5, KV, dual=True)
    torch.cuda.synchronize()
    err_dual = (mine2.float() - ref).abs().max().item()
    print(f"  [correctness] single={err_mine:.4f} dual={err_dual:.4f}", flush=True)

    best = (1e9, None)
    for splits, bn, wp, st, dual in [
            (8, 64, 4, 1, False), (8, 64, 4, 1, True), (8, 64, 2, 1, True),
            (16, 64, 4, 1, True), (4, 64, 4, 1, True), (8, 128, 4, 1, True),
            (8, 64, 4, 2, True), (16, 128, 4, 1, True), (8, 64, 8, 1, True),
            (32, 64, 4, 1, True), (8, 32, 4, 1, True)]:
        try:
            t = timeit(lambda: decode_attention(q, kc, vc, bt, seqlens,
                                                D ** -0.5, KV, splits=splits,
                                                block_n=bn, num_warps=wp,
                                                num_stages=st, dual=dual))
        except Exception:
            continue
        print(f"    cfg sp={splits} bn={bn} w={wp} st={st} dual={dual}: {t:7.1f}us", flush=True)
        if t < best[0]:
            best = (t, (splits, bn, wp, st, dual))
    print(f"kv~{KV}: stock {t_stock:7.1f}us err={err_stock:.4f} | "
          f"custom best {best[0]:7.1f}us cfg={best[1]} err={err_mine:.4f} | "
          f"custom {gb/best[0]*1e3:.2f} TB/s, speedup {t_stock/best[0]:.2f}x",
          flush=True)
