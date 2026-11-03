import argparse
import json
import statistics
import time

import torch
from flag_gems.fused import paged_decode_attention
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
import vllm.v1.attention.ops.triton_unified_attention as tua


def measure(fn, warmup=5, reps=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def run_case(kv, ragged):
    b, hq, hkv, d, bs = 64, 16, 2, 128, 16
    nb = kv // bs
    pages = b * nb
    torch.manual_seed(20261026)
    q = torch.randn(b, hq, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(pages, bs, hkv, d, device="cuda", dtype=torch.bfloat16) * 0.1
    v = torch.randn(pages, bs, hkv, d, device="cuda", dtype=torch.bfloat16) * 0.1
    bt = torch.randperm(pages, dtype=torch.int32, device="cuda").view(b, nb).contiguous()
    if ragged:
        lens = torch.tensor([kv - (i % 17) * 179 for i in range(b)], dtype=torch.int32, device="cuda")
    else:
        lens = torch.full((b,), kv, dtype=torch.int32, device="cuda")
    cu = torch.arange(b + 1, dtype=torch.int32, device="cuda")
    scale = d ** -0.5
    ref = torch.empty_like(q)
    out16 = torch.empty_like(q)
    out32 = torch.empty_like(q)
    so = torch.empty((b, hq, 16, d), dtype=torch.float32, device="cuda")
    sm = torch.empty((b, hq, 16), dtype=torch.float32, device="cuda")
    se = torch.empty((b, hq, 16), dtype=torch.float32, device="cuda")

    def stock():
        unified_attention(q=q, k=k, v=v, out=ref, cu_seqlens_q=cu,
            max_seqlen_q=1, seqused_k=lens, max_seqlen_k=kv,
            softmax_scale=scale, causal=True, window_size=(-1, -1),
            block_table=bt, softcap=0.0, q_descale=None,
            k_descale=None, v_descale=None, seq_threshold_3D=64,
            num_par_softmax_segments=16, softmax_segm_output=so,
            softmax_segm_max=sm, softmax_segm_expsum=se)

    def fast16():
        paged_decode_attention(q, k, v, out16, bt, lens, scale,
                               num_segments=16, block_n=bs, num_warps=1)

    def fast32():
        paged_decode_attention(q, k, v, out32, bt, lens, scale,
                               num_segments=32, block_n=bs, num_warps=1)

    kern = tua.kernel_unified_attention

    class Pin:
        def __getitem__(self, grid):
            fn = kern[grid]

            def call(**kw):
                if kw.get("IS_3D"):
                    kw["num_warps"] = 1
                    kw["num_stages"] = 1
                return fn(**kw)

            return call

    tua.kernel_unified_attention = Pin()
    stock()
    fast16()
    fast32()
    torch.cuda.synchronize()
    max_diff16 = (out16.float() - ref.float()).abs().max().item()
    max_diff32 = (out32.float() - ref.float()).abs().max().item()
    torch.testing.assert_close(out16, ref, rtol=0.05, atol=0.05)
    torch.testing.assert_close(out32, ref, rtol=0.05, atol=0.05)
    t32_first = measure(fast32)
    t16 = measure(fast16)
    t32_second = measure(fast32)
    return {
        "kv": kv,
        "ragged": ragged,
        "max_diff16": max_diff16,
        "max_diff32": max_diff32,
        "nseg32_first": t32_first,
        "nseg16": t16,
        "nseg32_second": t32_second,
        "nseg16_over_32_first": t32_first["median_ms"] / t16["median_ms"],
        "nseg16_over_32_second": t32_second["median_ms"] / t16["median_ms"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "torch": torch.__version__,
        "reps": 30,
        "cases": [run_case(4096, False), run_case(4096, True),
                  run_case(16384, False), run_case(16384, True)],
    }
    text = json.dumps(result, indent=2)
    print(text)
    with open(args.output, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
