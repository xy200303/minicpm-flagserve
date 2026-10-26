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
        lens = torch.tensor([kv - (i % 17) * 179 for i in range(b)],
                            dtype=torch.int32, device="cuda")
    else:
        lens = torch.full((b,), kv, dtype=torch.int32, device="cuda")
    cu = torch.arange(b + 1, dtype=torch.int32, device="cuda")
    scale = d ** -0.5
    ref = torch.empty_like(q)
    out = torch.empty_like(q)
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

    def fast():
        paged_decode_attention(q, k, v, out, bt, lens, scale,
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
    fast()
    torch.cuda.synchronize()
    max_diff = (out.float() - ref.float()).abs().max().item()
    stock_t = measure(stock)
    fast_t = measure(fast)
    tokens = int(lens.sum().item())
    kv_gb = tokens * hkv * d * 2 * 2 / 1e9
    return {
        "kv": kv,
        "ragged": ragged,
        "batch": b,
        "max_diff": max_diff,
        "stock": stock_t,
        "fast": fast_t,
        "speedup": stock_t["median_ms"] / fast_t["median_ms"],
        "stock_effective_gb_s": kv_gb / (stock_t["median_ms"] / 1e3),
        "fast_effective_gb_s": kv_gb / (fast_t["median_ms"] / 1e3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "torch": torch.__version__,
        "reps": 30,
        "cases": [run_case(4096, False), run_case(4096, True)],
    }
    text = json.dumps(result, indent=2)
    print(text)
    with open(args.output, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
