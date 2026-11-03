import json
import statistics
import torch
from flag_gems.fused import paged_decode_attention
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
import vllm.v1.attention.ops.triton_unified_attention as tua


def measure(fn, reps=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b))
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def main():
    b, hq, hkv, d, bs, kv = 64, 16, 2, 128, 16, 4096
    nb = kv // bs
    pages = b * nb
    torch.manual_seed(20261026)
    q = torch.randn(b, hq, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(pages, bs, hkv, d, device="cuda", dtype=torch.bfloat16) * 0.1
    v = torch.randn(pages, bs, hkv, d, device="cuda", dtype=torch.bfloat16) * 0.1
    bt = torch.randperm(pages, dtype=torch.int32, device="cuda").view(b, nb).contiguous()
    lens = torch.full((b,), kv, dtype=torch.int32, device="cuda")
    cu = torch.arange(b + 1, dtype=torch.int32, device="cuda")
    scale = d ** -0.5
    ref = torch.empty_like(q)
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
    outs = {n: torch.empty_like(q) for n in (4, 8, 16, 32)}
    def fast(n):
        paged_decode_attention(q, k, v, outs[n], bt, lens, scale,
                               num_segments=n, block_n=bs, num_warps=1)
    for n in outs:
        fast(n)
    torch.cuda.synchronize()
    diffs = {}
    for n, out in outs.items():
        diffs[n] = (out.float() - ref.float()).abs().max().item()
        torch.testing.assert_close(out, ref, rtol=0.05, atol=0.05)
    timings = {}
    for n in (32, 16, 8, 4, 4, 8, 16, 32):
        timings.setdefault(n, []).append(measure(lambda n=n: fast(n)))
    result = {"kv": kv, "ragged": False, "max_diff": diffs, "timings": timings}
    text = json.dumps(result, indent=2)
    print(text)
    with open("/workspace/experiments/runs/a53-nseg-small/results.json", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
