import argparse
import json
import statistics
import time

import torch
import flag_gems

from flag_gems.fused.top_p_sample import fused_top_p_sample
from vllm_fl.dispatch.backends.vendor.metax.patches.mcblas_mm import mcblas_mm


def eager_top_p(logits, temperature, top_p):
    probs = torch.softmax(logits.float() / temperature.unsqueeze(1), dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    keep = sorted_probs.cumsum(dim=-1) - sorted_probs <= top_p.unsqueeze(1)
    sorted_probs = sorted_probs * keep
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    token = torch.multinomial(sorted_probs, 1)
    return sorted_indices.gather(1, token).squeeze(1).to(torch.int32)


def measure(fn, warmup, reps):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--reps", type=int, default=30)
    args = parser.parse_args()
    torch.manual_seed(20261025)
    device = "cuda"
    batch, vocab = 64, 130560
    logits = torch.randn(batch, vocab, device=device, dtype=torch.bfloat16)
    temperature = torch.ones(batch, device=device)
    top_p = torch.full((batch,), 0.95, device=device)
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "torch": torch.__version__,
        "shape": [batch, vocab],
        "dtype": str(logits.dtype),
        "top_p": 0.95,
        "temperature": 1.0,
        "reps": args.reps,
    }
    fused = lambda: fused_top_p_sample(logits, temperature, top_p, zoom_rounds=1)
    eager = lambda: eager_top_p(logits, temperature, top_p)
    out = fused()
    assert out.shape == (batch,) and out.dtype == torch.int32
    result["fused_top_p"] = measure(fused, 10, args.reps)
    result["eager_top_p"] = measure(eager, 5, args.reps)
    result["top_p_speedup"] = (
        result["eager_top_p"]["median_ms"] / result["fused_top_p"]["median_ms"]
    )

    m, k, n = 64, 2048, 130560
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b = torch.randn(k, n, device=device, dtype=torch.bfloat16)
    result["lm_head_shape"] = [m, k, n]
    result["native_lm_head"] = measure(lambda: torch.mm(a, b), 10, args.reps)
    flag_gems.enable()
    result["triton_lm_head"] = measure(lambda: torch.mm(a, b), 10, args.reps)
    result["mcblas_lm_head"] = measure(lambda: mcblas_mm(a, b), 10, args.reps)
    result["lm_head_speedup"] = (
        result["triton_lm_head"]["median_ms"] / result["mcblas_lm_head"]["median_ms"]
    )
    text = json.dumps(result, indent=2)
    print(text)
    with open(args.output, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
