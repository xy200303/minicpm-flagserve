import argparse
import json
import statistics

import torch

import flag_gems


SHAPES = {
    "prefill_qkv": (2048, 2048, 2560),
    "prefill_o": (2048, 2048, 2048),
    "prefill_gateup": (2048, 2048, 12288),
    "prefill_down": (2048, 6144, 2048),
    "decode64_qkv": (64, 2048, 2560),
    "decode64_o": (64, 2048, 2048),
    "decode64_gateup": (64, 2048, 12288),
    "decode64_down": (64, 6144, 2048),
}


def measure(a, b, warmup, reps):
    for _ in range(warmup):
        a @ b
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        a @ b
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--shape", choices=list(SHAPES), default="decode64_gateup")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()
    flag_gems.enable()
    results = []
    for name, (m, k, n) in SHAPES.items():
        if not args.all and name != args.shape:
            continue
        torch.manual_seed(20261028)
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        c = a @ b
        assert torch.isfinite(c).all()
        ms = measure(a, b, 5, args.reps)
        results.append({
            "name": name,
            "shape": [m, k, n],
            "median_ms": ms,
            "tflops": 2 * m * n * k / (ms / 1000) / 1e12,
        })
    text = json.dumps(results, indent=2)
    print(text)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
