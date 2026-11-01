import argparse
import ctypes
import json
import statistics

import torch

from vllm_fl.dispatch.backends.vendor.metax.patches import mcblas_mm as binding


SHAPES = {
    "decode64_qkv": (64, 2048, 2560),
    "decode64_o": (64, 2048, 2048),
    "decode64_gateup": (64, 2048, 12288),
    "decode64_down": (64, 6144, 2048),
    "prefill_qkv": (2048, 2048, 2560),
    "prefill_o": (2048, 2048, 2048),
    "prefill_gateup": (2048, 2048, 12288),
    "prefill_down": (2048, 6144, 2048),
}


def call(a, b, out, algo):
    m, k = a.shape
    n = b.shape[1]
    status = binding._lib.mcblasSetStream(
        binding._handle, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    )
    if status != 0:
        raise RuntimeError(f"mcblasSetStream: {status}")
    status = binding._lib.mcblasGemmEx(
        binding._handle, binding._MCBLAS_OP_N, binding._MCBLAS_OP_N,
        n, m, k,
        ctypes.byref(binding._alpha),
        ctypes.c_void_p(b.data_ptr()), binding._MACA_R_16BF, n,
        ctypes.c_void_p(a.data_ptr()), binding._MACA_R_16BF, k,
        ctypes.byref(binding._beta),
        ctypes.c_void_p(out.data_ptr()), binding._MACA_R_16BF, n,
        binding._MCBLAS_COMPUTE_32F, algo,
    )
    if status != 0:
        raise RuntimeError(f"mcblasGemmEx: {status}")


def measure(a, b, out, algo, reps):
    for _ in range(3):
        call(a, b, out, algo)
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call(a, b, out, algo)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=list(SHAPES), required=True)
    parser.add_argument("--algos", type=int, nargs="+",
                        default=[-1, 99] + list(range(100, 116)))
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    m, k, n = SHAPES[args.shape]
    torch.manual_seed(20261101)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    reference = None
    results = []
    for algo in args.algos:
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        try:
            ms = measure(a, b, out, algo, args.reps)
            torch.cuda.synchronize()
            if reference is None:
                reference = out.clone()
            max_diff = (out - reference).abs().max().item()
            results.append({"algo": algo, "median_ms": ms, "max_diff": max_diff,
                            "tflops": 2 * m * n * k / (ms / 1000) / 1e12})
        except Exception as exc:
            results.append({"algo": algo, "error": str(exc)})
    valid = [row for row in results if "median_ms" in row]
    baseline = next(row for row in valid if row["algo"] == -1)
    best = min(valid, key=lambda row: row["median_ms"])
    payload = {"shape": args.shape, "mkn": [m, k, n], "reps": args.reps,
               "baseline_ms": baseline["median_ms"], "best": best,
               "best_speedup": baseline["median_ms"] / best["median_ms"],
               "results": results}
    text = json.dumps(payload, indent=2)
    print(text)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
