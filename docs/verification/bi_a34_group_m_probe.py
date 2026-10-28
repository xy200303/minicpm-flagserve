import argparse
import json
import statistics

import torch
import triton

from flag_gems.runtime.backend._iluvatar.ops.mm import mm_kernel


SHAPES = {
    "prefill_qkv": ((2048, 2048, 2560), (128, 128, 32, 1, 8, 2)),
    "prefill_o": ((2048, 2048, 2048), (256, 256, 32, 1, 16, 2)),
    "prefill_gateup": ((2048, 2048, 12288), (256, 256, 32, 1, 8, 2)),
    "prefill_down": ((2048, 6144, 2048), (256, 256, 32, 1, 16, 2)),
    "decode64_qkv": ((64, 2048, 2560), (64, 128, 64, 4, 8, 1)),
    "decode64_o": ((64, 2048, 2048), (64, 128, 64, 2, 8, 2)),
    "decode64_gateup": ((64, 2048, 12288), (64, 256, 64, 1, 8, 1)),
    "decode64_down": ((64, 6144, 2048), (64, 64, 32, 4, 4, 2)),
}
RAW_MM = mm_kernel.fn.fn


def run_cfg(a, b, c, m, n, k, cfg, group_m):
    bm, bn, bk, sk, warps, stages = cfg
    if sk > 1:
        c.zero_()
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), sk)
    RAW_MM[grid](
        a, b, c, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        acc_dtype=triton.language.float32,
        input_precision=None,
        fp8_fast_accum=True,
        GROUP_M=group_m,
        AB_DTYPE=triton.language.bfloat16,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        SPLIT_K=sk,
        num_warps=warps,
        num_stages=stages,
    )


def measure(a, b, c, m, n, k, cfg, group_m, reps):
    for _ in range(5):
        run_cfg(a, b, c, m, n, k, cfg, group_m)
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_cfg(a, b, c, m, n, k, cfg, group_m)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--groups", type=int, nargs="+", default=[2, 4, 8, 16])
    args = parser.parse_args()
    results = []
    for name, ((m, k, n), cfg) in SHAPES.items():
        torch.manual_seed(20261028)
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        c = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        timings = {}
        for group_m in args.groups:
            timings[str(group_m)] = measure(a, b, c, m, n, k, cfg, group_m, args.reps)
        results.append({
            "name": name,
            "shape": [m, k, n],
            "cfg": cfg,
            "median_ms_by_group_m": timings,
            "best_group_m": min(timings, key=timings.get),
            "best_ms": min(timings.values()),
            "best_vs_group8": timings["8"] / min(timings.values()),
        })
    text = json.dumps(results, indent=2)
    print(text)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
