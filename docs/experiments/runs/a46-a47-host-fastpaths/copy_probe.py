import argparse
import json
import statistics
import time

import torch
import triton

import flag_gems
from flag_gems.ops.copy import copy_

try:
    from flag_gems.ops.copy import _copy_contiguous_kernel
except ImportError:
    _copy_contiguous_kernel = None


def measure(fn, inner):
    samples = []
    for _ in range(9):
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / inner)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    src = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty_like(src)
    inner = 2000
    copy_(dst, src)
    torch.cuda.synchronize()
    assert torch.equal(dst, src)
    samples = measure(lambda: copy_(dst, src), inner)
    result = {
        "module": flag_gems.__file__,
        "shape": [64, 256],
        "dtype": "bfloat16",
        "inner": inner,
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
        "correct": True,
    }
    if args.sweep:
        if _copy_contiguous_kernel is None:
            raise RuntimeError("sweep requires the contiguous copy candidate")
        result["sweep"] = []
        n = dst.numel()
        for block in (256, 512, 1024, 2048):
            for warps in (1, 2, 4):
                def run(block=block, warps=warps):
                    _copy_contiguous_kernel[(triton.cdiv(n, block),)](
                        dst, src, n, BLOCK=block, num_warps=warps
                    )

                run()
                torch.cuda.synchronize()
                assert torch.equal(dst, src)
                config_samples = measure(run, inner)
                result["sweep"].append({
                    "block": block,
                    "num_warps": warps,
                    "median_ms": statistics.median(config_samples),
                    "samples_ms": config_samples,
                    "correct": True,
                })
    print(json.dumps(result, indent=2), flush=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
