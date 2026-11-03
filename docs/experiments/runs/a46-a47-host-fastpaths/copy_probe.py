import argparse
import json
import statistics
import time

import torch

import flag_gems
from flag_gems.ops.copy import copy_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    src = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty_like(src)
    copy_(dst, src)
    torch.cuda.synchronize()
    assert torch.equal(dst, src)
    samples = []
    inner = 2000
    for _ in range(9):
        start = time.perf_counter()
        for _ in range(inner):
            copy_(dst, src)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / inner)
    result = {
        "module": flag_gems.__file__,
        "shape": [64, 256],
        "dtype": "bfloat16",
        "inner": inner,
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
        "correct": True,
    }
    print(json.dumps(result, indent=2), flush=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
