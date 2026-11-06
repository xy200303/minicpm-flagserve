import json
import statistics
import torch
import torch.nn.functional as F
import triton
from flag_gems.runtime.backend._iluvatar.fused.mm_silu_mul import (
    _mm_silu_mul_kernel,
    mm_silu_mul,
)


def event_ms(fn, reps=30):
    for _ in range(5):
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
    return statistics.median(samples), samples


def run_shape(m, bm, bn, bk, warps, stages):
    k, n = 2048, 6144
    torch.manual_seed(20261103 + m)
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * (k ** -0.5)
    b = torch.randn(2 * n, k, dtype=torch.bfloat16, device="cuda").t() * (k ** -0.5)
    reference = F.silu(torch.mm(a, b[:, :n])) * torch.mm(a, b[:, n:])
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)

    def current():
        _mm_silu_mul_kernel[grid](
            a, b, out, m, n, k,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            num_warps=warps, num_stages=stages,
        )

    current()
    torch.cuda.synchronize()
    current_diff = (out.float() - reference.float()).abs().max().item()
    packed = mm_silu_mul(a, b)
    torch.cuda.synchronize()
    packed_diff = (packed.float() - reference.float()).abs().max().item()
    torch.testing.assert_close(packed, reference, rtol=0.05, atol=0.05)
    current_1, current_samples_1 = event_ms(current)
    packed_1, packed_samples_1 = event_ms(lambda: mm_silu_mul(a, b))
    current_2, current_samples_2 = event_ms(current)
    packed_2, packed_samples_2 = event_ms(lambda: mm_silu_mul(a, b))
    current_median = statistics.median([current_1, current_2])
    packed_median = statistics.median([packed_1, packed_2])
    return {
        "M": m,
        "config": [bm, bn, bk, warps, stages],
        "current_diff": current_diff,
        "packed_diff": packed_diff,
        "current_ms": [current_1, current_2],
        "packed_ms": [packed_1, packed_2],
        "current_median_ms": current_median,
        "packed_median_ms": packed_median,
        "speedup": current_median / packed_median,
        "current_samples_ms": current_samples_1 + current_samples_2,
        "packed_samples_ms": packed_samples_1 + packed_samples_2,
    }


def main():
    results = [
        run_shape(64, 64, 128, 32, 8, 2),
        run_shape(64, 64, 128, 32, 16, 2),
        run_shape(2111, 256, 128, 32, 8, 2),
    ]
    text = json.dumps(results, indent=2)
    print(text)
    with open("/workspace/experiments/runs/a60-tile-packed/results-corrected.json", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
