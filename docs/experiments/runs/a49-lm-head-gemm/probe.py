import argparse
import json
import statistics

import torch
import triton
import triton.language as tl

from flag_gems.runtime.backend._iluvatar.ops.mm import mm, _raw_mm_kernel


def event_ms(fn, reps=20):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    M, K, N = 64, 2048, 130560
    torch.manual_seed(20261104)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * (K ** -0.5)
    b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda") * (K ** -0.5)
    reference = mm(a, b)
    baseline_ms, baseline_samples = event_ms(lambda: mm(a, b))
    result = {
        "shape": [M, K, N],
        "baseline_ms": baseline_ms,
        "baseline_samples_ms": baseline_samples,
        "configs": [],
    }
    out = torch.empty_like(reference)
    configs = [
        (64, 128, 32, 1, 8, 1),
        (64, 256, 32, 1, 8, 1),
        (64, 256, 64, 1, 8, 1),
        (64, 128, 64, 1, 8, 2),
        (64, 256, 32, 1, 16, 2),
        (64, 256, 64, 2, 8, 1),
        (64, 128, 64, 4, 4, 2),
        (128, 256, 32, 1, 8, 2),
    ]
    for BM, BN, BK, SK, warps, stages in configs:
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), SK)
        def run():
            if SK > 1:
                out.zero_()
            _raw_mm_kernel[grid](
                a, b, out, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                out.stride(0), out.stride(1),
                acc_dtype=tl.float32, input_precision=None, fp8_fast_accum=True,
                GROUP_M=8, AB_DTYPE=tl.bfloat16,
                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, SPLIT_K=SK,
                num_warps=warps, num_stages=stages,
            )
        item = {"config": [BM, BN, BK, SK, warps, stages]}
        try:
            out.zero_()
            run()
            torch.cuda.synchronize()
            diff = (out.float() - reference.float()).abs()
            item.update({
                "max_diff": diff.max().item(),
                "mean_abs_diff": diff.mean().item(),
                "finite": bool(torch.isfinite(out).all().item()),
            })
            torch.testing.assert_close(out, reference, rtol=0.05, atol=0.05)
            median, samples = event_ms(run)
            item.update({"median_ms": median, "samples_ms": samples})
        except Exception as exc:
            item["error"] = str(exc)
        result["configs"].append(item)
        print(item, flush=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
