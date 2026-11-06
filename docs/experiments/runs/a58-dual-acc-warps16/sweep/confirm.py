import argparse
import json
import statistics

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flag_gems.runtime.backend._iluvatar.ops.mm import mm


@triton.jit
def dual_acc_silu_mul_kernel(
    A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = A + ram[:, None] * stride_am + rk[None, :] * stride_ak
    gate_ptrs = B + rk[:, None] * stride_bk + rbn[None, :] * stride_bn
    up_ptrs = gate_ptrs + N * stride_bn
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        gate = tl.load(gate_ptrs)
        up = tl.load(up_ptrs)
        gate_acc = tl.dot(a, gate, gate_acc, out_dtype=tl.float32)
        up_acc = tl.dot(a, up, up_acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        gate_ptrs += BLOCK_K * stride_bk
        up_ptrs += BLOCK_K * stride_bk
    result = gate_acc * tl.sigmoid(gate_acc) * up_acc
    c_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(c_ptrs, result.to(C.dtype.element_ty), mask=mask)


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


def run_shape(M, configs):
    K, N = 2048, 6144
    torch.manual_seed(20261103 + M)
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * (K ** -0.5)
    b = torch.randn(2 * N, K, dtype=torch.bfloat16, device="cuda").t() * (K ** -0.5)
    gate_up = mm(a, b)
    reference = F.silu(gate_up[:, :N]) * gate_up[:, N:]
    baseline_ms, baseline_samples = event_ms(lambda: F.silu(gate_up[:, :N]) * gate_up[:, N:])

    def full_baseline():
        current = mm(a, b)
        return F.silu(current[:, :N]) * current[:, N:]

    full_baseline_ms, full_baseline_samples = event_ms(full_baseline)
    result = {
        "M": M,
        "baseline_total_ms": full_baseline_ms,
        "baseline_total_samples_ms": full_baseline_samples,
        "activation_only_ms": baseline_ms,
        "configs": [],
    }
    out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    for BM, BN, BK, warps, stages in configs:
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        def run():
            dual_acc_silu_mul_kernel[grid](
                a, b, out, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                out.stride(0), out.stride(1),
                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                num_warps=warps, num_stages=stages,
            )
        item = {"config": [BM, BN, BK, warps, stages]}
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
        print(M, item, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    decode = [
        (64, 128, 32, 8, 2),
        (64, 128, 32, 16, 2),
    ]
    results = [run_shape(64, decode) for _ in range(5)]
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
