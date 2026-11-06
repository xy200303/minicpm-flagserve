import json
import statistics
import time

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


@triton.jit
def tile_packed_silu_mul_kernel(
    A, GATE, UP, C, M, N, K,
    stride_am, stride_ak, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    rn_local = tl.arange(0, BLOCK_N)
    tile_offsets = rk[:, None] * BLOCK_N + rn_local[None, :]
    a_ptrs = A + ram[:, None] * stride_am + rk[None, :] * stride_ak
    gate_ptrs = GATE + pid_n * BLOCK_K * BLOCK_N + tile_offsets
    up_ptrs = UP + pid_n * BLOCK_K * BLOCK_N + tile_offsets
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        gate = tl.load(gate_ptrs)
        up = tl.load(up_ptrs)
        gate_acc = tl.dot(a, gate, gate_acc, out_dtype=tl.float32)
        up_acc = tl.dot(a, up, up_acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        gate_ptrs += grid_n * BLOCK_K * BLOCK_N
        up_ptrs += grid_n * BLOCK_K * BLOCK_N
    result = gate_acc * tl.sigmoid(gate_acc) * up_acc
    c_ptrs = C + rm[:, None] * stride_cm + rn_local[None, :] * stride_cn
    mask = (rm < M)[:, None] & (rn_local < N)[None, :]
    tl.store(c_ptrs, result.to(C.dtype.element_ty), mask=mask)


def pack_tile(weight, block_k, block_n):
    k, n = weight.shape[1], weight.shape[0]
    return weight.t().contiguous().view(
        k // block_k, block_k, n // block_n, block_n
    ).permute(0, 2, 1, 3).contiguous()


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
    torch.manual_seed(20261106 + m)
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * (k ** -0.5)
    weight = torch.randn(2 * n, k, dtype=torch.bfloat16, device="cuda") * (k ** -0.5)
    b = weight.t()
    reference = F.silu(mm(a, b)[:, :n]) * mm(a, b)[:, n:]
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)

    def current():
        dual_acc_silu_mul_kernel[grid](
            a, b, out, m, n, k,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            num_warps=warps, num_stages=stages,
        )

    start = time.perf_counter()
    gate_packed = pack_tile(weight[:n], bk, bn)
    up_packed = pack_tile(weight[n:], bk, bn)
    torch.cuda.synchronize()
    pack_seconds = time.perf_counter() - start

    def packed():
        tile_packed_silu_mul_kernel[grid](
            a, gate_packed, up_packed, out, m, n, k,
            a.stride(0), a.stride(1), out.stride(0), out.stride(1),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            num_warps=warps, num_stages=stages,
        )

    current()
    torch.cuda.synchronize()
    current_diff = (out.float() - reference.float()).abs().max().item()
    out.zero_()
    packed()
    torch.cuda.synchronize()
    packed_diff = (out.float() - reference.float()).abs().max().item()
    torch.testing.assert_close(out, reference, rtol=0.05, atol=0.05)
    current_1, current_samples_1 = event_ms(current)
    packed_1, packed_samples_1 = event_ms(packed)
    current_2, current_samples_2 = event_ms(current)
    packed_2, packed_samples_2 = event_ms(packed)
    return {
        "M": m,
        "config": [bm, bn, bk, warps, stages],
        "current_diff": current_diff,
        "packed_diff": packed_diff,
        "pack_seconds": pack_seconds,
        "current_ms": [current_1, current_2],
        "packed_ms": [packed_1, packed_2],
        "current_median_ms": statistics.median([current_1, current_2]),
        "packed_median_ms": statistics.median([packed_1, packed_2]),
        "speedup": statistics.median([current_1, current_2]) / statistics.median([packed_1, packed_2]),
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
    with open("/workspace/experiments/runs/a60-tile-packed/results.json", "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
