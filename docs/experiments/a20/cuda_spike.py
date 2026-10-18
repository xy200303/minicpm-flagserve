"""Spike: can a hand-written CUDA kernel stream paged K/V faster than Triton?

Only measures the memory path (paged gather + accumulate), same shape as the
A20 kernel: B=64, kv=4096, HKV=2, D=128, block=16, NSEG=32.
"""
import time

import torch
from torch.utils.cpp_extension import load_inline

src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void kv_stream_kernel(
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const int* __restrict__ BT,
    const int* __restrict__ SEQ,
    float* __restrict__ OUT,
    int B, int HKV, int D, int BS, int NSEG, int bt_stride)
{
    const int g = blockIdx.x;
    const int h = blockIdx.y;
    const int s = blockIdx.z;
    const int kvlen = SEQ[s];
    int seg = (kvlen + NSEG - 1) / NSEG;
    int start = g * seg;
    int end = min(start + seg, kvlen);
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // accumulate to prevent DCE
    float acc = 0.f;
    const int vec = 8;                    // 8 bf16 = 16 bytes
    const int dvec = D / vec;             // 16 vecs per row

    for (int t = start; t < end; ++t) {
        const int page = BT[s * bt_stride + t / BS];
        const int pos = t % BS;
        const int row = ((page * BS + pos) * HKV + h) * D;
        // each thread handles a strided set of 16B chunks
        for (int i = tid; i < dvec; i += nthreads) {
            const float4 kk = *reinterpret_cast<const float4*>(K + row + i * vec);
            const float4 vv = *reinterpret_cast<const float4*>(V + row + i * vec);
            const float* kf = reinterpret_cast<const float*>(&kk);
            const float* vf = reinterpret_cast<const float*>(&vv);
            acc += kf[0] + kf[1] + kf[2] + kf[3] + vf[0] + vf[1] + vf[2] + vf[3];
        }
    }
    if (acc == 1234.5678f) OUT[0] = acc;   // never true
}

void kv_stream(torch::Tensor K, torch::Tensor V, torch::Tensor BT,
               torch::Tensor SEQ, torch::Tensor OUT, int64_t nseg, int64_t bs,
               int64_t hkv, int64_t d, int64_t threads) {
    const int B = SEQ.size(0);
    dim3 grid(nseg, hkv, B);
    kv_stream_kernel<<<grid, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
        BT.data_ptr<int>(), SEQ.data_ptr<int>(), OUT.data_ptr<float>(),
        B, (int)hkv, (int)d, (int)bs, (int)nseg, (int)BT.size(1));
}
"""

mod = load_inline(
    name="kv_stream_spike",
    cpp_sources="void kv_stream(torch::Tensor, torch::Tensor, torch::Tensor,"
                " torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t,"
                " int64_t, int64_t);",
    cuda_sources=src,
    functions=["kv_stream"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)

B, KV, HQ, HKV, D, BS, NSEG = 64, 4096, 16, 2, 128, 16, 32
NB = KV // BS
nb = B * NB
k = torch.randn(nb, BS, HKV, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(nb, BS, HKV, D, dtype=torch.bfloat16, device="cuda")
bt = torch.randperm(nb, dtype=torch.int32, device="cuda").view(B, NB).contiguous()
seq = torch.full((B,), KV, dtype=torch.int32, device="cuda")
out = torch.zeros(1, dtype=torch.float32, device="cuda")
kv_gb = B * KV * HKV * D * 2 * 2 / 1e9


def bench(fn, n=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


for threads in (128, 256):
    t = bench(lambda: mod.kv_stream(k, v, bt, seq, out, NSEG, BS, HKV, D, threads))
    print(f"cuda loads threads={threads}: {t:.3f} ms/layer  {kv_gb/(t/1e3):4.0f} GB/s "
          f"(triton load-only reference: 0.53 ms / 505 GB/s)", flush=True)
