// Spike: paged K/V streaming throughput on BI-V150, no torch headers.
#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void kv_stream_kernel(
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const int* __restrict__ BT,
    const int* __restrict__ SEQ,
    float* __restrict__ OUT,
    int HKV, int D, int BS, int NSEG, int bt_stride)
{
    const int g = blockIdx.x;
    const int h = blockIdx.y;
    const int s = blockIdx.z;
    const int kvlen = SEQ[s];
    const int seg = (kvlen + NSEG - 1) / NSEG;
    const int start = g * seg;
    const int end = min(start + seg, kvlen);
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    float acc = 0.f;
    const int vec = 8;          // 8 bf16 == 16 bytes
    const int dvec = D / vec;   // 16 chunks per (token, head) row

    for (int t = start; t < end; ++t) {
        const int page = BT[s * bt_stride + t / BS];
        const int pos = t % BS;
        const int row = ((page * BS + pos) * HKV + h) * D;
        for (int i = tid; i < dvec; i += nthreads) {
            const float4 kk = *reinterpret_cast<const float4*>(K + row + i * vec);
            const float4 vv = *reinterpret_cast<const float4*>(V + row + i * vec);
            const float* kf = reinterpret_cast<const float*>(&kk);
            const float* vf = reinterpret_cast<const float*>(&vv);
            acc += kf[0] + kf[1] + kf[2] + kf[3] + vf[0] + vf[1] + vf[2] + vf[3];
        }
    }
    if (acc == 1234.5678f) OUT[0] = acc;
}

extern "C" void kv_stream(const void* K, const void* V, const int* BT,
                          const int* SEQ, float* OUT, int B, int HKV, int D,
                          int BS, int NSEG, int bt_stride, int threads) {
    dim3 grid(NSEG, HKV, B);
    kv_stream_kernel<<<grid, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(K),
        reinterpret_cast<const __nv_bfloat16*>(V), BT, SEQ, OUT,
        HKV, D, BS, NSEG, bt_stride);
}
