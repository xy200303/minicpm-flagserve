#!/usr/bin/env python3
"""ctypes -> mcblasGemmEx prototype: reach vendor GEMM regardless of dispatcher.

Row-major C[M,N] = A[M,K] @ B[K,N] is computed as the column-major identity
C^T = B^T A^T: gemmEx(OP_N, OP_N, m'=N, n'=M, k'=K, A=B_ptr lda=N,
B=A_ptr ldb=K, C=C_ptr ldc=N).
"""
import ctypes
import torch

dev = "cuda"
MCBLAS_OP_N = 0
MACA_R_16BF = 14
MCBLAS_COMPUTE_32F = 68
MCBLAS_GEMM_DEFAULT = -1

_lib = ctypes.CDLL("/opt/maca/lib/libmcblas.so")
_lib.mcblasCreate.restype = ctypes.c_int
_lib.mcblasCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_lib.mcblasSetStream.restype = ctypes.c_int
_lib.mcblasSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_lib.mcblasGemmEx.restype = ctypes.c_int
_lib.mcblasGemmEx.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
]

_handle = ctypes.c_void_p()
st = _lib.mcblasCreate(ctypes.byref(_handle))
assert st == 0, f"mcblasCreate failed: {st}"
_alpha = ctypes.c_float(1.0)
_beta = ctypes.c_float(0.0)


def vendor_mm(a, b, out=None):
    """a: [M,K] bf16 row-major, b: [K,N] bf16 row-major -> [M,N] bf16."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    assert a.is_contiguous() and b.is_contiguous()
    if out is None:
        out = torch.empty(M, N, device=a.device, dtype=torch.bfloat16)
    st = _lib.mcblasSetStream(_handle, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    assert st == 0
    st = _lib.mcblasGemmEx(
        _handle, MCBLAS_OP_N, MCBLAS_OP_N,
        N, M, K,
        ctypes.byref(_alpha),
        ctypes.c_void_p(b.data_ptr()), MACA_R_16BF, N,
        ctypes.c_void_p(a.data_ptr()), MACA_R_16BF, K,
        ctypes.byref(_beta),
        ctypes.c_void_p(out.data_ptr()), MACA_R_16BF, N,
        MCBLAS_COMPUTE_32F, MCBLAS_GEMM_DEFAULT,
    )
    assert st == 0, f"mcblasGemmEx failed: {st}"
    return out


def timeit(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3


for (K, N) in [(2048, 2560), (2048, 2048), (2048, 12288), (6144, 2048)]:
    M = 2048
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
    ref = torch.mm(a, b)  # gems NOT enabled here: this is vendor aten
    mine = vendor_mm(a, b)
    err = (mine.float() - ref.float()).abs().max().item()
    rel = err / ref.float().abs().max().item()
    t_ref = timeit(lambda: torch.mm(a, b))
    t_me = timeit(lambda: vendor_mm(a, b))
    print(f"K={K} N={N}: rel_err={rel:.2e}  aten {t_ref:7.1f}us  ctypes {t_me:7.1f}us", flush=True)
