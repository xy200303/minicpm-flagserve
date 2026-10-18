"""ctypes driver for the CUDA spike (kv_stream.cu)."""
import ctypes
import os
import subprocess
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "kv_stream.cu")
SO = os.path.join(HERE, "kv_stream.so")

if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(SRC):
    cmd = [
        "/usr/local/corex/bin/nvcc", "-O3", "-shared", "-Xcompiler", "-fPIC",
        "-I/usr/local/corex/include", "-L/usr/local/corex/lib64", "-lcudart",
        "-o", SO, SRC,
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

lib = ctypes.CDLL(SO)
lib.kv_stream.restype = None
lib.kv_stream.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]

B, KV, HQ, HKV, D, BS, NSEG = 64, 4096, 16, 2, 128, 16, 32
NB = KV // BS
nb = B * NB
torch.manual_seed(0)
k = torch.randn(nb, BS, HKV, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(nb, BS, HKV, D, dtype=torch.bfloat16, device="cuda")
bt = torch.randperm(nb, dtype=torch.int32, device="cuda").view(B, NB).contiguous()
seq = torch.full((B,), KV, dtype=torch.int32, device="cuda")
out = torch.zeros(1, dtype=torch.float32, device="cuda")
kv_gb = B * KV * HKV * D * 2 * 2 / 1e9


def run(threads):
    lib.kv_stream(k.data_ptr(), v.data_ptr(), bt.data_ptr(), seq.data_ptr(),
                  out.data_ptr(), B, HKV, D, BS, NSEG, bt.size(1), threads)


def bench(fn, n=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


for threads in (64, 128, 256):
    t = bench(lambda: run(threads))
    print(f"cuda loads threads={threads:3d}: {t:.3f} ms/layer  "
          f"{kv_gb/(t/1e3):4.0f} GB/s   (Triton load-only: 0.53 ms / 505 GB/s)",
          flush=True)
