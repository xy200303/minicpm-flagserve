"""What does Triton actually emit for the PV dot on BI-V150 (sm_71)?

Runs the FlagGems op once, then inspects the compiled kernel's TTGIR/LLIR:
dot ops, layout conversions and shared-memory round trips around the loop.
"""
import re

import torch

import sys
from flag_gems.fused import paged_decode_attention as op
m = sys.modules["flag_gems.fused.paged_decode_attention"]

HQ, HKV, D, BS, KV, B, NSEG = 16, 2, 128, 16, 4096, 64, 32
NB = KV // BS
nb = B * NB
torch.manual_seed(0)
q = torch.randn(B, HQ, D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(nb, BS, HKV, D, dtype=torch.bfloat16, device="cuda")
v = torch.randn(nb, BS, HKV, D, dtype=torch.bfloat16, device="cuda")
out = torch.empty_like(q)
bt = torch.randperm(nb, dtype=torch.int32, device="cuda").view(B, NB).contiguous()
seq = torch.full((B,), KV, dtype=torch.int32, device="cuda")
op(q, k, v, out, bt, seq, D ** -0.5, num_segments=NSEG, block_n=BS, num_warps=1)
torch.cuda.synchronize()

for name, fn in (("segment_paged", m._segment_kernel_paged),
                 ("segment_generic", m._segment_kernel),
                 ("combine", m._combine_kernel)):
    cache = getattr(fn, "cache", None)
    if not cache:
        print(f"== {name}: no cache"); continue
    for dev, entry in cache.items():
        for key, ck in entry.items():
            asm = getattr(ck, "asm", {}) or {}
            print(f"== {name} (device {dev}) asm keys: {sorted(asm.keys())}")
            ttgir = asm.get("ttgir", "")
            if ttgir:
                dots = re.findall(r"tt\.dot[^\n]*", ttgir)
                print(f"   dot ops: {len(dots)}")
                for d in dots[:4]:
                    print("     ", d[:150])
                for pat in ("ttg.convert_layout", "ttg.local_alloc",
                            "ttg.local_load", "ttg.local_store",
                            "ttg.async_copy_global_to_local",
                            "ttng.async_copy", "ttg.warp_specialize"):
                    n = ttgir.count(pat)
                    if n:
                        print(f"   {pat}: {n}")
            llir = asm.get("llir", "")
            if llir:
                mmas = re.findall(r"(?:mma|wmma|hmma|wgmma)[.\w]*", llir)
                uniq = sorted(set(mmas))
                print(f"   mma-ish in llir: {uniq[:8]}")
            ptx = asm.get("ptx", "")
            if ptx:
                for pat in ("mma.sync", "wmma", "ldmatrix", "cvt.rn.f16.bf16",
                            "bar.sync", "cp.async"):
                    n = ptx.count(pat)
                    if n:
                        print(f"   ptx {pat}: {n}")
            break
        break
