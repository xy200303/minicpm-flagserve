#!/usr/bin/env python3
"""Full mm sweep for MiniCPM decode shapes: dense nt configs (incl. cpasync) + forced split-K."""
import sys
import torch
import triton

sys.path.insert(0, "/workspace/FlagGems/src")
import flag_gems
from flag_gems import runtime

dev = "cuda"
M = 64
SHAPES = [("qkv", 2048, 2560), ("o_proj", 2048, 2048), ("gate_up", 2048, 12288), ("down", 6144, 2048)]

def timeit(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3

# candidate dense-nt configs
cands = []
for BM in (32, 64):
    for BN in (64, 128, 256):
        for BK in (64, 128, 256):
            for ns in (2, 3, 4):
                for nw in (4, 8):
                    for pl in ("basic", "cpasync"):
                        cands.append(triton.Config(
                            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK,
                             "GROUP_M": 8, "pipeline": pl, "scenario": ""},
                            num_stages=ns, num_warps=nw))
print(f"dense nt candidates: {len(cands)}", flush=True)

import importlib
mxmm = importlib.import_module("flag_gems.runtime.backend._metax.ops.mm")

flag_gems.enable(record=False)

for name, K, N in SHAPES:
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)

    # 1) expanded dense nt space (fresh tuner per shape by reloading module state)
    orig_cfg = runtime.get_tuned_config
    runtime.get_tuned_config = lambda op, **kw: cands if op == "mm_nt" else orig_cfg(op, **kw)
    # bust libtuner cache by re-importing? use a fresh process per shape instead — here we
    # just let the tuner autotune on first mm call for this shape key.
    try:
        t_dense = timeit(lambda: torch.mm(a, w.t()), 100)
    except Exception as e:
        t_dense = float("nan"); print(f"{name} dense failed: {e}")
    runtime.get_tuned_config = orig_cfg
    bw = lambda t: K * N * 2 / (t * 1e-6) / 1e9
    print(f"{name:<9} dense_tuned={t_dense:7.1f} us ({bw(t_dense):6.0f} GB/s)", flush=True)

    # 2) forced splitk
    orig_scen = mxmm.splitk_mm_scenario
    mxmm.splitk_mm_scenario = lambda M_, N_, K_: True
    try:
        torch.mm(a, w.t())  # trigger tuner for splitk path
        t_split = timeit(lambda: torch.mm(a, w.t()), 100)
    except Exception as e:
        t_split = float("nan"); print(f"{name} splitk failed: {e}")
    mxmm.splitk_mm_scenario = orig_scen

    print(f"{name:<9} splitk={t_split:7.1f} us ({bw(t_split):6.0f} GB/s)", flush=True)
