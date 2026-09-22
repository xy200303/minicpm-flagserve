#!/usr/bin/env python3
"""Analyze a torch-profiler Chrome trace: GPU busy vs idle, top kernels, top CPU ops."""
import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
t0_range = float(sys.argv[2]) if len(sys.argv) > 2 else None  # optional: skip first N seconds

with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
    data = json.load(f)

events = data["traceEvents"]
print(f"total events: {len(events)}")

kernels = []   # (ts, dur, name, stream)
cpu_ops = []
extern = []    # cuda_runtime / driver calls (launch overhead)
for e in events:
    if e.get("ph") != "X":
        continue
    cat = e.get("cat", "")
    ts, dur = e.get("ts", 0), e.get("dur", 0)
    if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
        kernels.append((ts, dur, e["name"], e.get("args", {}).get("stream", e.get("tid"))))
    elif cat == "cpu_op":
        cpu_ops.append((ts, dur, e["name"]))
    elif cat in ("cuda_runtime", "cuda_driver"):
        extern.append((ts, dur, e["name"]))

if not kernels:
    print("NO GPU KERNELS in trace (profiling GPU activity may be unsupported on this backend)")
    sys.exit(0)

start = min(k[0] for k in kernels)
end = max(k[0] + k[1] for k in kernels)
wall = end - start
print(f"kernel span: {wall/1e6:.2f}s, kernels: {len(kernels)}")

# GPU busy time: union of kernel intervals
iv = sorted((k[0], k[0] + k[1]) for k in kernels)
busy = 0
cur_s, cur_e = iv[0]
for s, e2 in iv[1:]:
    if s > cur_e:
        busy += cur_e - cur_s
        cur_s, cur_e = s, e2
    else:
        cur_e = max(cur_e, e2)
busy += cur_e - cur_s
print(f"GPU busy: {busy/1e6:.2f}s = {busy/wall*100:.1f}% of span  (idle {100-busy/wall*100:.1f}%)")

# top kernels by total time
agg = defaultdict(lambda: [0, 0.0])
for ts, dur, name, stream in kernels:
    a = agg[name]
    a[0] += 1
    a[1] += dur
top = sorted(agg.items(), key=lambda x: -x[1][1])[:30]
print("\n=== TOP GPU KERNELS (by total us) ===")
for name, (cnt, tot) in top:
    print(f"{tot:>12.0f} us  n={cnt:<6} avg={tot/cnt:>9.1f} us  {name[:110]}")

# top cpu ops
aggc = defaultdict(lambda: [0, 0.0])
for ts, dur, name in cpu_ops:
    a = aggc[name]
    a[0] += 1
    a[1] += dur
topc = sorted(aggc.items(), key=lambda x: -x[1][1])[:20]
print("\n=== TOP CPU OPS (by total us) ===")
for name, (cnt, tot) in topc:
    print(f"{tot:>12.0f} us  n={cnt:<6} avg={tot/cnt:>9.1f} us  {name[:100]}")

# launch overhead
if extern:
    agge = defaultdict(lambda: [0, 0.0])
    for ts, dur, name in extern:
        a = agge[name]
        a[0] += 1
        a[1] += dur
    tope = sorted(agge.items(), key=lambda x: -x[1][1])[:10]
    print("\n=== TOP RUNTIME/DRIVER CALLS ===")
    for name, (cnt, tot) in tope:
        print(f"{tot:>12.0f} us  n={cnt:<6} avg={tot/cnt:>9.1f} us  {name[:100]}")
