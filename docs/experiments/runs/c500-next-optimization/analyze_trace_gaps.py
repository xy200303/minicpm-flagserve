import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
skip_seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
    events = json.load(f)["traceEvents"]

raw = []
for event in events:
    if event.get("ph") == "X" and event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
        raw.append((event.get("ts", 0), event.get("dur", 0), event["name"]))
first_ts = min(ts for ts, _, _ in raw)
start = first_ts + int(skip_seconds * 1e6)
kernels = sorted((max(ts, start), ts + dur, name) for ts, dur, name in raw if ts + dur > start)

gaps = []
by_pair = defaultdict(lambda: [0, 0.0])
cur_end = start
cur_name = "START"
busy = 0
for ts, end, name in kernels:
    if ts > cur_end:
        gap = ts - cur_end
        gaps.append((cur_end - start, gap, cur_name, name))
        pair = by_pair[(cur_name, name)]
        pair[0] += 1
        pair[1] += gap
        cur_end = end
        cur_name = name
    elif end > cur_end:
        busy += cur_end - ts if ts > start else 0
        cur_end = end
        cur_name = name
    if ts <= cur_end and end == cur_end:
        pass

intervals = []
cur_start, cur_end = kernels[0][0], kernels[0][1]
for ts, end, _ in kernels[1:]:
    if ts > cur_end:
        intervals.append((cur_start, cur_end))
        cur_start, cur_end = ts, end
    else:
        cur_end = max(cur_end, end)
intervals.append((cur_start, cur_end))
busy = sum(end - ts for ts, end in intervals)
wall = intervals[-1][1] - start
print(f"skip: {skip_seconds:.1f}s, span: {wall/1e6:.2f}s, GPU busy: {busy/1e6:.2f}s ({busy/wall*100:.1f}%)")
print(f"gap count: {len(gaps)}, total: {sum(g[1] for g in gaps) / 1e6:.2f}s")
print("\n=== TOP GAPS BY PAIR ===")
for (before, after), (count, total) in sorted(by_pair.items(), key=lambda item: -item[1][1])[:25]:
    print(f"{total:>10.0f} us n={count:<5} avg={total/count:>8.1f} us\n  PREV {before[:105]}\n  NEXT {after[:105]}")
print("\n=== LARGEST INDIVIDUAL GAPS ===")
for ts, gap, before, after in sorted(gaps, key=lambda item: -item[1])[:35]:
    print(f"t={ts/1e6:>8.3f}s gap={gap:>9.1f} us\n  PREV {before[:105]}\n  NEXT {after[:105]}")
