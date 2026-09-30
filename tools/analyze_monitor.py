#!/usr/bin/env python3
"""Correlate monitor.log power/CPU with bench run boundaries (power-detected)."""
import re
from datetime import datetime

rows = []
t = None
pw = temp = load = None
for line in open("/workspace/monitor.log"):
    if line.startswith("=== "):
        if t and pw:
            rows.append((t, pw, temp, load))
        t = line.split()[1]
        pw = temp = load = None
    elif "W / 350W" in line:
        m = re.search(r"\|\s*(\d+)W / 350W\s*\|\s*(\d+)C", line)
        if m:
            pw, temp = int(m.group(1)), int(m.group(2))
    elif re.match(r"^\d", line) and "load" not in line:
        load = float(line.split()[0])
if t and pw:
    rows.append((t, pw, temp, load))

# detect busy segments (power > 120W)
print("time     watts temp load  note")
busy = False
start = None
segs = []
for t, pw, temp, load in rows:
    if pw and pw > 120 and not busy:
        busy, start = True, t
    elif pw and pw <= 120 and busy:
        busy = False
        segs.append((start, t))
for s, e in segs:
    print(f"BUSY {s} -> {e}")

# per-segment stats
def to_sec(t):
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s

for i, (s, e) in enumerate(segs):
    seg = [r for r in rows if to_sec(s) <= to_sec(r[0]) <= to_sec(e)]
    watts = [r[1] for r in seg if r[1]]
    loads = [r[3] for r in seg if r[3]]
    temps = [r[2] for r in seg if r[2]]
    if watts:
        print(f"seg{i+1} {s}-{e}: dur={to_sec(e)-to_sec(s)}s avgW={sum(watts)/len(watts):.0f} "
              f"minW={min(watts)} maxTemp={max(temps)} avgLoad={sum(loads)/len(loads):.2f} maxLoad={max(loads):.2f}")
