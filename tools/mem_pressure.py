#!/usr/bin/env python3
"""Memory pressure probe: allocate step-by-step, log every step to disk,
keep going until the kernel/platform kills us.  The last log line is the
kill threshold.

Usage: python3 mem_pressure.py [step_mb=512] [interval_s=1.0] [cap_gb=60]
Log:   /workspace/mem_pressure.log (line-buffered, survives SIGKILL)
"""
import datetime
import os
import sys
import time

STEP_MB = int(sys.argv[1]) if len(sys.argv) > 1 else 512
INTERVAL = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
CAP_GB = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
LOG = "/workspace/mem_pressure.log"


def meminfo():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            out[k] = int(v.strip().split()[0])  # kB
    return out


def cgroup_current():
    for p in ("/sys/fs/cgroup/memory.current",
              "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            with open(p) as f:
                return int(f.read().strip())
        except OSError:
            pass
    return -1


def main():
    blocks = []
    step = STEP_MB * 1024 * 1024
    page = 4096
    total_mb = 0
    log = open(LOG, "w", buffering=1)  # line buffered
    log.write(f"# start step={STEP_MB}MB interval={INTERVAL}s cap={CAP_GB}GB\n")
    while total_mb / 1024 < CAP_GB:
        b = bytearray(step)
        for i in range(0, step, page):
            b[i] = 1
        blocks.append(b)
        total_mb += STEP_MB
        mi = meminfo()
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        log.write(
            f"{ts} alloc={total_mb / 1024:.1f}GB "
            f"memfree={mi['MemAvailable'] / 1048576:.1f}GB "
            f"cgroup={cgroup_current() / 1073741824:.1f}GB\n"
        )
        time.sleep(INTERVAL)
    log.write(f"# reached cap {CAP_GB}GB, holding 60s...\n")
    time.sleep(60)
    log.write("# SURVIVED\n")


if __name__ == "__main__":
    main()
