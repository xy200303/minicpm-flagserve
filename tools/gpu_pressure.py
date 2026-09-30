#!/usr/bin/env python3
"""GPU pressure probe: find which GPU activity gets the process killed
on the 64GB instances.  Stages, each logged to /workspace/gpu_pressure.log:

  S1  torch.cuda context + tiny matmul (baseline sanity)
  S2  VRAM climb: allocate GPU memory in 2GB steps up to 54GB, release
  S3  stream/queue climb: create torch.cuda.Stream one by one, launch a
      small kernel on each — mirrors driver queue-block allocation
  S4  sequential CUDA subprocesses: spawn child procs that each create a
      CUDA context (mirrors `vllm bench` children)

Usage: python3 gpu_pressure.py [s1|s2|s3|s4|all]
The last log line before death is the culprit stage.
"""
import datetime
import subprocess
import sys
import time

STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"
LOG = "/workspace/gpu_pressure.log"

log = open(LOG, "a", buffering=1)


def say(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    log.write(f"{ts} {msg}\n")
    print(msg, flush=True)


import torch  # noqa: E402

say(f"=== probe start stage={STAGE} torch={torch.__version__}")

# ---------------- S1 ----------------
if STAGE in ("s1", "all"):
    a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    c = a @ a
    torch.cuda.synchronize()
    say(f"S1 cuda ctx + matmul OK mean={c.float().mean().item():.4f}")

# ---------------- S2 ----------------
if STAGE in ("s2", "all"):
    blocks = []
    for i in range(1, 28):
        blocks.append(torch.empty(2 * 1024**3, dtype=torch.uint8, device="cuda"))
        torch.cuda.synchronize()
        say(f"S2 vram {i * 2}GB allocated OK")
        time.sleep(0.5)
    del blocks
    torch.cuda.empty_cache()
    say("S2 vram climb OK, released")

# ---------------- S3 ----------------
if STAGE in ("s3", "all"):
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    streams = []
    for i in range(1, 129):
        s = torch.cuda.Stream()
        streams.append(s)
        with torch.cuda.stream(s):
            b = a @ a
        say(f"S3 stream #{i} created + kernel launched")
        time.sleep(0.2)
    torch.cuda.synchronize()
    say("S3 128 streams OK")

# ---------------- S4 ----------------
if STAGE in ("s4", "all"):
    for i in range(1, 9):
        r = subprocess.run(
            [sys.executable, "-c",
             "import torch; x=torch.zeros(2000,2000,device='cuda'); "
             "print('ctx ok')"],
            capture_output=True, text=True, timeout=180)
        say(f"S4 child #{i} rc={r.returncode} out={r.stdout.strip()[-40:]} "
            f"err={r.stderr.strip()[-80:]}")
        time.sleep(1)
    say("S4 8 cuda subprocesses OK")

say("=== ALL STAGES SURVIVED ===")
