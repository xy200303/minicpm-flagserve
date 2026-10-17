"""Greedy parity for the A20 decode kernel: long prompts, temp 0, compare text.

Usage: python3 /workspace/parity_a20b.py <out_file>
"""
import json
import sys
import urllib.request

FILLER = [
    "The principle of locality states that a program tends to reuse the same "
    "memory locations and the same instructions in the near future. ",
    "def merge_sort(a):\n    if len(a) <= 1:\n        return a\n    mid = len(a) // 2\n",
    "Consider a particle moving in a potential well; its energy is quantized "
    "and the wavefunction vanishes at the boundaries of the well. ",
]

PROMPTS = []
for reps in (400, 1100, 60):
    PROMPTS.append((FILLER[0] * reps)[: reps * 60])
PROMPTS.append((FILLER[1] * 500)[: 20000])
PROMPTS.append((FILLER[2] * 900)[: 36000])


def gen(prompt, max_tokens=48):
    body = json.dumps({
        "model": "minicpm", "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "top_p": 1.0, "seed": 0,
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:9031/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["choices"][0]["text"]


def main(out):
    with open(out, "w") as f:
        for i, p in enumerate(PROMPTS):
            t = gen(p)
            f.write(f"=== prompt {i} ({len(p)} chars) ===\n{t}\n")
            print(f"prompt {i}: {len(p)} chars -> {len(t)} chars, "
                  f"head={t[:40]!r}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/parity_a20b.txt")
