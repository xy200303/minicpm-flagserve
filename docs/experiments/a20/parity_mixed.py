"""Mixed-step greedy parity: concurrent requests with different prompt lengths
so prefill and decode share steps (exercises the A20 decode-slice path)."""
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

FILLER = ("The principle of locality states that a program tends to reuse the "
          "same memory locations and the same instructions in the near future. ")


def one(args):
    i, nchars = args
    prompt = (FILLER * (nchars // len(FILLER) + 1))[:nchars]
    body = json.dumps({
        "model": "minicpm", "prompt": prompt, "max_tokens": 24,
        "temperature": 0.0, "top_p": 1.0, "seed": 0, "ignore_eos": True,
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:9031/v1/completions",
                                data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return i, json.load(r)["choices"][0]["text"]


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/parity_mixed.txt"
    lens = [12000, 4000, 8000, 2000, 16000, 4000, 2000, 24000]
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(one, enumerate(lens)))
    with open(out, "w") as f:
        for i, t in sorted(res):
            f.write(f"=== req {i} ({lens[i]} chars) ===\n{t}\n")
            print(f"req {i}: {t[:50]!r}", flush=True)
