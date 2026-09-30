#!/usr/bin/env python3
"""Lightweight single-process throughput client for the cursed host.

The official benchmark driver respawns a 12GB+ `vllm bench serve` child per
run, which gets the serve SIGKILLed on the 64GB instances.  This client issues
the same load (64-way concurrency, fixed output length, ignore_eos) from one
small process and accounts tokens from the response `usage` fields.

Usage: python3 mini_bench.py [input_len] [output_len] [concurrency] [num_prompts] [port]
"""
import asyncio
import sys
import time

import aiohttp

INPUT_LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
OUTPUT_LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 64
NUM = int(sys.argv[4]) if len(sys.argv) > 4 else 256
PORT = int(sys.argv[5]) if len(sys.argv) > 5 else 9031
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 13
URL = f"http://127.0.0.1:{PORT}/v1/completions"

# deterministic pseudo-random prompt; "wNNNNN" words measure ~3.0 tokens each
# on the MiniCPM5 tokenizer, so n/3 words ~= n tokens
def make_prompt(n_words, seed):
    words = []
    x = seed
    for _ in range(max(1, n_words // 3)):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        words.append(f"w{x % 100000:05d}")
    return " ".join(words)


async def one(session, sem, idx, results):
    prompt = make_prompt(INPUT_LEN, idx * 7919 + SEED)
    body = {
        "model": "minicpm",
        "prompt": prompt,
        "max_tokens": OUTPUT_LEN,
        "ignore_eos": True,
        "temperature": 1.0,
        "top_p": 0.95,
    }
    async with sem:
        t0 = time.time()
        try:
            async with session.post(URL, json=body,
                                    timeout=aiohttp.ClientTimeout(total=1200)) as r:
                data = await r.json()
            dt = time.time() - t0
            usage = data.get("usage", {})
            results.append((dt, usage.get("prompt_tokens", 0),
                            usage.get("completion_tokens", 0)))
        except Exception as e:
            results.append((None, 0, 0))
            if idx < 3:
                print("REQ-ERR", type(e).__name__, flush=True)


async def main():
    sem = asyncio.Semaphore(CONC)
    results = []
    async with aiohttp.ClientSession() as session:
        t0 = time.time()
        await asyncio.gather(*(one(session, sem, i, results) for i in range(NUM)))
        wall = time.time() - t0
    ok = [r for r in results if r[0] is not None]
    tin = sum(r[1] for r in ok)
    tout = sum(r[2] for r in ok)
    print(f"ok={len(ok)}/{NUM} wall={wall:.1f}s "
          f"input={tin} output={tout} "
          f"total_tok/s={(tin + tout) / wall:.1f} output_tok/s={tout / wall:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
