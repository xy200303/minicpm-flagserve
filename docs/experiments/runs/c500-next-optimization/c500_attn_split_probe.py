import argparse
import json
import statistics

import torch
from flash_attn import flash_attn_with_kvcache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", type=int, required=True)
    parser.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    batch, hq, hkv, dim, block = 64, 16, 2, 128, 16
    blocks_per_seq = (args.ctx + block - 1) // block
    num_blocks = batch * blocks_per_seq + 8
    torch.manual_seed(20261101)
    k_cache = torch.randn(num_blocks, block, hkv, dim, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn(num_blocks, block, hkv, dim, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(batch, 1, hq, dim, device="cuda", dtype=torch.bfloat16)
    block_table = torch.arange(num_blocks - 8, device="cuda", dtype=torch.int32)
    block_table = block_table[torch.randperm(num_blocks - 8, device="cuda")]
    block_table = block_table[:batch * blocks_per_seq].reshape(batch, blocks_per_seq)
    seqlens = torch.full((batch,), args.ctx, device="cuda", dtype=torch.int32)

    def run(split):
        return flash_attn_with_kvcache(
            q=q, k_cache=k_cache, v_cache=v_cache, block_table=block_table,
            cache_seqlens=seqlens, softmax_scale=dim ** -0.5,
            causal=True, num_splits=split,
        )

    results = []
    reference = None
    for split in args.splits:
        for _ in range(5):
            out = run(split)
        torch.cuda.synchronize()
        if reference is None:
            reference = out.clone()
        samples = []
        for _ in range(args.reps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = run(split)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        results.append({"num_splits": split, "median_ms": statistics.median(samples),
                        "max_diff": (out - reference).abs().max().item(),
                        "samples_ms": samples})
    baseline = next(row["median_ms"] for row in results if row["num_splits"] == 0)
    best = min(results, key=lambda row: row["median_ms"])
    payload = {"ctx": args.ctx, "reps": args.reps, "baseline_ms": baseline,
               "best": best, "best_speedup": baseline / best["median_ms"],
               "results": results}
    text = json.dumps(payload, indent=2)
    print(text)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
