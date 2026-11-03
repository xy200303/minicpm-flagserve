import dataclasses
import json
import os
import statistics
import time

import torch

from vllm_fl.dispatch.backends.vendor.iluvatar.impl.vendor_flash_attn import _build_split


@dataclasses.dataclass
class Meta:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    num_actual_tokens: int
    max_query_len: int
    _fl_split: object = None
    _fl_pages: object = None
    _fl_cu_q: object = None
    _fl_cu_k: object = None


def original_split(m, cu_l, seq_l):
    pre_idx, dec_idx = [], []
    for i in range(len(seq_l)):
        (dec_idx if cu_l[i + 1] - cu_l[i] == 1 else pre_idx).append(i)
    dev = m.query_start_loc.device
    if dec_idx:
        n = len(dec_idx)
        dec_rows = m.query_start_loc[dec_idx]
        dec_meta = dataclasses.replace(
            m,
            num_actual_tokens=n,
            max_query_len=1,
            query_start_loc=torch.arange(0, n + 1, device=dev, dtype=m.query_start_loc.dtype),
            seq_lens=m.seq_lens[dec_idx],
            block_table=m.block_table[dec_idx],
        )
    else:
        dec_rows = None
        dec_meta = None
    m._fl_split = (cu_l, seq_l, pre_idx, dec_rows, dec_meta)
    m._fl_pages = {}
    if pre_idx:
        dev32 = dict(dtype=torch.int32, device=dev)
        m._fl_cu_q = torch.tensor([[0, cu_l[i + 1] - cu_l[i]] for i in pre_idx], **dev32)
        m._fl_cu_k = torch.tensor([[0, seq_l[i]] for i in pre_idx], **dev32)
    else:
        m._fl_cu_q = None
        m._fl_cu_k = None


def measure(fn, pre_pos, inner=30, outer=9):
    samples = []
    for _ in range(outer):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for j in range(inner):
            pre = (pre_pos + j) % 64
            lengths = [1] * 64
            lengths[pre] = 2048
            cu = [0]
            for length in lengths:
                cu.append(cu[-1] + length)
            seq = [16384] * 64
            m = Meta(
                torch.tensor(cu, dtype=torch.int32, device="cuda"),
                torch.tensor(seq, dtype=torch.int32, device="cuda"),
                torch.arange(64 * 8192, dtype=torch.int32, device="cuda").reshape(64, 8192),
                2111,
                2048,
            )
            fn(m, cu, seq)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / inner)
    return statistics.median(samples), samples


def main():
    results = {}
    for pre_pos in (0, 32, 63):
        original, original_samples = measure(original_split, pre_pos)
        optimized, optimized_samples = measure(_build_split, pre_pos)
        results[str(pre_pos)] = {
            "original_ms": original,
            "optimized_ms": optimized,
            "speedup": original / optimized,
            "original_samples_ms": original_samples,
            "optimized_samples_ms": optimized_samples,
        }
        print(pre_pos, results[str(pre_pos)], flush=True)
    out = "/workspace/experiments/runs/a45-metadata-probe/results.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
