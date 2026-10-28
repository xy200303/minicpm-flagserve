import argparse
import json
import time
from dataclasses import dataclass

import torch

from vllm_fl.dispatch.backends.vendor.iluvatar.impl.vendor_flash_attn import (
    _build_split,
)


@dataclass
class Metadata:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    num_actual_tokens: int = 0
    max_query_len: int = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--calls", type=int, default=1000)
    args = parser.parse_args()
    n = 64
    m = Metadata(
        query_start_loc=torch.arange(n + 1, dtype=torch.int32, device="cuda"),
        seq_lens=torch.arange(1, n + 1, dtype=torch.int32, device="cuda"),
        block_table=torch.arange(n, dtype=torch.int32, device="cuda"),
    )
    cu_l = list(range(n + 1))
    seq_l = [(i % 17) + 1 for i in range(n)]
    for _ in range(10):
        _build_split(m, cu_l, seq_l)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.calls):
        _build_split(m, cu_l, seq_l)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    _, _, pre_idx, dec_rows, dec_meta = m._fl_split
    result = {
        "calls": args.calls,
        "elapsed_ms": elapsed_ms,
        "us_per_call": elapsed_ms * 1000 / args.calls,
        "direct_seq_lens": dec_meta.seq_lens is m.seq_lens,
        "direct_block_table": dec_meta.block_table is m.block_table,
        "pre_idx": pre_idx,
        "dec_rows_correct": bool(torch.equal(dec_rows, torch.arange(n, device="cuda"))),
        "seq_lens_correct": bool(torch.equal(dec_meta.seq_lens, m.seq_lens)),
        "block_table_correct": bool(torch.equal(dec_meta.block_table, m.block_table)),
    }
    text = json.dumps(result, indent=2)
    print(text)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
