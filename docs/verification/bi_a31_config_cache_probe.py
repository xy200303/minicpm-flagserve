import argparse
import json
import time
from pathlib import Path

import torch

import flag_gems
from flag_gems.utils.models.sql import SQLPersistantModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--calls", type=int, default=500)
    args = parser.parse_args()
    flag_gems.enable()
    source = str(Path(flag_gems.__file__).resolve())
    original = SQLPersistantModel.get_config
    queries = 0

    def counted(self, name, keys):
        nonlocal queries
        queries += 1
        return original(self, name, keys)

    SQLPersistantModel.get_config = counted
    x = torch.arange(256, dtype=torch.int32, device="cuda")
    indices = torch.arange(64, dtype=torch.int64, device="cuda")
    for _ in range(10):
        x[indices]
    torch.cuda.synchronize()
    queries = 0
    start = time.perf_counter()
    for _ in range(args.calls):
        x[indices]
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    result = {
        "source": source,
        "calls": args.calls,
        "get_config_calls": queries,
        "elapsed_ms": elapsed_ms,
        "us_per_index": elapsed_ms * 1000 / args.calls,
    }
    text = json.dumps(result, indent=2)
    print(text)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
