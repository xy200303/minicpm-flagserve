import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch
import triton

import flag_gems
from flag_gems.fused import paged_decode_attention
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
import vllm.v1.attention.ops.triton_unified_attention as tua


def git_commit(path):
    try:
        root = Path(path).resolve().parents[2]
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def summarize(samples):
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def run_case(kv, ragged, stages, reps, split_tail=False, batch=64, unroll_pages=1):
    hq, hkv, head_dim, block_size = 16, 2, 128, 16
    blocks_per_seq = (kv + block_size - 1) // block_size
    pages = batch * blocks_per_seq
    torch.manual_seed(20261027 + kv + int(ragged))
    q = torch.randn(batch, hq, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        pages, block_size, hkv, head_dim, device="cuda", dtype=torch.bfloat16
    ) * 0.1
    v = torch.randn(
        pages, block_size, hkv, head_dim, device="cuda", dtype=torch.bfloat16
    ) * 0.1
    block_table = torch.randperm(
        pages, dtype=torch.int32, device="cuda"
    ).view(batch, blocks_per_seq).contiguous()
    if ragged:
        lens = torch.tensor(
            [kv - (i % 17) * 179 for i in range(batch)],
            dtype=torch.int32,
            device="cuda",
        )
    else:
        lens = torch.full((batch,), kv, dtype=torch.int32, device="cuda")
    cu = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    scale = head_dim**-0.5
    ref = torch.empty_like(q)
    seg_out = torch.empty((batch, hq, 16, head_dim), dtype=torch.float32, device="cuda")
    seg_max = torch.empty((batch, hq, 16), dtype=torch.float32, device="cuda")
    seg_sum = torch.empty((batch, hq, 16), dtype=torch.float32, device="cuda")

    kernel = tua.kernel_unified_attention

    class Pin:
        def __getitem__(self, grid):
            fn = kernel[grid]

            def call(**kwargs):
                if kwargs.get("IS_3D"):
                    kwargs["num_warps"] = 1
                    kwargs["num_stages"] = 1
                return fn(**kwargs)

            return call

    tua.kernel_unified_attention = Pin()
    unified_attention(
        q=q,
        k=k,
        v=v,
        out=ref,
        cu_seqlens_q=cu,
        max_seqlen_q=1,
        seqused_k=lens,
        max_seqlen_k=kv,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=64,
        num_par_softmax_segments=16,
        softmax_segm_output=seg_out,
        softmax_segm_max=seg_max,
        softmax_segm_expsum=seg_sum,
    )

    outputs = {stage: torch.empty_like(q) for stage in stages}

    def run(stage):
        paged_decode_attention(
            q,
            k,
            v,
            outputs[stage],
            block_table,
            lens,
            scale,
            num_segments=32,
            block_n=block_size,
            num_warps=1,
            num_stages=stage,
            **({"split_tail": True} if split_tail else {}),
            **({"unroll_pages": unroll_pages} if unroll_pages != 1 else {}),
        )

    correctness = {}
    for stage in stages:
        run(stage)
        torch.cuda.synchronize()
        diff = (outputs[stage].float() - ref.float()).abs()
        correctness[str(stage)] = {
            "max_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "finite": bool(torch.isfinite(outputs[stage]).all().item()),
        }

    for _ in range(5):
        for stage in stages:
            run(stage)
    torch.cuda.synchronize()
    samples = {stage: [] for stage in stages}
    for rep in range(reps):
        order = stages[rep % len(stages):] + stages[:rep % len(stages)]
        for stage in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run(stage)
            end.record()
            end.synchronize()
            samples[stage].append(start.elapsed_time(end))

    timings = {str(stage): summarize(samples[stage]) for stage in stages}
    baseline_stage = 1 if 1 in stages else stages[0]
    baseline = timings[str(baseline_stage)]["median_ms"]
    tokens = int(lens.sum().item())
    kv_gb = tokens * hkv * head_dim * 2 * 2 / 1e9
    return {
        "kv": kv,
        "ragged": ragged,
        "batch": batch,
        "kv_gb_read": kv_gb,
        "correctness": correctness,
        "timings": timings,
        "speedup_vs_stage1": {
            str(stage): baseline / timings[str(stage)]["median_ms"] for stage in stages
        },
        "effective_gb_s": {
            str(stage): kv_gb / (timings[str(stage)]["median_ms"] / 1e3)
            for stage in stages
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--kv", type=int, nargs="+", default=[4096, 16384])
    parser.add_argument("--batch", type=int, nargs="+", default=[64])
    parser.add_argument("--split-tail", action="store_true")
    parser.add_argument("--unroll-pages", type=int, default=1, choices=[1, 2])
    args = parser.parse_args()
    if (args.reps < 1 or not args.stages or not args.batch
            or any(stage < 1 for stage in args.stages)
            or any(batch < 1 for batch in args.batch)):
        parser.error("reps, stages and batch must be positive")
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "flag_gems_source": str(Path(flag_gems.__file__).resolve()),
        "flag_gems_commit": git_commit(flag_gems.__file__),
        "reps": args.reps,
        "stages": args.stages,
        "num_segments": 32,
        "block_n": 16,
        "num_warps": 1,
        "split_tail": args.split_tail,
        "batch": args.batch,
        "unroll_pages": args.unroll_pages,
        "cases": [
            run_case(kv, ragged, args.stages, args.reps, args.split_tail, batch,
                     args.unroll_pages)
            for batch in args.batch
            for kv in args.kv
            for ragged in (False, True)
        ],
    }
    text = json.dumps(result, indent=2)
    print(text)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
