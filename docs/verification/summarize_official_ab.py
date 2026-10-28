import argparse
import csv
import json
import statistics
from pathlib import Path


def read_summary(run_dir):
    run_dir = Path(run_dir)
    direct = run_dir / "summary.csv"
    candidates = [direct] if direct.is_file() else sorted(
        (run_dir / "benchmark_results").glob("summary_*.csv"))
    if len(candidates) != 1:
        raise ValueError(f"{run_dir}: expected exactly one complete summary")
    with candidates[0].open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"{candidates[0]}: expected one summary row")
    row = rows[0]
    config = run_dir / "config.txt"
    case = None
    if config.is_file():
        for line in config.read_text(encoding="utf-8").splitlines():
            if line.startswith("case="):
                case = line.removeprefix("case=")
                break
    return {
        "run": run_dir.name,
        "summary": str(candidates[0]),
        "case": case,
        "total_tok_s": float(row["Total tok/s"]),
        "mean_ttft_ms": float(row["Mean TTFT (ms)"]),
    }


def median(rows, key):
    return statistics.median(row[key] for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs=2, required=True)
    parser.add_argument("--candidate", nargs=2, required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    baseline = [read_summary(path) for path in args.baseline]
    candidate = [read_summary(path) for path in args.candidate]
    cases = {row["case"] for row in baseline + candidate}
    if len(cases) != 1 or None in cases:
        raise ValueError(f"all runs must have the same recorded official case: {cases}")

    baseline_tok_s = median(baseline, "total_tok_s")
    candidate_tok_s = median(candidate, "total_tok_s")
    baseline_ttft_ms = median(baseline, "mean_ttft_ms")
    candidate_ttft_ms = median(candidate, "mean_ttft_ms")
    result = {
        "case": cases.pop(),
        "baseline": baseline,
        "candidate": candidate,
        "baseline_median_total_tok_s": baseline_tok_s,
        "candidate_median_total_tok_s": candidate_tok_s,
        "throughput_change_pct": (candidate_tok_s / baseline_tok_s - 1) * 100,
        "baseline_median_ttft_ms": baseline_ttft_ms,
        "candidate_median_ttft_ms": candidate_ttft_ms,
        "ttft_limit_ms": baseline_ttft_ms * 1.01,
        "throughput_at_least_3pct": candidate_tok_s >= baseline_tok_s * 1.03,
        "ttft_within_baseline_plus_1pct": candidate_ttft_ms <= baseline_ttft_ms * 1.01,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
