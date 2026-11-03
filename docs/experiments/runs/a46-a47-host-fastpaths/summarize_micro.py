import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "a46-a47-micro"


def read(tag):
    with (ROOT / f"{tag}.json").open(encoding="utf-8") as f:
        data = json.load(f)
    assert data["correct"] is True
    assert f"/{tag.split('-')[0]}-gems/" in data["module"]
    return data


def main():
    variants = {}
    for variant in ("a31", "a46", "a47"):
        runs = [read(f"{variant}-{i}") for i in (1, 2)]
        variants[variant] = {
            "run_medians_ms": [run["median_ms"] for run in runs],
            "median_ms": statistics.median(run["median_ms"] for run in runs),
        }
    baseline = variants["a31"]["median_ms"]
    for variant in ("a46", "a47"):
        variants[variant]["speedup_vs_a31"] = baseline / variants[variant]["median_ms"]
    sweep = {}
    for tag in ("a47-1", "a47-2"):
        for config in read(tag)["sweep"]:
            key = f"block={config['block']},warps={config['num_warps']}"
            sweep.setdefault(key, []).append(config["median_ms"])
    result = {
        "variants": variants,
        "sweep_median_ms": {key: statistics.median(values) for key, values in sweep.items()},
    }
    text = json.dumps(result, indent=2)
    print(text)
    with (ROOT / "summary.json").open("w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
