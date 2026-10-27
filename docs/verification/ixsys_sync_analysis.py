import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path


def analyze(path, limit):
    conn = sqlite3.connect(path)
    syncs = conn.execute(
        """
        SELECT id, name, ts, dur, threadId FROM cuda_api
        WHERE name IN ('cudaEventSynchronize', 'cudaStreamSynchronize')
          AND dur > 1000000
        ORDER BY dur DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    sync_patterns = Counter()
    sync_details = []
    for ident, name, ts, dur, thread in syncs:
        next_kernel = conn.execute(
            "SELECT name, ts, dur FROM cuda_kernel WHERE ts >= ? ORDER BY ts LIMIT 1",
            (ts,),
        ).fetchone()
        next_api = conn.execute(
            "SELECT name, ts, dur FROM cuda_api WHERE threadId = ? AND ts >= ? ORDER BY ts LIMIT 1",
            (thread, ts + dur),
        ).fetchone()
        key = (name, next_api[0] if next_api else "trace_end",
               next_kernel[0] if next_kernel else "trace_end")
        sync_patterns[key] += 1
        if len(sync_details) < 30:
            sync_details.append({
                "api": name,
                "wait_ms": dur / 1e6,
                "next_api": next_api[0] if next_api else None,
                "next_kernel": next_kernel[0] if next_kernel else None,
                "next_kernel_delay_ms": (next_kernel[1] - ts - dur) / 1e6 if next_kernel else None,
            })
    kernels = conn.execute(
        "SELECT name, ts, dur FROM cuda_kernel ORDER BY ts"
    ).fetchall()
    gap_counts = Counter()
    gap_ms = Counter()
    gap_details = []
    for previous, current in zip(kernels, kernels[1:]):
        gap = current[1] - previous[1] - previous[2]
        if gap < 1e6:
            continue
        key = (previous[0], current[0])
        gap_counts[key] += 1
        gap_ms[key] += gap / 1e6
        if gap >= 1e7 and len(gap_details) < 30:
            gap_start = previous[1] + previous[2]
            active_apis = conn.execute(
                """
                SELECT name, ts, dur FROM cuda_api
                WHERE ts <= ? AND ts + dur > ?
                ORDER BY dur DESC LIMIT 4
                """,
                (gap_start, gap_start),
            ).fetchall()
            gap_details.append({
                "before": previous[0],
                "after": current[0],
                "gap_ms": gap / 1e6,
                "before_dur_ms": previous[2] / 1e6,
                "after_dur_ms": current[2] / 1e6,
                "active_apis": [
                    {"name": api[0], "start_ms": (api[1] - gap_start) / 1e6,
                     "dur_ms": api[2] / 1e6}
                    for api in active_apis
                ],
            })
    gap_details.sort(key=lambda item: item["gap_ms"], reverse=True)
    conn.close()
    return {
        "file": str(path),
        "sync_patterns": [
            {"sync": key[0], "next_api": key[1], "next_kernel": key[2], "count": count}
            for key, count in sync_patterns.most_common(30)
        ],
        "longest_sync": sync_details,
        "gap_patterns": [
            {"before": key[0], "after": key[1], "count": gap_counts[key],
             "total_gap_ms": total}
            for key, total in gap_ms.most_common(30)
        ],
        "longest_gaps": gap_details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    result = analyze(args.trace, args.limit)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
