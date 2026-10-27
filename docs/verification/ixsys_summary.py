import argparse
import json
import sqlite3
from pathlib import Path


def rows_to_dicts(cursor, rows):
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row)) for row in rows]


def query_dicts(conn, sql, params=()):
    cursor = conn.execute(sql, params)
    return rows_to_dicts(cursor, cursor.fetchall())


def summarize(path):
    conn = sqlite3.connect(path)
    kernels = query_dicts(
        conn,
        """
        SELECT name, COUNT(*) AS calls, SUM(dur) / 1e6 AS total_ms,
               AVG(dur) / 1e6 AS mean_ms, MIN(dur) / 1e6 AS min_ms,
               MAX(dur) / 1e6 AS max_ms
        FROM cuda_kernel GROUP BY name ORDER BY SUM(dur) DESC LIMIT 40
        """,
    )
    variants = query_dicts(
        conn,
        """
        SELECT name, grid, block, regs, staticSharedMemory, dynamicSharedMemory,
               COUNT(*) AS calls, SUM(dur) / 1e6 AS total_ms,
               AVG(dur) / 1e6 AS mean_ms
        FROM cuda_kernel
        GROUP BY name, grid, block, regs, staticSharedMemory, dynamicSharedMemory
        ORDER BY SUM(dur) DESC LIMIT 40
        """,
    )
    apis = query_dicts(
        conn,
        """
        SELECT name, COUNT(*) AS calls, SUM(dur) / 1e6 AS total_ms,
               AVG(dur) / 1e6 AS mean_ms, MAX(dur) / 1e6 AS max_ms
        FROM cuda_api GROUP BY name ORDER BY SUM(dur) DESC LIMIT 30
        """,
    )
    graphs = query_dicts(
        conn,
        """
        SELECT name, COUNT(*) AS calls, SUM(dur) / 1e6 AS total_ms,
               AVG(dur) / 1e6 AS mean_ms
        FROM cuda_graph GROUP BY name ORDER BY SUM(dur) DESC LIMIT 20
        """,
    )
    streams = []
    for (stream,) in conn.execute(
        "SELECT DISTINCT streamId FROM cuda_kernel ORDER BY streamId"
    ):
        events = conn.execute(
            "SELECT ts, dur FROM cuda_kernel WHERE streamId=? ORDER BY ts", (stream,)
        ).fetchall()
        gaps = []
        for previous, current in zip(events, events[1:]):
            gap = current[0] - previous[0] - previous[1]
            if gap > 0:
                gaps.append(gap)
        streams.append({
            "stream": stream,
            "kernel_calls": len(events),
            "kernel_ms": sum(event[1] for event in events) / 1e6,
            "gap_ms": sum(gaps) / 1e6,
            "max_gap_ms": max(gaps, default=0) / 1e6,
            "gaps_over_1ms": sum(gap > 1e6 for gap in gaps),
            "gaps_over_10ms": sum(gap > 1e7 for gap in gaps),
        })
    counts = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("cuda_kernel", "cuda_api", "cuda_graph", "cuda_memcpy")
    }
    conn.close()
    return {
        "file": str(path),
        "counts": counts,
        "top_kernels": kernels,
        "kernel_variants": variants,
        "top_apis": apis,
        "graphs": graphs,
        "streams": streams,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = {str(Path(trace)): summarize(trace) for trace in args.traces}
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
