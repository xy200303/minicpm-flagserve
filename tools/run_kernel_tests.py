#!/usr/bin/env python3
"""L1 unit-test runner for all custom MiniCPM-FlagServe kernels.

Runs each kernel's validation suite and reports PASS/FAIL per suite.
Usage on the dev box:  python3 /workspace/run_kernel_tests.py
"""
import subprocess
import sys

SUITES = [
    # (name, script, must_contain, must_not_contain_extra)
    ("top_k_top_p (sort-free threshold)", "/workspace/test_top_k_top_p.py", "ALL PASS", ()),
    ("gumbel_max_sample (fused sampling)", "/workspace/test_gumbel.py", "OK (ours", ("SUSPECT",)),
    ("nt_db GEMM (double-buffered)", "/workspace/verify_nt_db.py", "done", ("BAD",)),
    ("nn_db GEMM (nn-layout double-buffered)", "/workspace/verify_nn_db.py", "ALL PASS", ("BAD",)),
]


def main() -> int:
    failed = []
    for name, script, must_contain, must_not in SUITES:
        print(f"\n{'='*60}\n[SUITE] {name}\n{'='*60}", flush=True)
        r = subprocess.run(
            [sys.executable, script], capture_output=True, text=True, timeout=900
        )
        out = r.stdout + r.stderr
        interesting = [
            ln for ln in out.splitlines()
            if any(k in ln for k in ("PASS", "FAIL", "OK", "BAD", "SUSPECT",
                                     "violation", "determinism", "edge",
                                     "perf", "speedup", "relerr", "TV", "Traceback"))
        ]
        print("\n".join(interesting[-15:]))
        bad = (
            r.returncode != 0
            or "FAIL" in out
            or "Traceback" in out
            or (must_contain and must_contain not in out)
            or any(m in out for m in must_not)
        )
        if bad:
            failed.append(name)
        print(f"[{'FAIL' if bad else 'PASS'}] {name}")
    print(f"\n{'='*60}\nRESULT: {'FAIL: ' + ', '.join(failed) if failed else 'ALL SUITES PASS'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
