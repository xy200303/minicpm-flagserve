#!/bin/bash
# Build the submission package: XLANG2026.zip
#   XLANG2026/vllm-plugin-FL/   (repo@84abed0 + A20, committed)
#   XLANG2026/FlagGems/         (v5.3.5 + fused/paged_decode_attention.py, committed)
#   XLANG2026/readme.md
#   XLANG2026/report.pdf
#
# Run on the dev instance:  bash tools/pack_submission.sh
set -eu
ROOT=${1:-/workspace}
OUT=${2:-/tmp/pkg}

rm -rf "$OUT"
mkdir -p "$OUT/XLANG2026"
tar -C "$ROOT" -cf - \
    --exclude=.git --exclude=build --exclude='*.egg-info' \
    --exclude=__pycache__ --exclude='*.pyc' \
    vllm-plugin-FL FlagGems | tar -C "$OUT/XLANG2026" -xf -
cp "$ROOT/pkg_readme.md" "$OUT/XLANG2026/readme.md"
cp "$ROOT/pkg_report.pdf" "$OUT/XLANG2026/report.pdf"

python3 - "$OUT" <<'PY'
import os, sys, zipfile
out_dir = sys.argv[1]
root = os.path.join(out_dir, "XLANG2026")
zip_path = os.path.join(out_dir, "XLANG2026.zip")
skip_dirs = {".git", "build", "__pycache__", ".pytest_cache"}
n = 0
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in skip_dirs and not d.endswith(".egg-info")]
        for f in filenames:
            if f.endswith((".pyc", ".pyo")):
                continue
            full = os.path.join(dirpath, f)
            z.write(full, os.path.relpath(full, out_dir))
            n += 1
print(f"{zip_path}: {n} files, {os.path.getsize(zip_path)/1e6:.1f} MB")
PY
