#!/bin/bash
# Build XLANG2026.zip from two already-cloned, verified repositories.
#   XLANG2026/vllm-plugin-FL/  (integration/vllm-plugin-fl, with .git history)
#   XLANG2026/FlagGems/         (integration/flaggems, with .git history)
#   XLANG2026/readme.md
#   XLANG2026/report.pdf
#
# Usage on the dev instance: bash tools/pack_submission.sh [/workspace] [/tmp/pkg]
set -eu
ROOT=${1:-/workspace}
OUT=${2:-/tmp/pkg}

for repo in vllm-plugin-FL FlagGems; do
    test -d "$ROOT/$repo/.git"
    printf '%s: ' "$repo"
    git -C "$ROOT/$repo" log -1 --oneline
done
for required in \
    vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/metax/patches/mcblas_mm.py \
    vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/decode_attn_fast.py \
    FlagGems/src/flag_gems/fused/paged_decode_attention.py \
    FlagGems/src/flag_gems/fused/top_p_sample.py; do
    test -f "$ROOT/$required"
done

test -f "$ROOT/pkg_readme.md"
test -f "$ROOT/pkg_report.pdf"
rm -rf "$OUT"
mkdir -p "$OUT/XLANG2026"
# Keep .git: reviewers can inspect the development history inside the zip.
tar -C "$ROOT" -cf - \
    --exclude=build --exclude='*.egg-info' \
    --exclude=__pycache__ --exclude='*.pyc' \
    vllm-plugin-FL FlagGems | tar -C "$OUT/XLANG2026" -xf -
cp "$ROOT/pkg_readme.md" "$OUT/XLANG2026/readme.md"
cp "$ROOT/pkg_report.pdf" "$OUT/XLANG2026/report.pdf"

python3 - "$OUT" <<'PY'
import os, sys, zipfile
out_dir = sys.argv[1]
root = os.path.join(out_dir, "XLANG2026")
zip_path = os.path.join(out_dir, "XLANG2026.zip")
skip_dirs = {"build", "__pycache__", ".pytest_cache"}
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
