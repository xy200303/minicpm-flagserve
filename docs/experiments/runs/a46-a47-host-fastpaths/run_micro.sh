#!/bin/bash
set -euo pipefail
export VLLM_PLUGINS=fl
ROOT=/workspace/experiments/runs/a46-a47-micro
mkdir -p "$ROOT"
for variant in a46 a47; do
  root=/workspace/${variant}-gems
  export PYTHONPATH=/workspace/audit-plugin:$root/src
  cd "$root"
  /workspace/audit-venv/bin/python -m pytest -q \
    tests/test_libentry.py::test_libentry_key_preserves_tensor_and_integer_specialization \
    tests/test_libentry.py::test_decorator_cascade \
    tests/test_libentry.py::test_pass_kernel_arg_via_kw \
    tests/test_libentry.py::test_kernel_arg_apply_default \
    tests/test_copy.py > "$ROOT/$variant-pytest.log" 2>&1
  tail -8 "$ROOT/$variant-pytest.log"
done
for spec in a31-1:a31 a46-1:a46 a47-1:a47 a47-2:a47 a46-2:a46 a31-2:a31; do
  tag=${spec%%:*}
  variant=${spec#*:}
  export PYTHONPATH=/workspace/audit-plugin:/workspace/${variant}-gems/src
  sweep=()
  if [ "$variant" = a47 ]; then sweep=(--sweep); fi
  /workspace/audit-venv/bin/python /workspace/bi_a46_libentry_probe.py \
    --output "$ROOT/$tag.json" "${sweep[@]}" > "$ROOT/$tag.log" 2>&1
  tail -14 "$ROOT/$tag.log"
done
