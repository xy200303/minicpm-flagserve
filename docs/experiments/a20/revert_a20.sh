#!/bin/bash
# Remove A20 from both trees (repo checkout + installed package).
set -u
for ROOT in /workspace/vllm-plugin-FL /usr/local/lib/python3.12/site-packages/vllm_fl; do
  INIT=$ROOT/dispatch/backends/vendor/iluvatar/iluvatar.py
  IMPL=$ROOT/dispatch/backends/vendor/iluvatar/impl/decode_attn_fast.py
  [ -f "$INIT" ] || continue
  python3 - "$INIT" <<'PY'
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
start = s.find("# fl opt-A20:")
end = s.find("# fl opt-A5: fused sampler tail")
if start != -1 and end > start:
    s = s[:start] + s[end:]
    open(p, "w", encoding="utf-8").write(s)
    print("reverted hook in", p)
PY
  rm -f "$IMPL"
done
echo "A20 reverted"
