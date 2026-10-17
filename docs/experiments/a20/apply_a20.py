#!/usr/bin/env python3
"""Wire A20 (self-developed decode-attention kernel) into the installed plugin.

Copies the kernel module and inserts the apply() hook into the Iluvatar
backend init, in BOTH the repo checkout (/workspace/vllm-plugin-FL) and the
installed package site-packages/vllm_fl, because the runtime imports the
latter.

Usage: python3 apply_a20.py [/workspace/vllm-plugin-FL]
       (run again on a fresh instance; idempotent)
"""
import os
import shutil
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else "/workspace/vllm-plugin-FL"
SITE = "/usr/local/lib/python3.12/site-packages/vllm_fl"
HERE = os.path.dirname(os.path.abspath(__file__))

REL_IMPL = "dispatch/backends/vendor/iluvatar/impl"
REL_INIT = "dispatch/backends/vendor/iluvatar/iluvatar.py"

HOOK = '''# fl opt-A20: self-developed paged decode-attention kernel.
# One warp per (seq, kv_head, segment), fixed segment count with the segment
# length derived from seq_lens on device (CUDA-graph safe), base-2 softmax,
# unmasked main loop + masked tail. 0.72 ms/layer vs 1.04 ms stock at the
# official 4k shape (max|diff| 2e-4 vs stock).
# A/B switch: VLLM_FL_DISABLE_DECODE_FAST=1.
try:
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.decode_attn_fast import (
        apply as _apply_decode_attn_fast,
    )
    _apply_decode_attn_fast()
except Exception as _e:
    logger.warning("decode_attn_fast patch failed: %s", _e)

'''


def patch_tree(root: str) -> None:
    impl = os.path.join(root, REL_IMPL)
    init = os.path.join(root, REL_INIT)
    if not os.path.isdir(impl):
        print(f"skip {root}: not a plugin tree")
        return
    src = os.path.join(HERE, "decode_attn_fast.py")
    shutil.copyfile(src, os.path.join(impl, "decode_attn_fast.py"))
    with open(init, encoding="utf-8") as f:
        text = f.read()
    if "decode_attn_fast" not in text:
        anchor = "# fl opt-A5: fused sampler tail"
        assert anchor in text, "anchor not found in iluvatar.py"
        text = text.replace(anchor, HOOK + anchor, 1)
        with open(init, "w", encoding="utf-8") as f:
            f.write(text)
    print(f"patched {root}")


patch_tree(REPO)
patch_tree(SITE)
print("A20 applied; rebuild the serve (plugin is imported at process start)")
