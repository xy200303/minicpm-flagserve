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
# One warp per (seq, kv_head, 256-token segment); base-2 softmax; unmasked
# main loop + masked tail. 0.72 ms/layer vs 1.04 ms stock at the official
# 4k shape (max|diff| 2e-4). A/B switch: VLLM_FL_DISABLE_DECODE_FAST=1.
try:
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.decode_attn_fast import (
        apply as _apply_decode_attn_fast,
    )
    _apply_decode_attn_fast()
except Exception as _e:
    logger.warning("decode_attn_fast patch failed: %s", _e)

'''


VENDOR_IMPORT = """

try:  # A20: self-developed decode-attention kernel for the decode slice
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl import (
        decode_attn_fast as _decode_fast,
    )
except Exception:  # pragma: no cover
    _decode_fast = None"""

VENDOR_OLD = """    if dec_meta is not None:
        q_dec = q.index_select(0, dec_rows)
        out_dec = torch.empty_like(q_dec)
        _orig_forward(self, layer, q_dec, key[:n_tok], value[:n_tok], kv_cache,
                      dec_meta, out_dec)
        out.index_copy_(0, dec_rows, out_dec)"""

VENDOR_NEW = """    if dec_meta is not None:
        q_dec = q.index_select(0, dec_rows)
        out_dec = torch.empty_like(q_dec)
        # A20: decode rows of a mixed step go through the self-developed
        # kernel when the shape qualifies; otherwise stock unified attention.
        if not (_decode_fast is not None
                and _decode_fast.try_fast_decode(self, q_dec, kv_cache,
                                                 dec_meta, out_dec)):
            _orig_forward(self, layer, q_dec, key[:n_tok], value[:n_tok],
                          kv_cache, dec_meta, out_dec)
        out.index_copy_(0, dec_rows, out_dec)"""


def patch_vendor(root: str) -> None:
    path = os.path.join(root, REL_IMPL, "vendor_flash_attn.py")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if "_decode_fast" in text:
        return
    text = text.replace("logger = init_logger(__name__)",
                        "logger = init_logger(__name__)" + VENDOR_IMPORT, 1)
    assert VENDOR_OLD in text, "vendor_flash_attn decode branch not found"
    text = text.replace(VENDOR_OLD, VENDOR_NEW, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"patched vendor_flash_attn in {root}")


def patch_tree(root: str) -> None:
    # accept either the repo root (…/vllm-plugin-FL) or the package dir
    pkg = os.path.join(root, "vllm_fl")
    if os.path.isdir(os.path.join(pkg, REL_IMPL)):
        root = pkg
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
    patch_vendor(root)
    print(f"patched {root}")


patch_tree(REPO)
patch_tree(SITE)
print("A20 applied; rebuild the serve (plugin is imported at process start)")
