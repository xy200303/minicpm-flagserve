"""fl opt-A11: uniform ngram_gpu speculation for cudagraph replay (experiment).

Two modes, selected by VLLM_FL_UNIFORM_SPEC_MODE:

* "padonly" (default): pad every request's draft slots to k so that all
  requests are scheduled with the same number of tokens (uniform decode, a
  prerequisite for FULL_DECODE_ONLY cudagraphs) *and* report k valid drafts so
  the stock trimming step has nothing to trim.  The scheduler's accounting
  path stays exactly as shipped.
* "notrim": the earlier, stronger variant that also disables
  update_scheduler_for_invalid_drafts (incl. its event synchronize).  Measured
  2026-10-16: this variant breaks scheduler/block accounting (requests churn in
  Waiting with ~2% KV usage at ~1 tok/s) regardless of cudagraph mode.

Padding value: the request's own last token.  Verification decides every token,
so accuracy is unaffected; a placeholder that happens to match is a legitimate
accepted draft.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_MODE = os.getenv("VLLM_FL_UNIFORM_SPEC_MODE", "padonly").strip().lower()
_DISABLED = os.getenv("VLLM_FL_DISABLE_UNIFORM_SPEC", "0") == "1"

_APPLIED = False
_STOCK_PROPOSE = None


def _propose_uniform(
    self,
    num_speculative_tokens,
    num_tokens_no_spec,
    token_ids_gpu,
    valid_sampled_token_ids_gpu,
    valid_sampled_tokens_count,
):
    draft_tokens, num_valid_draft_tokens = _STOCK_PROPOSE(
        self,
        num_speculative_tokens,
        num_tokens_no_spec,
        token_ids_gpu,
        valid_sampled_token_ids_gpu,
        valid_sampled_tokens_count,
    )
    if _DISABLED:
        return draft_tokens, num_valid_draft_tokens

    k = draft_tokens.shape[1]
    last_idx = (num_tokens_no_spec.to(torch.long) - 1).clamp(min=0)
    placeholder = token_ids_gpu.gather(1, last_idx.unsqueeze(1))
    draft_tokens = torch.where(draft_tokens == -1, placeholder, draft_tokens)

    if _MODE == "padonly":
        num_tokens_tmp = num_tokens_no_spec + valid_sampled_tokens_count
        eligible = (valid_sampled_tokens_count > 0) & (num_tokens_tmp >= self.min_n)
        num_valid_draft_tokens = torch.where(
            eligible, torch.full_like(num_valid_draft_tokens, k), num_valid_draft_tokens
        )
    return draft_tokens, num_valid_draft_tokens


def _trim_disabled(
    num_valid_draft_tokens_event,
    num_valid_draft_tokens_cpu,
    scheduler_output,
    req_id_to_index,
):
    return


def apply() -> bool:
    global _APPLIED, _STOCK_PROPOSE
    if _APPLIED:
        return True
    try:
        from vllm.v1.spec_decode import ngram_proposer_gpu
        from vllm.v1.worker import gpu_model_runner
    except Exception as exc:
        logger.warning("uniform_spec_decode: vllm modules unavailable: %s", exc)
        return False

    _STOCK_PROPOSE = ngram_proposer_gpu.NgramProposerGPU.propose
    ngram_proposer_gpu.NgramProposerGPU.propose = _propose_uniform

    if _MODE == "notrim":
        for mod in (gpu_model_runner,):
            if hasattr(mod, "update_scheduler_for_invalid_drafts"):
                mod.update_scheduler_for_invalid_drafts = _trim_disabled
        try:
            from vllm_fl.worker import model_runner as _fl_runner

            _fl_runner.update_scheduler_for_invalid_drafts = _trim_disabled
        except Exception:
            pass

    _APPLIED = True
    logger.warning(
        "fl opt-A11: uniform ngram_gpu speculation installed, mode=%s "
        "(VLLM_FL_DISABLE_UNIFORM_SPEC=1 to disable)",
        _MODE,
    )
    return True
