"""Short-circuit `ModelRunner.forward` for EXTEND mode.

`ScheduleBatch.prepare_for_extend()` (which allocates KV slots) still runs.
Only the actual model-forward kernel path is replaced with a fake
`LogitsProcessorOutput`. That means:

  - KV cache slots get allocated as expected — subsequent DECODE calls see
    a properly-sized KV table with (uninitialised / garbage) values
  - No prefill kernels run, so bs × isl never drives OOM regardless of shape
  - bench_one_batch's `extend()` returns normally with fake next-token ids
  - bench_one_batch's `decode()` runs the real model on the "fake-prefilled"
    KV; kernel shapes match production, timings are meaningful

Use together with `layer_skip_patch.py` for per-layer target:
  - skip_prefill_patch → skip prefill entirely
  - layer_skip_patch   → identity-replace non-target DecoderLayer forwards

Env:
  LAYERWISE_SKIP_PREFILL=1 (default)  — enable patch
  LAYERWISE_SKIP_PREFILL=0            — disable
"""
import logging
import os

import torch

logger = logging.getLogger(__name__)


def _fake_extend_output(model_runner, forward_batch):
    """Build a minimally-populated ModelRunnerOutput that satisfies bench
    code reading `.logits_output.next_token_logits`.

    `next_token_logits` shape [batch_size, vocab_size] — one logit per seq
    at its last token (extend convention).
    """
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.model_runner import ModelRunnerOutput

    bs = forward_batch.batch_size
    hf = model_runner.model_config.hf_config
    # Multimodal / nested configs (e.g. Qwen3.5) keep vocab_size under text_config
    vocab = getattr(hf, "vocab_size", None) or getattr(
        getattr(hf, "text_config", None), "vocab_size", None
    )
    if vocab is None:
        raise AttributeError(
            "Could not resolve vocab_size on hf_config or hf_config.text_config"
        )
    device = forward_batch.input_ids.device
    dtype = torch.bfloat16  # model's native output dtype for sglang FP8 models

    fake_logits = torch.zeros(bs, vocab, dtype=dtype, device=device)
    lpo = LogitsProcessorOutput(next_token_logits=fake_logits)
    return ModelRunnerOutput(logits_output=lpo, can_run_graph=False)


def _install():
    if os.environ.get("LAYERWISE_SKIP_PREFILL", "1") != "1":
        logger.info("[skip-prefill] disabled via LAYERWISE_SKIP_PREFILL=0")
        return

    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner

    orig_forward = ModelRunner.forward

    def patched(self, forward_batch, *args, **kwargs):
        fm = forward_batch.forward_mode
        # Any extend variant: EXTEND, TARGET_VERIFY (spec), etc. We match the
        # canonical extend-without-speculative since that's what bench_one_batch
        # uses. For speculative cases the user can disable.
        if fm == ForwardMode.EXTEND:
            return _fake_extend_output(self, forward_batch)
        return orig_forward(self, forward_batch, *args, **kwargs)

    ModelRunner.forward = patched
    logger.warning(
        "[skip-prefill] installed ModelRunner.forward short-circuit for EXTEND "
        "(fake LogitsProcessorOutput; KV slots still allocated)"
    )


_install()
