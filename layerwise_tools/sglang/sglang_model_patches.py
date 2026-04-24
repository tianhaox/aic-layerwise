"""Install model-behavior patches for SGLang layerwise profiling.

This module intentionally groups the small model-execution monkey patches used
by ``run_bench_skip.py``.  Profiling-only instrumentation, such as step NVTX
markers and MoE top-k distribution hacks, stays in separate modules.
"""

from __future__ import annotations

import logging
import os
import types

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer skip
# ---------------------------------------------------------------------------


def _parse_target_layers() -> set[int]:
    raw = os.environ.get("LAYERWISE_TARGET_LAYERS", "3")
    return {int(x) for x in raw.split(",") if x.strip()}


_RETURN_ARITY = {
    "DeepseekV2DecoderLayer": 3,
    "DeepseekV3DecoderLayer": 3,
    "Glm4MoeDecoderLayer": 3,
    "Qwen3_5AttentionDecoderLayer": 2,
    "Qwen3_5LinearDecoderLayer": 2,
    "Qwen3DecoderLayer": 2,
    "Qwen3MoeDecoderLayer": 2,
    "Qwen2MoeDecoderLayer": 2,
    "LlamaDecoderLayer": 2,
    "MixtralDecoderLayer": 2,
    "DeepseekV4DecoderLayer": 1,
}


def _make_identity_forward(arity: int):
    def _identity_forward(self, *args, **kwargs):
        hidden_states = kwargs.get("hidden_states")
        residual = kwargs.get("residual")
        prev_topk = kwargs.get("prev_topk_indices")

        if hidden_states is None:
            for arg in args:
                if hasattr(arg, "shape") and arg.dim() >= 2:
                    hidden_states = arg
                    break

        if arity == 3:
            return hidden_states, residual, prev_topk
        if arity == 1:
            return hidden_states
        return hidden_states, residual

    return _identity_forward


def _identity_for_layer(layer):
    return _make_identity_forward(_RETURN_ARITY.get(type(layer).__name__, 2))


def _apply_layer_skip(model, target_layers: set[int]) -> int:
    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    if layers is None:
        logger.warning("[layer-skip] cannot find model.model.layers; no-op")
        return 0

    skipped = 0
    kept = []
    unknown_classes = set()
    for idx, layer in enumerate(layers):
        if idx in target_layers:
            kept.append(idx)
            continue
        cls = type(layer).__name__
        if cls not in _RETURN_ARITY:
            unknown_classes.add(cls)
        layer.forward = types.MethodType(_identity_for_layer(layer), layer)
        skipped += 1

    if unknown_classes:
        logger.warning(
            "[layer-skip] unknown DecoderLayer class(es) %s, defaulting to arity=2. "
            "Add to _RETURN_ARITY if incorrect.",
            unknown_classes,
        )
    logger.warning(
        "[layer-skip] kept=%s, skipped=%s layers (identity forward); total layers=%s",
        sorted(kept),
        skipped,
        len(layers),
    )
    return skipped


def install_layer_skip_patch() -> None:
    if os.environ.get("LAYERWISE_SKIP_ENABLE", "1") != "1":
        logger.info("[layer-skip] disabled by LAYERWISE_SKIP_ENABLE=0")
        return

    from sglang.srt.model_executor import model_runner as model_runner_mod

    if getattr(model_runner_mod.ModelRunner, "_layerwise_layer_skip_installed", False):
        return

    orig = model_runner_mod.ModelRunner.load_model
    targets = _parse_target_layers()

    def patched(self, *args, **kwargs):
        ret = orig(self, *args, **kwargs)
        try:
            if self.model is not None:
                _apply_layer_skip(self.model, targets)
        except Exception as exc:
            logger.error("[layer-skip] failed: %s", exc)
            raise
        return ret

    model_runner_mod.ModelRunner.load_model = patched
    model_runner_mod.ModelRunner._layerwise_layer_skip_installed = True
    logger.warning(
        "[layer-skip] installed ModelRunner.load_model patch, targets=%s",
        sorted(targets),
    )


# ---------------------------------------------------------------------------
# Skip prefill
# ---------------------------------------------------------------------------


def _fake_extend_output(model_runner, forward_batch):
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.model_runner import ModelRunnerOutput

    batch_size = forward_batch.batch_size
    hf_config = model_runner.model_config.hf_config
    vocab_size = getattr(hf_config, "vocab_size", None) or getattr(
        getattr(hf_config, "text_config", None), "vocab_size", None
    )
    if vocab_size is None:
        raise AttributeError(
            "Could not resolve vocab_size on hf_config or hf_config.text_config"
        )

    fake_logits = torch.zeros(
        batch_size,
        vocab_size,
        dtype=torch.bfloat16,
        device=forward_batch.input_ids.device,
    )
    logits_output = LogitsProcessorOutput(next_token_logits=fake_logits)
    return ModelRunnerOutput(logits_output=logits_output, can_run_graph=False)


def install_skip_prefill_patch() -> None:
    if os.environ.get("LAYERWISE_SKIP_PREFILL", "1") != "1":
        logger.info("[skip-prefill] disabled via LAYERWISE_SKIP_PREFILL=0")
        return

    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner

    if getattr(ModelRunner, "_layerwise_skip_prefill_installed", False):
        return

    orig_forward = ModelRunner.forward

    def patched(self, forward_batch, *args, **kwargs):
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            return _fake_extend_output(self, forward_batch)
        return orig_forward(self, forward_batch, *args, **kwargs)

    ModelRunner.forward = patched
    ModelRunner._layerwise_skip_prefill_installed = True
    logger.warning(
        "[skip-prefill] installed ModelRunner.forward short-circuit for EXTEND "
        "(fake LogitsProcessorOutput; KV slots still allocated)"
    )


# ---------------------------------------------------------------------------
# DeepSeek-V4 cache fix
# ---------------------------------------------------------------------------


def install_deepseekv4_cache_patch() -> None:
    from sglang.srt.mem_cache.deepseekv4_memory_pool import DeepSeekV4TokenToKVPool

    if getattr(
        DeepSeekV4TokenToKVPool,
        "_layerwise_lazy_cached_loc_installed",
        False,
    ):
        return

    orig = DeepSeekV4TokenToKVPool.set_swa_key_buffer_radix_fused

    def patched(self, layer_id, raw_loc, cache_k):
        if getattr(self, "_should_cache_swa", False) and not hasattr(self, "cached_loc"):
            self.cached_loc = self.translate_loc_from_full_to_swa(raw_loc)
        return orig(self, layer_id, raw_loc, cache_k)

    DeepSeekV4TokenToKVPool.set_swa_key_buffer_radix_fused = patched
    DeepSeekV4TokenToKVPool._layerwise_lazy_cached_loc_installed = True
    logger.warning("[deepseekv4-cache] installed lazy cached_loc patch")


# ---------------------------------------------------------------------------
# FlashMLA local-TP head padding
# ---------------------------------------------------------------------------


def _pad_heads(x: torch.Tensor, target_hq: int, fill_value: float = 0.0):
    if x is None:
        return None
    orig_hq = x.shape[-1] if x.dim() == 1 else x.shape[2]
    if orig_hq >= target_hq:
        return x
    if x.dim() == 1:
        out = x.new_full((target_hq,), fill_value)
        out[:orig_hq] = x
        return out
    out = x.new_zeros((*x.shape[:2], target_hq, *x.shape[3:]))
    out[:, :, :orig_hq, ...] = x
    return out


def _slice_heads(x: torch.Tensor, orig_hq: int):
    if x is None:
        return None
    if x.dim() >= 4:
        return x[:, :, :orig_hq, ...]
    if x.dim() == 3:
        return x[:, :, :orig_hq]
    return x


def install_flashmla_pad_patch() -> None:
    if os.environ.get("LAYERWISE_FLASHMLA_PAD_HQ", "0") != "1":
        return

    from sglang.srt.layers.attention import debug_flash_mla_adapter as adapter

    if getattr(adapter, "_layerwise_flashmla_pad_installed", False):
        return

    orig = adapter.flash_mla_with_kvcache_entrypoint

    def patched_flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):
        q = kwargs.get("q")
        if q is None or q.dim() != 4:
            return orig(backend=backend, **kwargs)

        orig_hq = q.shape[2]
        if orig_hq in (64, 128):
            return orig(backend=backend, **kwargs)
        if orig_hq > 64:
            raise RuntimeError(
                f"LAYERWISE_FLASHMLA_PAD_HQ only supports h_q <= 64, got {orig_hq}"
            )

        kwargs["q"] = _pad_heads(q, 64)
        if kwargs.get("attn_sink") is not None:
            kwargs["attn_sink"] = _pad_heads(
                kwargs["attn_sink"], 64, fill_value=float("inf")
            )

        out, lse = orig(backend=backend, **kwargs)
        return _slice_heads(out, orig_hq), _slice_heads(lse, orig_hq)

    adapter.flash_mla_with_kvcache_entrypoint = patched_flash_mla_with_kvcache_entrypoint
    adapter._layerwise_flashmla_pad_installed = True
    logger.warning(
        "[flashmla-pad] installed h_q padding patch; FlashMLA attention timing "
        "will reflect padded heads"
    )


def install_all() -> None:
    install_layer_skip_patch()
    install_skip_prefill_patch()
    install_flashmla_pad_patch()
    install_deepseekv4_cache_patch()


install_all()
