"""Identity-replace non-target DecoderLayer forwards.

Import BEFORE `sglang.bench_one_batch` runs. Monkey-patches
`ModelRunner.load_model` so that after the model is loaded, each
`model.model.layers[i]` whose index is not in `LAYERWISE_TARGET_LAYERS`
gets its forward replaced with an identity pass-through.

Effect:
  - Full sglang serving stack boots normally
  - All layer weights stay allocated (we don't touch them)
  - Only target layers actually compute — other layers are pure
    `hidden_states, residual = hidden_states, residual` no-ops
  - bench_one_batch goes through full-model forward, but 77 out of 78
    layers are ~free

Why not use `--json-model-override-args '{"num_hidden_layers": N}'`:
  - That distorts `max_total_num_tokens` (divides rest_mem by num_layers)
  - Changes memory budgets, DeepGEMM workspace assumptions
  - Shifts the `first_k_dense_replace` boundary
  - Our benchmark shape would no longer reflect real-model memory state

Skip-forward keeps the full 78-layer model intact; only the compute is
short-circuited on non-target layers.

Env var:
  LAYERWISE_TARGET_LAYERS=3,4      # comma-separated layer indices (default 3)
  LAYERWISE_SKIP_ENABLE=1          # set to 0 to disable patching
"""
import logging
import os
import types

logger = logging.getLogger(__name__)


def _parse_targets() -> set[int]:
    raw = os.environ.get("LAYERWISE_TARGET_LAYERS", "3")
    return {int(x) for x in raw.split(",") if x.strip()}


# DecoderLayer return arity per model architecture. Ints = tuple size.
# Add new arches as needed. Fallback for unknown = 2 (most HF models).
_RETURN_ARITY = {
    # DeepSeek / GLM family: (hidden, residual, topk_indices)
    "DeepseekV2DecoderLayer": 3,
    "DeepseekV3DecoderLayer": 3,
    "Glm4MoeDecoderLayer": 3,
    # Qwen3.5 family: (hidden, residual)
    "Qwen3_5AttentionDecoderLayer": 2,
    "Qwen3_5LinearDecoderLayer": 2,
    # Qwen3 / Qwen2 / generic: (hidden, residual)
    "Qwen3DecoderLayer": 2,
    "Qwen3MoeDecoderLayer": 2,
    "Qwen2MoeDecoderLayer": 2,
    "LlamaDecoderLayer": 2,
    "MixtralDecoderLayer": 2,
}


def _make_identity_forward(arity: int):
    """Build an identity forward that returns the right-sized tuple."""
    def _identity_forward(self, *args, **kwargs):
        hidden_states = kwargs.get("hidden_states")
        residual = kwargs.get("residual")
        prev_topk = kwargs.get("prev_topk_indices")

        # Fallback for positional. DeepseekV2 signature:
        #   (positions, hidden_states, forward_batch, residual, ...)
        # Qwen3.5 Attention signature:
        #   (positions, hidden_states, residual, forward_batch, **kwargs)
        # Qwen3.5 Linear signature:
        #   (hidden_states, residual, **kwargs)
        # We try the most common: kwargs first. Positional rarely hit in
        # practice because Model.forward calls with keywords.
        if hidden_states is None:
            # Scan args for first Tensor with >=2 dims (hidden_states is 2D+).
            for a in args:
                if hasattr(a, "shape") and a.dim() >= 2:
                    hidden_states = a
                    break

        if arity == 3:
            return hidden_states, residual, prev_topk
        return hidden_states, residual  # arity == 2
    return _identity_forward


def _identity_for_layer(layer):
    """Pick an identity forward matching `type(layer).__name__`'s arity."""
    arity = _RETURN_ARITY.get(type(layer).__name__, 2)
    return _make_identity_forward(arity)


def _apply_skip(model, target_layers: set[int]) -> int:
    """Replace non-target layer forwards with identity. Returns number skipped."""
    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    if layers is None:
        logger.warning("[layer-skip] cannot find model.model.layers; no-op")
        return 0
    skipped = 0
    kept = []
    unknown_classes = set()
    for i, layer in enumerate(layers):
        if i in target_layers:
            kept.append(i)
        else:
            cls = type(layer).__name__
            if cls not in _RETURN_ARITY:
                unknown_classes.add(cls)
            layer.forward = types.MethodType(_identity_for_layer(layer), layer)
            skipped += 1
    if unknown_classes:
        logger.warning(
            f"[layer-skip] unknown DecoderLayer class(es) {unknown_classes}, "
            f"defaulting to arity=2. Add to _RETURN_ARITY if incorrect."
        )
    logger.warning(
        f"[layer-skip] kept={sorted(kept)}, skipped={skipped} layers "
        f"(identity forward); total layers={len(layers)}"
    )
    return skipped


def _install_patch():
    if os.environ.get("LAYERWISE_SKIP_ENABLE", "1") != "1":
        logger.info("[layer-skip] disabled by LAYERWISE_SKIP_ENABLE=0")
        return

    from sglang.srt.model_executor import model_runner as _mr

    orig = _mr.ModelRunner.load_model
    targets = _parse_targets()

    def patched(self, *args, **kwargs):
        ret = orig(self, *args, **kwargs)
        try:
            if self.model is not None:
                _apply_skip(self.model, targets)
        except Exception as e:
            logger.error(f"[layer-skip] failed: {e}")
            raise
        return ret

    _mr.ModelRunner.load_model = patched
    logger.warning(
        f"[layer-skip] installed ModelRunner.load_model patch, "
        f"targets={sorted(targets)}"
    )


_install_patch()
