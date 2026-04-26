"""Full-layer EP-local routed-experts timing patch for SGLang benchmarks.

This is not a standalone MoE microbenchmark.  When enabled, it patches the
full-layer SGLang `mlp.experts` module so the routed expert compute inside a
single-GPU layer run uses the same post-dispatch, EP-local max-rank workload as
the AIC DeepGEMM baseline.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RankWorkload:
    target_rank: int
    rank_num_tokens: int
    masked_m: torch.Tensor
    rank_selection_counts: torch.Tensor
    expected_m: int
    m_capacity: int


_AIC_HELPERS = None
_CACHE: dict[tuple, dict[str, object]] = {}
_LOGGED: set[tuple] = set()
_DISPOSE_PATCHED = False


def _enabled() -> bool:
    return os.getenv("LAYERWISE_FULL_LAYER_EP_LOCAL_EXPERTS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _alpha() -> float:
    return float(os.getenv("LAYERWISE_MOE_TOPK_POWER_LAW_ALPHA", "1.01"))


def _seed() -> int:
    return int(os.getenv("LAYERWISE_MOE_TOPK_HACK_SEED", "0"))


def _ep_size(num_experts: int) -> int:
    raw = os.getenv("LAYERWISE_MOE_TOPK_HACK_EP_SIZE", "1")
    ep = max(1, int(raw))
    return ep if num_experts % ep == 0 else 1


def _global_token_multiplier(ep: int) -> int:
    raw = os.getenv("LAYERWISE_MOE_TOPK_HACK_GLOBAL_MULTIPLIER", "auto").strip().lower()
    if raw != "auto":
        return max(1, int(raw))
    return ep


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _aic_collector_path() -> Path:
    return Path(
        os.getenv(
            "LAYERWISE_AIC_COLLECTOR_PATH",
            "/tianhao/debug/dsv4/aiconfigurator/collector",
        )
    )


def _load_aic_helpers():
    global _AIC_HELPERS
    if _AIC_HELPERS is not None:
        return _AIC_HELPERS
    collector_path = _aic_collector_path()
    path_str = str(collector_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    try:
        from helper import _generate_power_law_distribution, balanced_logits
    except Exception as exc:
        raise RuntimeError(
            "LAYERWISE_FULL_LAYER_EP_LOCAL_EXPERTS=1 requires AIC collector "
            f"helpers. Could not import helper from {collector_path}. "
            "Set LAYERWISE_AIC_COLLECTOR_PATH to the aiconfigurator/collector path."
        ) from exc
    _AIC_HELPERS = (_generate_power_law_distribution, balanced_logits)
    return _AIC_HELPERS


def _selected_experts(
    *,
    num_tokens: int,
    num_experts: int,
    topk: int,
    ep: int,
    mode: str,
    alpha: float,
    seed: int,
) -> torch.Tensor:
    generate_power_law, balanced_logits = _load_aic_helpers()
    rng_state = torch.random.get_rng_state()
    try:
        # torch.manual_seed also seeds CUDA generators, which is illegal during
        # CUDA graph capture. AIC's helpers only use CPU randomness here.
        torch.random.default_generator.manual_seed(seed)
        if mode == "power_law":
            _, selected = generate_power_law(num_tokens, num_experts, topk, ep, alpha)
        elif mode == "balanced":
            logits = balanced_logits(num_tokens, num_experts, topk)
            selected = torch.topk(logits, topk, dim=-1).indices
        else:
            raise ValueError(
                "LAYERWISE_FULL_LAYER_EP_LOCAL_EXPERTS supports only "
                f"power_law/balanced distributions, got {mode!r}"
            )
    finally:
        torch.random.set_rng_state(rng_state)
    return selected.to(torch.int64).cpu().contiguous()


def _build_rank_workload(
    *,
    num_tokens: int,
    num_experts: int,
    topk: int,
    ep: int,
    mode: str,
    alpha: float,
    seed: int,
) -> _RankWorkload:
    selected = _selected_experts(
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        ep=ep,
        mode=mode,
        alpha=alpha,
        seed=seed,
    )
    experts_per_rank = num_experts // ep
    expert_counts = torch.bincount(selected.reshape(-1), minlength=num_experts).to(
        torch.int64
    )
    rank_selection_counts = expert_counts.view(ep, experts_per_rank).sum(dim=1)
    target_rank_raw = os.getenv("LAYERWISE_FULL_LAYER_EP_LOCAL_EXPERTS_RANK", "max")
    if target_rank_raw == "max":
        target_rank = int(torch.argmax(rank_selection_counts).item())
    else:
        target_rank = int(target_rank_raw)
    rank_start = target_rank * experts_per_rank
    rank_end = rank_start + experts_per_rank
    masked_m = expert_counts[rank_start:rank_end].to(torch.int32).contiguous()
    rank_token_mask = ((selected >= rank_start) & (selected < rank_end)).any(dim=1)
    rank_num_tokens = int(rank_token_mask.sum().item())
    tokens_per_rank = (num_tokens + ep - 1) // ep
    expected_m = (tokens_per_rank * ep * topk + num_experts) // num_experts
    max_masked_m = int(masked_m.max().item()) if masked_m.numel() else 0
    env_capacity = int(
        os.getenv("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "128")
    )
    m_capacity = max(env_capacity, _round_up(max_masked_m, 128), 128)
    return _RankWorkload(
        target_rank=target_rank,
        rank_num_tokens=rank_num_tokens,
        masked_m=masked_m,
        rank_selection_counts=rank_selection_counts,
        expected_m=expected_m,
        m_capacity=m_capacity,
    )


def _make_quant_info(layer, local_start: int, local_end: int):
    from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo

    quant_method = layer.quant_method
    if not getattr(quant_method, "block_quant", False):
        raise RuntimeError(
            "LAYERWISE_FULL_LAYER_EP_LOCAL_EXPERTS currently supports FP8 block "
            "quantized DeepGEMM MoE only."
        )
    return DeepGemmMoeQuantInfo(
        w13_weight=layer.w13_weight[local_start:local_end].contiguous(),
        w2_weight=layer.w2_weight[local_start:local_end].contiguous(),
        use_fp8=True,
        w13_scale=layer.w13_weight_scale_inv[local_start:local_end].contiguous(),
        w2_scale=layer.w2_weight_scale_inv[local_start:local_end].contiguous(),
        block_shape=quant_method.quant_config.weight_block_size,
    )


def _get_cached_run_state(layer, hidden_states: torch.Tensor, topk: int):
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.deep_gemm import (
        DeepGemmRunnerCore,
        DeepGemmRunnerInput,
    )

    num_experts = int(layer.num_experts - layer.num_fused_shared_experts)
    ep = _ep_size(num_experts)
    experts_per_rank = num_experts // ep
    mode = os.getenv("LAYERWISE_MOE_TOPK_HACK_MODE", "power_law").strip().lower()
    alpha = _alpha()
    seed = _seed()
    global_tokens = int(hidden_states.shape[0]) * _global_token_multiplier(ep)
    workload = _build_rank_workload(
        num_tokens=global_tokens,
        num_experts=num_experts,
        topk=topk,
        ep=ep,
        mode=mode,
        alpha=alpha,
        seed=seed,
    )
    local_start = workload.target_rank * experts_per_rank
    local_end = local_start + experts_per_rank
    block_shape = layer.quant_method.quant_config.weight_block_size
    hidden_size = int(layer.hidden_size)
    scale_k = (hidden_size + block_shape[1] - 1) // block_shape[1]
    key = (
        int(layer.layer_id),
        str(hidden_states.device),
        str(hidden_states.dtype),
        int(hidden_states.shape[0]),
        hidden_size,
        int(layer.intermediate_size_per_partition),
        topk,
        num_experts,
        ep,
        mode,
        alpha,
        seed,
        workload.target_rank,
        workload.m_capacity,
        tuple(int(x) for x in workload.masked_m.tolist()),
    )
    cached = _CACHE.get(key)
    if cached is None:
        device = hidden_states.device
        local_experts = experts_per_rank
        runner_config = MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=local_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=int(layer.intermediate_size_per_partition),
            layer_id=int(layer.layer_id),
            top_k=topk,
            num_fused_shared_experts=0,
            params_dtype=torch.bfloat16,
            activation=layer.moe_runner_config.activation,
            is_gated=layer.moe_runner_config.is_gated,
            swiglu_limit=layer.moe_runner_config.swiglu_limit,
        )
        hidden_states_ep = torch.empty(
            (local_experts, workload.m_capacity, hidden_size),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        hidden_states_scale = torch.empty(
            (local_experts, workload.m_capacity, scale_k),
            device=device,
            dtype=torch.float32,
        )
        cached = {
            "runner": DeepGemmRunnerCore(runner_config),
            "runner_input": DeepGemmRunnerInput(
                hidden_states=hidden_states_ep,
                hidden_states_scale=hidden_states_scale,
                use_masked_gemm=True,
                masked_m=workload.masked_m.to(device=device, dtype=torch.int32),
                expected_m=workload.expected_m,
            ),
            "quant_info": _make_quant_info(layer, local_start, local_end),
            "running_state": {"hidden_states_device": device},
            "workload": workload,
            "global_tokens": global_tokens,
        }
        _CACHE[key] = cached

    if os.getenv("LAYERWISE_FULL_LAYER_EP_LOCAL_EXPERTS_LOG", "0") == "1" and key not in _LOGGED:
        _LOGGED.add(key)
        wl = cached["workload"]
        assert isinstance(wl, _RankWorkload)
        print(
            "[full-layer-ep-local-experts] "
            f"layer={layer.layer_id} global_tokens={cached['global_tokens']} "
            f"experts={num_experts} topk={topk} ep={ep} "
            f"target_rank={wl.target_rank} rank_num_tokens={wl.rank_num_tokens} "
            f"masked_m={wl.masked_m.tolist()} "
            f"rank_selection_counts={wl.rank_selection_counts.tolist()} "
            f"expected_m={wl.expected_m} m_capacity={wl.m_capacity}",
            flush=True,
        )
    return cached


def _install() -> None:
    if not _enabled():
        logger.info("[full-layer-ep-local-experts] disabled")
        return

    global _DISPOSE_PATCHED
    if not _DISPOSE_PATCHED:
        from sglang.srt.layers.moe.moe_runner import deep_gemm as deep_gemm_mod

        # DeepGEMM disposes runner inputs after each call. This patch reuses a
        # cached synthetic EP-local input to keep the full-layer CUDA graph clean,
        # so disposal must be disabled for this env-gated profiling path.
        deep_gemm_mod.dispose_tensor = lambda _tensor: None
        _DISPOSE_PATCHED = True

    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    if getattr(FusedMoE, "_layerwise_ep_local_experts_installed", False):
        return

    orig_forward = FusedMoE.forward

    def patched_forward(self, hidden_states: torch.Tensor, topk_output):
        if hidden_states.numel() == 0:
            return orig_forward(self, hidden_states, topk_output)
        if not hasattr(self.quant_method, "runner") or not self.quant_method.runner.runner_backend.is_deep_gemm():
            return orig_forward(self, hidden_states, topk_output)

        topk = int(getattr(self, "top_k", 0) or topk_output.topk_ids.shape[-1])
        cached = _get_cached_run_state(self, hidden_states, topk)
        runner = cached["runner"]
        runner_input = cached["runner_input"]
        quant_info = cached["quant_info"]
        running_state = cached["running_state"]
        out = runner.run(runner_input, quant_info, running_state).hidden_states
        del out
        return torch.empty_like(hidden_states)

    FusedMoE.forward = patched_forward
    FusedMoE._layerwise_ep_local_experts_installed = True
    logger.warning("[full-layer-ep-local-experts] installed FusedMoE.forward patch")


_install()
