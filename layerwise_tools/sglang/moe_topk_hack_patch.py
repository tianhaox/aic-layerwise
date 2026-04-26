"""Local MoE top-k distribution hacks for layerwise SGLang benchmarks.

This keeps the existing SGLANG_HACK_OVERRIDE_TOPK_IDS_RANDOM entry point but
lets benchmarks replace the random expert ids with the power-law logits pattern
used by aiconfigurator/collector/sglang/collect_moe.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from sglang.srt.layers.moe import topk as _topk_module


_ORIGINAL_OVERRIDE = _topk_module._maybe_override_topk_ids_random
_CACHE: dict[tuple, torch.Tensor] = {}
_LOGGED: set[tuple] = set()
_AIC_HELPERS = None


def _disable_torch_compile(fn):
    """Keep env-driven top-k hacks outside Dynamo graphs."""
    disable = getattr(getattr(torch, "compiler", None), "disable", None)
    return disable(fn) if disable is not None else fn


def _mode() -> str:
    return os.getenv("LAYERWISE_MOE_TOPK_HACK_MODE", "off").strip().lower()


def _alpha() -> float:
    return float(os.getenv("LAYERWISE_MOE_TOPK_POWER_LAW_ALPHA", "1.01"))


def _seed() -> int:
    return int(os.getenv("LAYERWISE_MOE_TOPK_HACK_SEED", "0"))


def _infer_ep_rank() -> int:
    try:
        from sglang.srt.distributed import get_moe_expert_parallel_rank

        return max(0, int(get_moe_expert_parallel_rank()))
    except Exception:
        return 0


def _infer_ep_size(num_experts: int) -> int:
    env_ep = os.getenv("LAYERWISE_MOE_TOPK_HACK_EP_SIZE")
    if env_ep:
        ep = max(1, int(env_ep))
    else:
        try:
            from sglang.srt.distributed import get_moe_expert_parallel_world_size

            ep = max(1, int(get_moe_expert_parallel_world_size()))
        except Exception:
            ep = 1
    return ep if num_experts % ep == 0 else 1


def _global_token_multiplier(ep: int) -> int:
    env_multiplier = os.getenv("LAYERWISE_MOE_TOPK_HACK_GLOBAL_MULTIPLIER", "auto")
    if env_multiplier.strip().lower() != "auto":
        return max(1, int(env_multiplier))
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            return ep
    except Exception:
        pass
    return 1


def _distributed_world_size() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
    except Exception:
        pass
    return 1


def _swap_max_to_rank0() -> bool:
    env_swap = os.getenv("LAYERWISE_MOE_TOPK_HACK_SWAP_MAX_TO_RANK0", "auto")
    value = env_swap.strip().lower()
    if value not in ("", "auto"):
        return value in ("1", "true", "yes", "on")
    # AIC helper semantics always swap the max-load EP rank into rank0.
    return True


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
            "LAYERWISE_MOE_TOPK_HACK_MODE=power_law/balanced requires the AIC "
            f"collector helper. Could not import helper from {collector_path}. "
            "Set LAYERWISE_AIC_COLLECTOR_PATH to the aiconfigurator/collector path."
        ) from exc
    _AIC_HELPERS = (_generate_power_law_distribution, balanced_logits)
    return _AIC_HELPERS


def _power_law_selected_experts(
    num_tokens: int,
    num_experts: int,
    topk: int,
    ep: int,
    alpha: float,
    *,
    seed: int,
    swap_max_to_rank0: bool,
) -> torch.Tensor:
    if not swap_max_to_rank0:
        raise ValueError(
            "AIC power-law topk hack requires swapping the max-load EP rank to rank0. "
            "Set LAYERWISE_MOE_TOPK_HACK_SWAP_MAX_TO_RANK0=1 or leave it as auto."
        )
    generate_power_law, _ = _load_aic_helpers()
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        _, selected = generate_power_law(num_tokens, num_experts, topk, ep, alpha)
    finally:
        torch.random.set_rng_state(rng_state)
    return selected.to(torch.int32).cpu().contiguous()


def _balanced_selected_experts(
    num_tokens: int, num_experts: int, topk: int
) -> torch.Tensor:
    _, balanced_logits = _load_aic_helpers()
    router_logits = balanced_logits(num_tokens, num_experts, topk)
    return torch.topk(router_logits, topk, dim=-1).indices.to(torch.int32).cpu().contiguous()


def _cached_ids(
    *,
    mode: str,
    num_tokens: int,
    num_experts: int,
    topk: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ep = _infer_ep_size(num_experts)
    ep_rank = _infer_ep_rank()
    global_multiplier = _global_token_multiplier(ep)
    global_tokens = num_tokens * global_multiplier
    row_start = (ep_rank % global_multiplier) * num_tokens
    alpha = _alpha()
    seed = _seed()
    swap_max = _swap_max_to_rank0()
    key = (
        mode,
        num_tokens,
        global_tokens,
        row_start,
        num_experts,
        topk,
        ep,
        ep_rank,
        global_multiplier,
        alpha,
        seed,
        swap_max,
        str(device),
        str(dtype),
    )
    cached = _CACHE.get(key)
    if cached is None:
        if mode == "power_law":
            ids = _power_law_selected_experts(
                global_tokens,
                num_experts,
                topk,
                ep,
                alpha,
                seed=seed,
                swap_max_to_rank0=swap_max,
            )
        elif mode == "balanced":
            ids = _balanced_selected_experts(global_tokens, num_experts, topk)
        else:
            raise ValueError(f"unsupported LAYERWISE_MOE_TOPK_HACK_MODE={mode!r}")
        ids = ids[row_start : row_start + num_tokens].contiguous()
        cached = ids.to(device=device, dtype=dtype, non_blocking=True).contiguous()
        _CACHE[key] = cached

    log_key = key[:-2]
    if os.getenv("LAYERWISE_MOE_TOPK_HACK_LOG", "0") == "1" and log_key not in _LOGGED:
        _LOGGED.add(log_key)
        print(
            "[moe-topk-hack] "
            f"mode={mode} tokens={num_tokens} global_tokens={global_tokens} "
            f"row_start={row_start} experts={num_experts} topk={topk} "
            f"ep={ep} ep_rank={ep_rank} alpha={alpha} seed={seed} "
            f"swap_max_to_rank0={swap_max}",
            flush=True,
        )
    return cached.clone()


@_disable_torch_compile
def _maybe_override_topk_ids(topk_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    mode = _mode()
    if mode in ("", "off", "none", "false", "0", "random"):
        return _ORIGINAL_OVERRIDE(topk_ids, num_experts)
    if topk_ids.numel() == 0:
        return topk_ids
    if mode not in ("power_law", "balanced"):
        raise ValueError(
            "LAYERWISE_MOE_TOPK_HACK_MODE must be off, random, balanced, or power_law; "
            f"got {mode!r}"
        )
    num_tokens, topk = topk_ids.shape
    return _cached_ids(
        mode=mode,
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        device=topk_ids.device,
        dtype=topk_ids.dtype,
    )


_topk_module._maybe_override_topk_ids_random = _maybe_override_topk_ids

try:
    from sglang.srt.layers.moe import deepseek_v4_topk as _dsv4_topk_module

    _dsv4_topk_module._maybe_override_topk_ids_random = _maybe_override_topk_ids
except Exception:
    pass
