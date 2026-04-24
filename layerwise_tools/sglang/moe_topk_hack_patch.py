"""Local MoE top-k distribution hacks for layerwise SGLang benchmarks.

This keeps the existing SGLANG_HACK_OVERRIDE_TOPK_IDS_RANDOM entry point but
lets benchmarks replace the random expert ids with the power-law logits pattern
used by aiconfigurator/collector/sglang/collect_moe.py.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch

from sglang.srt.layers.moe import topk as _topk_module


_ORIGINAL_OVERRIDE = _topk_module._maybe_override_topk_ids_random
_CACHE: dict[tuple, torch.Tensor] = {}
_LOGGED: set[tuple] = set()


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
    return _distributed_world_size() <= 1


def _sample_power_law(
    size: int,
    alpha: float,
    xmin: float,
    xmax: float,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    u = torch.rand(size, generator=generator, device="cpu")
    if abs(alpha - 1.0) < 1e-6:
        return xmin * (xmax / xmin) ** u
    return ((xmax ** (1 - alpha) - xmin ** (1 - alpha)) * u + xmin ** (1 - alpha)) ** (
        1 / (1 - alpha)
    )


def _round_robin_adjust_per_rank(
    counts_2d: torch.Tensor,
    remaining: int,
    *,
    add: bool,
    upper_bound: int,
) -> torch.Tensor:
    while remaining > 0:
        progressed = False
        for rank_idx in range(counts_2d.size(0)):
            local = counts_2d[rank_idx]
            valid = torch.nonzero(local < upper_bound if add else local > 0).flatten()
            if valid.numel() == 0:
                continue
            local_idx = valid[torch.argmin(local[valid]) if add else torch.argmax(local[valid])]
            local[local_idx] += 1 if add else -1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return counts_2d


def _assign_experts_from_counts(
    counts: torch.Tensor, num_tokens: int, topk: int
) -> torch.Tensor:
    sorted_experts = torch.argsort(counts, descending=True)
    sorted_counts = counts[sorted_experts]
    expert_ids_flat = torch.repeat_interleave(sorted_experts, sorted_counts)
    return expert_ids_flat.reshape(topk, num_tokens).t().contiguous()


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
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    if num_tokens * topk > num_experts:
        counts = _sample_power_law(
            num_experts, alpha, 1, num_tokens * 0.8, generator=generator
        )
    else:
        counts = _sample_power_law(num_experts, alpha, 0.01, 2, generator=generator)

    target_sum = num_tokens * topk
    counts = torch.round(counts / counts.sum() * target_sum).to(torch.int64)
    upper_bound = num_tokens

    overflow = int((counts - upper_bound).clamp(min=0).sum().item())
    counts = counts.clamp(max=upper_bound)

    experts_per_rank = num_experts // ep
    if overflow > 0:
        counts = _round_robin_adjust_per_rank(
            counts.view(ep, experts_per_rank),
            overflow,
            add=True,
            upper_bound=upper_bound,
        ).view(-1)

    delta = target_sum - int(counts.sum().item())
    if delta != 0:
        counts = _round_robin_adjust_per_rank(
            counts.view(ep, experts_per_rank),
            abs(delta),
            add=delta > 0,
            upper_bound=upper_bound,
        ).view(-1)

    rank_loads = counts.view(ep, experts_per_rank).sum(dim=1)
    max_rank = int(torch.argmax(rank_loads).item())
    if swap_max_to_rank0 and max_rank != 0:
        counts_2d = counts.view(ep, experts_per_rank)
        counts_2d[[0, max_rank]] = counts_2d[[max_rank, 0]]
        counts = counts_2d.view(-1)

    return _assign_experts_from_counts(counts, num_tokens, topk).to(torch.int32)


def _balanced_selected_experts(
    num_tokens: int, num_experts: int, topk: int
) -> torch.Tensor:
    stride = math.ceil(num_experts / topk)
    token_indices = torch.arange(num_tokens).unsqueeze(1)
    topk_indices = torch.arange(topk).unsqueeze(0)
    if num_tokens >= stride:
        ids = (token_indices + topk_indices * stride) % num_experts
    else:
        ids = (token_indices * stride / num_tokens + topk_indices * stride) % num_experts
    return ids.to(torch.int32)


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
