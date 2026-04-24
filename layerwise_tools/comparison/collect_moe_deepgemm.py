# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SGLang DeepGEMM masked-MoE collector.

This collector is intentionally separate from collect_moe.py.  collect_moe.py
calls the fused Triton MoE function directly; this file constructs the
post-DeepEP-dispatch DeepGEMM runner input and measures only routed expert
compute.
"""

from __future__ import annotations

import itertools
import math
import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch

# Keep this collector generic.  The local SGLang fork defaults DSV4 submode to
# 2604B, which forces DSV4-specific SwiGLU clamp inside DeepGEMM.  Non-DSV4
# models such as GLM-5 should not use that unless the caller explicitly sets it.
os.environ.setdefault("SGLANG_DSV4_2604_SUBMODE", "")
os.environ.setdefault("SGLANG_DSV4_FP4_EXPERTS", "0")
os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")

try:
    from common_test_cases import get_common_moe_test_cases
    from helper import (
        _generate_power_law_distribution,
        balanced_logits,
        benchmark_with_power,
        get_sm_version,
        log_perf,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common_test_cases import get_common_moe_test_cases
    from helper import (
        _generate_power_law_distribution,
        balanced_logits,
        benchmark_with_power,
        get_sm_version,
        log_perf,
    )

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner import deep_gemm as deep_gemm_mod
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerCore,
    DeepGemmRunnerInput,
)


@dataclass(frozen=True)
class RankWorkload:
    selected_experts: torch.Tensor
    expert_counts: torch.Tensor
    rank_selection_counts: torch.Tensor
    rank_token_counts: torch.Tensor
    target_rank: int
    local_topk_ids: torch.Tensor
    local_topk_weights: torch.Tensor
    masked_m: torch.Tensor
    rank_num_tokens: int
    expected_m: int
    local_average_expected_m: int
    m_capacity: int
    tokens_per_rank: int


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def round_up(value: int, multiple: int) -> int:
    return cdiv(value, multiple) * multiple


def _env_csv_ints(name: str) -> set[int] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def _env_csv_strings(name: str) -> set[str] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def _get_sglang_version() -> str:
    try:
        return version("sglang")
    except PackageNotFoundError:
        return "unknown"


def get_moe_deepgemm_test_cases() -> list[list[Any]]:
    """Return AIC-style MoE test cases for DeepGEMM routed expert compute."""
    if get_sm_version() < 90:
        return []

    token_filter = _env_csv_ints("COLLECTOR_DEEPGEMM_MOE_NUM_TOKENS")
    tp_filter = _env_csv_ints("COLLECTOR_DEEPGEMM_MOE_TP")
    ep_filter = _env_csv_ints("COLLECTOR_DEEPGEMM_MOE_EP")
    distribution_filter = _env_csv_strings("COLLECTOR_DEEPGEMM_MOE_DISTRIBUTION")
    max_cases = os.environ.get("COLLECTOR_DEEPGEMM_MOE_MAX_CASES", "").strip()
    max_cases_value = int(max_cases) if max_cases else None

    test_cases: list[list[Any]] = []
    for common in get_common_moe_test_cases():
        if common.hidden_size % 128 != 0:
            continue
        if (common.inter_size // common.tp) % 128 != 0:
            continue
        if tp_filter is not None and common.tp not in tp_filter:
            continue
        if ep_filter is not None and common.ep not in ep_filter:
            continue
        if distribution_filter is not None and common.token_expert_distribution not in distribution_filter:
            continue

        num_tokens_list = [
            num_tokens
            for num_tokens in common.num_tokens_list
            if num_tokens <= 20480 and (token_filter is None or num_tokens in token_filter)
        ]
        for num_tokens in num_tokens_list:
            test_cases.append(
                [
                    "fp8_block",
                    num_tokens,
                    common.hidden_size,
                    common.inter_size,
                    common.topk,
                    common.num_experts,
                    common.tp,
                    common.ep,
                    common.model_name,
                    common.token_expert_distribution,
                    common.power_law_alpha or 0.0,
                ]
            )
            if max_cases_value is not None and len(test_cases) >= max_cases_value:
                return test_cases

    return test_cases


def _balanced_selected_experts(num_tokens: int, num_experts: int, topk: int) -> torch.Tensor:
    router_logits = balanced_logits(num_tokens, num_experts, topk)
    return torch.topk(router_logits, topk, dim=-1).indices.to(torch.int64).cpu().contiguous()


def _selected_experts(
    num_tokens: int,
    num_experts: int,
    topk: int,
    ep_size: int,
    distributed: str,
    power_law_alpha: float,
) -> torch.Tensor:
    if distributed == "power_law":
        _, selected = _generate_power_law_distribution(
            num_tokens,
            num_experts,
            topk,
            ep_size,
            power_law_alpha,
        )
        return selected.to(torch.int64).cpu().contiguous()
    if distributed == "balanced":
        return _balanced_selected_experts(num_tokens, num_experts, topk)
    raise ValueError(f"Unsupported distribution: {distributed}")


def build_rank_workload(
    *,
    num_tokens: int,
    num_experts: int,
    topk: int,
    ep_size: int,
    distributed: str,
    power_law_alpha: float,
    target_rank: str | int = "max",
    m_capacity: int | None = None,
) -> RankWorkload:
    if num_experts % ep_size != 0:
        raise ValueError("num_experts must be divisible by ep_size")

    selected = _selected_experts(
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        ep_size=ep_size,
        distributed=distributed,
        power_law_alpha=power_law_alpha,
    )
    experts_per_rank = num_experts // ep_size
    flat = selected.reshape(-1)
    expert_counts = torch.bincount(flat, minlength=num_experts).to(torch.int64)
    rank_selection_counts = expert_counts.view(ep_size, experts_per_rank).sum(dim=1)
    rank_token_counts = torch.stack(
        [
            ((selected >= rank * experts_per_rank) & (selected < (rank + 1) * experts_per_rank)).any(dim=1).sum()
            for rank in range(ep_size)
        ]
    ).to(torch.int64)

    if target_rank == "max":
        rank = int(torch.argmax(rank_selection_counts).item())
    else:
        rank = int(target_rank)
        if rank < 0 or rank >= ep_size:
            raise ValueError(f"target_rank={rank} is outside [0, {ep_size})")

    rank_start = rank * experts_per_rank
    rank_end = rank_start + experts_per_rank
    local_mask_all = (selected >= rank_start) & (selected < rank_end)
    rank_token_mask = local_mask_all.any(dim=1)
    rank_selected = selected[rank_token_mask]
    local_mask = (rank_selected >= rank_start) & (rank_selected < rank_end)

    local_topk_ids = (rank_selected - rank_start).to(torch.int32)
    local_topk_ids[~local_mask] = -1
    local_topk_weights = local_mask.to(torch.float32)
    masked_m = expert_counts[rank_start:rank_end].to(torch.int32).contiguous()
    rank_num_tokens = int(rank_token_mask.sum().item())
    tokens_per_rank = cdiv(num_tokens, ep_size)

    # Matches token_dispatcher/deepep.py:_DeepEPDispatcherImplLowLatency.dispatch_a.
    expected_m = (tokens_per_rank * ep_size * topk + num_experts) // num_experts
    local_average_expected_m = (local_topk_ids.numel() - 1) // experts_per_rank + 1

    env_capacity = int(os.environ.get("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "128"))
    requested_capacity = m_capacity if m_capacity is not None else env_capacity
    max_masked_m = int(masked_m.max().item()) if masked_m.numel() else 0
    capacity = max(requested_capacity, round_up(max_masked_m, 128), 128)

    return RankWorkload(
        selected_experts=selected,
        expert_counts=expert_counts,
        rank_selection_counts=rank_selection_counts,
        rank_token_counts=rank_token_counts,
        target_rank=rank,
        local_topk_ids=local_topk_ids.contiguous(),
        local_topk_weights=local_topk_weights.contiguous(),
        masked_m=masked_m,
        rank_num_tokens=rank_num_tokens,
        expected_m=expected_m,
        local_average_expected_m=local_average_expected_m,
        m_capacity=capacity,
        tokens_per_rank=tokens_per_rank,
    )


def _fp8_randn(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.randn(shape, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)


def _make_deepgemm_op(
    *,
    workload: RankWorkload,
    hidden_size: int,
    shard_intermediate_size: int,
    topk: int,
    num_experts: int,
    ep_size: int,
    device: torch.device,
) -> tuple[DeepGemmRunnerCore, DeepGemmRunnerInput, DeepGemmMoeQuantInfo, dict[str, Any]]:
    local_experts = num_experts // ep_size
    gateup_size = 2 * shard_intermediate_size
    block_shape = [128, 128]
    scale_k = hidden_size // block_shape[1]

    hidden_states = _fp8_randn((local_experts, workload.m_capacity, hidden_size), device)
    hidden_states_scale = torch.rand(
        (local_experts, workload.m_capacity, scale_k),
        device=device,
        dtype=torch.float32,
    )
    w13_weight = _fp8_randn((local_experts, gateup_size, hidden_size), device)
    w2_weight = _fp8_randn((local_experts, hidden_size, shard_intermediate_size), device)
    w13_scale = torch.rand(
        (local_experts, cdiv(gateup_size, block_shape[0]), scale_k),
        device=device,
        dtype=torch.float32,
    )
    w2_scale = torch.rand(
        (local_experts, cdiv(hidden_size, block_shape[0]), cdiv(shard_intermediate_size, block_shape[1])),
        device=device,
        dtype=torch.float32,
    )

    swiglu_limit = 10 if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B" else None
    runner_config = MoeRunnerConfig(
        num_experts=num_experts,
        num_local_experts=local_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=shard_intermediate_size,
        top_k=topk,
        params_dtype=torch.bfloat16,
        swiglu_limit=swiglu_limit,
    )
    runner = DeepGemmRunnerCore(runner_config)
    runner_input = DeepGemmRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        use_masked_gemm=True,
        masked_m=workload.masked_m.to(device=device, dtype=torch.int32),
        expected_m=workload.expected_m,
    )
    quant_info = DeepGemmMoeQuantInfo(
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        use_fp8=True,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
    )
    running_state = {"hidden_states_device": device}
    return runner, runner_input, quant_info, running_state


def run_moe_deepgemm(
    moe_type: str,
    num_tokens: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    num_experts: int,
    moe_tp_size: int,
    moe_ep_size: int,
    model_name: str,
    distributed: str = "power_law",
    power_law_alpha: float = 0.0,
    *,
    perf_filename,
    device: str = "cuda:0",
):
    assert moe_type == "fp8_block", "collect_moe_deepgemm currently supports fp8_block only"
    assert inter_size % moe_tp_size == 0, "inter_size must be divisible by moe_tp_size"
    assert num_experts % moe_ep_size == 0, "num_experts must be divisible by moe_ep_size"
    assert hidden_size % 128 == 0 and (inter_size // moe_tp_size) % 128 == 0

    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)

    # Production disposes DeepEP dispatch-owned tensors after one run.  The
    # collector reuses synthetic dispatch outputs across warmups/runs.
    deep_gemm_mod.dispose_tensor = lambda _tensor: None

    device_obj = torch.device(device)
    # Keep power-law routing on CPU RNG before allocating CUDA tensors.  This
    # matches the layerwise mock/top-k hack and avoids default-device-dependent
    # routing changes.
    workload = build_rank_workload(
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        ep_size=moe_ep_size,
        distributed=distributed,
        power_law_alpha=power_law_alpha,
        target_rank=os.environ.get("COLLECTOR_DEEPGEMM_MOE_RANK", "max"),
    )
    runner, runner_input, quant_info, running_state = _make_deepgemm_op(
        workload=workload,
        hidden_size=hidden_size,
        shard_intermediate_size=inter_size // moe_tp_size,
        topk=topk,
        num_experts=num_experts,
        ep_size=moe_ep_size,
        device=device_obj,
    )

    def run_op():
        out = runner.run(runner_input, quant_info, running_state).hidden_states
        del out

    outside_loop_count = int(os.environ.get("COLLECTOR_DEEPGEMM_MOE_OUTSIDE_LOOP", "1"))

    def kernel_func():
        for _ in range(outside_loop_count):
            run_op()

    with benchmark_with_power(
        device=device_obj,
        kernel_func=kernel_func,
        num_warmups=5,
        num_runs=int(os.environ.get("COLLECTOR_DEEPGEMM_MOE_NUM_RUNS", "20")),
        repeat_n=1,
        allow_graph_fail=True,
        use_cuda_graph=os.environ.get("COLLECTOR_DEEPGEMM_MOE_USE_CUDA_GRAPH", "1") != "0",
    ) as results:
        pass

    latency = results["latency_ms"] / outside_loop_count
    log_perf(
        item_list=[
            {
                "moe_dtype": moe_type,
                "num_tokens": num_tokens,
                "hidden_size": hidden_size,
                "inter_size": inter_size,
                "topk": topk,
                "num_experts": num_experts,
                "moe_tp_size": moe_tp_size,
                "moe_ep_size": moe_ep_size,
                "distribution": "power_law_" + str(power_law_alpha) if distributed == "power_law" else distributed,
                "rank": workload.target_rank,
                "rank_num_tokens": workload.rank_num_tokens,
                "masked_m": workload.masked_m.tolist(),
                "rank_selection_counts": workload.rank_selection_counts.tolist(),
                "expected_m": workload.expected_m,
                "m_capacity": workload.m_capacity,
                "latency": latency,
            }
        ],
        framework="SGLang",
        version=_get_sglang_version(),
        device_name=torch.cuda.get_device_name(device),
        op_name="moe",
        kernel_source="sglang_deepgemm_moe_masked",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    for test_case in get_moe_deepgemm_test_cases():
        print(test_case)
        run_moe_deepgemm(*test_case, perf_filename=PerfFile.MOE)
