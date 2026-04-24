#!/usr/bin/env python3
"""AIC-aligned single-GPU mock for EP-rank MoE compute.

This intentionally does not run a full SGLang model.  It mirrors the AIC
collector shape construction: build a global power-law routing distribution,
pick one EP rank's local workload, then run the compute object that exists
after DeepEP dispatch and immediately before DeepGEMM masked compute.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_AIC_COLLECTOR = Path("/tianhao/debug/dsv4/aiconfigurator/collector")
DEFAULT_SGLANG_PYTHON = Path("/workspace/sglang/python")


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
    deepep_ll_expected_m: int
    local_average_expected_m: int
    m_capacity: int
    tokens_per_rank: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an AIC-aligned EP-rank DeepGEMM MoE compute mock."
    )
    parser.add_argument("--global-tokens", type=int, default=8)
    parser.add_argument("--tokens-per-rank", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rank",
        default="max",
        help="'max' for the rank with the largest selection load, or an integer rank id.",
    )
    parser.add_argument(
        "--expected-m-mode",
        choices=("capacity", "deepep-ll", "local-average"),
        default="deepep-ll",
        help=(
            "capacity is runnable with the installed DeepGEMM masked API; "
            "deepep-ll matches token_dispatcher/deepep.py:dispatch_a; "
            "local-average matches local topk preprocessing."
        ),
    )
    parser.add_argument(
        "--m-capacity",
        type=int,
        default=None,
        help="Capacity dimension for DeepGEMM input. Defaults to SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--profile-kernels", action="store_true")
    parser.add_argument("--profile-iters", type=int, default=5)
    parser.add_argument("--profile-top", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--allow-deepgemm-precompile",
        action="store_true",
        help="Keep SGLang DeepGEMM all-M precompile enabled. Disabled by default for this microbench.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    parser.add_argument("--aic-collector-path", type=Path, default=DEFAULT_AIC_COLLECTOR)
    parser.add_argument("--sglang-python-path", type=Path, default=DEFAULT_SGLANG_PYTHON)
    return parser.parse_args()


def add_import_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def load_aic_power_law(aic_collector_path: Path):
    add_import_path(aic_collector_path)
    try:
        from helper import _generate_power_law_distribution
    except Exception as exc:  # pragma: no cover - environment-dependent import path
        raise RuntimeError(
            f"Failed to import AIC helper from {aic_collector_path}. "
            "Pass --aic-collector-path if the checkout moved."
        ) from exc
    return _generate_power_law_distribution


def select_target_rank(rank_arg: str, rank_selection_counts: torch.Tensor) -> int:
    if rank_arg == "max":
        return int(torch.argmax(rank_selection_counts).item())
    try:
        rank = int(rank_arg)
    except ValueError as exc:
        raise ValueError("--rank must be 'max' or an integer") from exc
    if rank < 0 or rank >= rank_selection_counts.numel():
        raise ValueError(f"--rank={rank} is outside [0, {rank_selection_counts.numel()})")
    return rank


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def build_rank_workload(args: argparse.Namespace) -> RankWorkload:
    if args.num_experts % args.ep_size != 0:
        raise ValueError("--num-experts must be divisible by --ep-size")
    if args.global_tokens <= 0:
        raise ValueError("--global-tokens must be positive")

    torch.manual_seed(args.seed)
    generate_power_law = load_aic_power_law(args.aic_collector_path)
    _, selected_experts = generate_power_law(
        args.global_tokens,
        args.num_experts,
        args.topk,
        args.ep_size,
        args.alpha,
    )
    selected_experts = selected_experts.to(torch.int64).cpu().contiguous()

    experts_per_rank = args.num_experts // args.ep_size
    flat_experts = selected_experts.reshape(-1)
    expert_counts = torch.bincount(flat_experts, minlength=args.num_experts).to(torch.int64)
    rank_selection_counts = expert_counts.view(args.ep_size, experts_per_rank).sum(dim=1)
    rank_token_counts = torch.stack(
        [
            ((selected_experts >= rank * experts_per_rank)
             & (selected_experts < (rank + 1) * experts_per_rank)).any(dim=1).sum()
            for rank in range(args.ep_size)
        ]
    ).to(torch.int64)

    target_rank = select_target_rank(args.rank, rank_selection_counts)
    rank_start = target_rank * experts_per_rank
    rank_end = rank_start + experts_per_rank
    local_mask_all = (selected_experts >= rank_start) & (selected_experts < rank_end)
    rank_token_mask = local_mask_all.any(dim=1)
    rank_selected = selected_experts[rank_token_mask]
    local_mask = (rank_selected >= rank_start) & (rank_selected < rank_end)

    local_topk_ids = (rank_selected - rank_start).to(torch.int32)
    local_topk_ids[~local_mask] = -1
    local_topk_weights = local_mask.to(torch.float32)

    masked_m = expert_counts[rank_start:rank_end].to(torch.int32).contiguous()
    rank_num_tokens = int(rank_token_mask.sum().item())
    tokens_per_rank = (
        args.tokens_per_rank
        if args.tokens_per_rank is not None
        else math.ceil(args.global_tokens / args.ep_size)
    )
    # Match token_dispatcher/deepep.py: dispatch_a.
    deepep_ll_expected_m = (
        tokens_per_rank * args.ep_size * args.topk + args.num_experts
    ) // args.num_experts
    local_average_expected_m = (local_topk_ids.numel() - 1) // experts_per_rank + 1

    env_capacity = int(os.environ.get("SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "128"))
    requested_capacity = args.m_capacity if args.m_capacity is not None else env_capacity
    max_masked_m = int(masked_m.max().item()) if masked_m.numel() else 0
    m_capacity = max(requested_capacity, round_up(max_masked_m, 128), 128)
    if args.expected_m_mode == "capacity":
        expected_m = m_capacity
    elif args.expected_m_mode == "deepep-ll":
        expected_m = deepep_ll_expected_m
    else:
        expected_m = local_average_expected_m

    return RankWorkload(
        selected_experts=selected_experts,
        expert_counts=expert_counts,
        rank_selection_counts=rank_selection_counts,
        rank_token_counts=rank_token_counts,
        target_rank=target_rank,
        local_topk_ids=local_topk_ids.contiguous(),
        local_topk_weights=local_topk_weights.contiguous(),
        masked_m=masked_m,
        rank_num_tokens=rank_num_tokens,
        expected_m=expected_m,
        deepep_ll_expected_m=deepep_ll_expected_m,
        local_average_expected_m=local_average_expected_m,
        m_capacity=m_capacity,
        tokens_per_rank=tokens_per_rank,
    )


def workload_summary(args: argparse.Namespace, workload: RankWorkload) -> dict[str, Any]:
    experts_per_rank = args.num_experts // args.ep_size
    return {
        "backend": "deep_gemm_masked_core",
        "distribution_source": "aiconfigurator.collector.helper._generate_power_law_distribution",
        "global_tokens": args.global_tokens,
        "tokens_per_rank": workload.tokens_per_rank,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_experts": args.num_experts,
        "ep_size": args.ep_size,
        "experts_per_rank": experts_per_rank,
        "topk": args.topk,
        "alpha": args.alpha,
        "seed": args.seed,
        "target_rank": workload.target_rank,
        "rank_num_tokens": workload.rank_num_tokens,
        "expected_m": workload.expected_m,
        "expected_m_mode": args.expected_m_mode,
        "deepep_ll_expected_m": workload.deepep_ll_expected_m,
        "local_average_expected_m": workload.local_average_expected_m,
        "m_capacity": workload.m_capacity,
        "expert_counts": workload.expert_counts.tolist(),
        "rank_selection_counts": workload.rank_selection_counts.tolist(),
        "rank_token_counts": workload.rank_token_counts.tolist(),
        "masked_m": workload.masked_m.tolist(),
        "local_topk_ids": workload.local_topk_ids.tolist(),
    }


def make_fp8_randn(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.randn(shape, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)


def run_deep_gemm_compute(args: argparse.Namespace, workload: RankWorkload) -> dict[str, Any]:
    add_import_path(args.sglang_python_path)
    if not args.allow_deepgemm_precompile:
        os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
    # Match the local DSV4-Pro bench scripts; without this, SGLang's default
    # env object enables the FP4-expert recipe, which is not the path profiled
    # by run_dsv4pro_tp8_mock.sh on this H20 setup.
    os.environ.setdefault("SGLANG_DSV4_FP4_EXPERTS", "0")

    from sglang.srt.layers.moe.moe_runner import deep_gemm as deep_gemm_mod
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.deep_gemm import (
        DeepGemmMoeQuantInfo,
        DeepGemmRunnerCore,
        DeepGemmRunnerInput,
    )
    from sglang.srt.environ import envs

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DeepGEMM compute benchmarking")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    local_experts = args.num_experts // args.ep_size
    gateup_size = 2 * args.intermediate_size
    block_shape = [128, 128]
    scale_k = args.hidden_size // block_shape[1]
    dsv4_fp4_experts = (
        envs.SGLANG_DSV4_MODE.get() == "2604"
        and envs.SGLANG_DSV4_FP4_EXPERTS.get()
    )
    # deep_gemm_wrapper passes recipe_b=(1, 32) in DSV4 FP4-expert mode.
    weight_scale_n_gran = 1 if dsv4_fp4_experts else block_shape[0]
    weight_scale_k_gran = 32 if dsv4_fp4_experts else block_shape[1]
    gateup_scale_n = gateup_size // weight_scale_n_gran
    gateup_scale_k = args.hidden_size // weight_scale_k_gran
    down_scale_n = args.hidden_size // weight_scale_n_gran
    down_scale_k = args.intermediate_size // weight_scale_k_gran

    hidden_states = make_fp8_randn(
        (local_experts, workload.m_capacity, args.hidden_size), device
    )
    hidden_states_scale = torch.rand(
        (local_experts, workload.m_capacity, scale_k),
        device=device,
        dtype=torch.float32,
    )
    w13_weight = make_fp8_randn((local_experts, gateup_size, args.hidden_size), device)
    w2_weight = make_fp8_randn((local_experts, args.hidden_size, args.intermediate_size), device)
    w13_scale = torch.rand(
        (local_experts, gateup_scale_n, gateup_scale_k),
        device=device,
        dtype=torch.float32,
    )
    w2_scale = torch.rand(
        (local_experts, down_scale_n, down_scale_k), device=device, dtype=torch.float32
    )
    masked_m = workload.masked_m.to(device=device, dtype=torch.int32)

    config = MoeRunnerConfig(
        num_experts=args.num_experts,
        num_local_experts=local_experts,
        hidden_size=args.hidden_size,
        intermediate_size_per_partition=args.intermediate_size,
        top_k=args.topk,
        params_dtype=torch.bfloat16,
        swiglu_limit=10,
    )
    runner = DeepGemmRunnerCore(config)
    quant_info = DeepGemmMoeQuantInfo(
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        use_fp8=True,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        block_shape=block_shape,
    )
    runner_input = DeepGemmRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=workload.expected_m,
    )
    running_state = {"hidden_states_device": device}

    # The production runner disposes dispatch-owned tensors after one run.  For
    # this standalone microbench we reuse the same synthetic dispatch output.
    deep_gemm_mod.dispose_tensor = lambda _tensor: None

    def run_once() -> None:
        out = runner.run(runner_input, quant_info, running_state).hidden_states
        del out

    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        run_once()
    end.record()
    torch.cuda.synchronize(device)
    result: dict[str, Any] = {
        "latency_us": start.elapsed_time(end) * 1000.0 / args.iters,
    }

    if args.profile_kernels:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.profile_iters):
                run_once()
            torch.cuda.synchronize(device)

        rows = []
        for event in prof.key_averages():
            device_time_us = float(getattr(event, "device_time_total", 0.0) or 0.0)
            if device_time_us <= 0:
                continue
            key = event.key
            if key.startswith(("aten::", "cuda", "Activity ", "Runtime ", "Lazy ")):
                continue
            rows.append(
                {
                    "name": key,
                    "time_us": device_time_us,
                    "per_iter_us": device_time_us / args.profile_iters,
                    "calls": int(event.count),
                }
            )
        rows.sort(key=lambda row: row["time_us"], reverse=True)
        result["profile_iters"] = args.profile_iters
        result["kernel_time_us_per_iter"] = sum(row["time_us"] for row in rows) / args.profile_iters
        result["kernel_breakdown"] = rows[: args.profile_top]

    return result


def main() -> None:
    args = parse_args()
    workload = build_rank_workload(args)
    summary = workload_summary(args, workload)

    if not args.dry_run:
        summary.update(run_deep_gemm_compute(args, workload))
        summary["warmup"] = args.warmup
        summary["iters"] = args.iters

    if args.json:
        print(json.dumps(summary, sort_keys=True))
        return

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
