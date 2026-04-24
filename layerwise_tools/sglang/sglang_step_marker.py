"""NVTX benchmark-step marker for SGLang bench_one_batch.

This is intentionally a local monkey patch for layerwise profiling runs. It
does not change model execution. It only wraps the benchmark run's prefill and
decode calls in NVTX ranges named:

    bench_step::N0000001::bs1::past000000

The existing nsys post-processors use those ranges to slice wall time and
attribute CUDA graph replay kernels back to Module NVTX ranges.

Env:
  LAYERWISE_STEP_MARKER=1            enable patch, default off
  LAYERWISE_STEP_MARKER_SKIP_RUNS=1  skip bench_one_batch warmup run
  LAYERWISE_STEP_MILESTONES=1,2      comma list of step numbers; empty = all
"""
import logging
import os

import torch.cuda.nvtx as nvtx

logger = logging.getLogger(__name__)


def _parse_milestones():
    raw = os.environ.get("LAYERWISE_STEP_MILESTONES", "")
    if not raw.strip():
        return None
    return {int(x) for x in raw.split(",") if x.strip()}


def _arg(args, kwargs, index, name, default=None):
    if len(args) > index:
        return args[index]
    return kwargs.get(name, default)


def _batch_size_from_decode_args(input_token_ids, batch):
    bs = getattr(batch, "batch_size", None)
    if callable(bs):
        bs = bs()
    if bs is not None:
        return int(bs)
    shape = getattr(input_token_ids, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    return 0


def _install():
    if os.environ.get("LAYERWISE_STEP_MARKER", "0") != "1":
        logger.info("[sglang-step-marker] disabled via LAYERWISE_STEP_MARKER!=1")
        return

    import sglang.bench_one_batch as bench

    if getattr(bench, "_layerwise_step_marker_installed", False):
        return

    milestones = _parse_milestones()
    skip_runs = int(os.environ.get("LAYERWISE_STEP_MARKER_SKIP_RUNS", "1"))
    orig_run_once = bench.latency_test_run_once
    orig_extend = bench.extend
    orig_decode = bench.decode
    orig_synchronize = bench.synchronize

    state = {
        "run_idx": 0,
        "active": False,
        "step": 0,
        "input_len": 0,
        "range_open": False,
    }

    def should_mark_step(step):
        return milestones is None or step in milestones

    def close_range():
        if state["range_open"]:
            nvtx.range_pop()
            state["range_open"] = False

    def open_range(label):
        close_range()
        nvtx.range_push(label)
        state["range_open"] = True

    def make_label(step, bs, past):
        return f"bench_step::N{step:07d}::bs{bs}::past{past:06d}"

    def patched_run_once(*args, **kwargs):
        state["run_idx"] += 1
        state["active"] = state["run_idx"] > skip_runs
        state["step"] = 0
        state["input_len"] = int(_arg(args, kwargs, 5, "input_len", 0) or 0)
        try:
            return orig_run_once(*args, **kwargs)
        finally:
            close_range()
            state["active"] = False

    def patched_extend(reqs, model_runner):
        if not state["active"]:
            return orig_extend(reqs, model_runner)

        state["step"] += 1
        step = state["step"]
        if not should_mark_step(step):
            return orig_extend(reqs, model_runner)

        label = make_label(step, len(reqs), 0)
        open_range(label)
        try:
            return orig_extend(reqs, model_runner)
        except Exception:
            close_range()
            raise

    def patched_decode(input_token_ids, batch, model_runner):
        if not state["active"]:
            return orig_decode(input_token_ids, batch, model_runner)

        state["step"] += 1
        step = state["step"]
        if not should_mark_step(step):
            return orig_decode(input_token_ids, batch, model_runner)

        bs = _batch_size_from_decode_args(input_token_ids, batch)
        past = state["input_len"] + max(0, step - 2)
        label = make_label(step, bs, int(past))
        open_range(label)
        try:
            return orig_decode(input_token_ids, batch, model_runner)
        except Exception:
            close_range()
            raise

    def patched_synchronize(device):
        ret = orig_synchronize(device)
        close_range()
        return ret

    bench.latency_test_run_once = patched_run_once
    bench.extend = patched_extend
    bench.decode = patched_decode
    bench.synchronize = patched_synchronize
    bench._layerwise_step_marker_installed = True
    logger.warning(
        "[sglang-step-marker] installed; skip_runs=%s, milestones=%s",
        skip_runs,
        "all" if milestones is None else sorted(milestones),
    )


_install()
