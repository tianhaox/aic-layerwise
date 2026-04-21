"""NVTX step-marker: wraps `GPUModelRunner.execute_model` at milestone step
numbers so that per-step kernel attribution can be sliced from the nsys trace.

Run style:
  - Submit bs=128 requests with isl=1, max_tokens=8192.
  - vLLM engine schedules step 1 = prefill (bs=128 × 1 token), step k (k>=2) =
    pure decode (bs=128, past_kv = k-1).
  - At each milestone step we push an outer NVTX range:
      bench_step::N<NNNNNNN>::bs<B>::past<PPPPPP>
    Example: `bench_step::N0000016::bs128::past000015`.
  - vLLM's layerwise NVTX hooks (enabled by `--enable-layerwise-nvtx-tracing`)
    push their own inner `{'Module': '...'}` ranges; our outer range becomes a
    parent. The sweep parser can then attribute kernels per (step, module).

Env:
  LAYERWISE_STEP_MILESTONES="1,16,32,64,128,256,512,1024,2048,4096,8192"
                             comma list of step numbers to mark (1-indexed)
  LAYERWISE_STEP_MARKER=0    disable
  LAYERWISE_BENCH_MIN_NEW=2  min `scheduled_new_reqs` to treat a call as the
                             start of a real bench iteration (resets counter).
                             Keeps vLLM warmup / profile-run (new_reqs=1) from
                             colliding with the real bs=N prefill at step 1.

Non-milestone steps run as-is (no outer marker). Keeps nsys overhead low
outside the sweep points.
"""
import logging
import os

import torch.cuda.nvtx as nvtx

logger = logging.getLogger(__name__)

_DEFAULT_MILESTONES = "1,16,32,64,128,256,512,1024,2048,4096,8192"


def _parse_milestones() -> set[int]:
    raw = os.environ.get("LAYERWISE_STEP_MILESTONES", _DEFAULT_MILESTONES)
    return {int(x) for x in raw.split(",") if x.strip()}


def _install():
    if os.environ.get("LAYERWISE_STEP_MARKER", "1") != "1":
        logger.info("[step-marker] disabled via LAYERWISE_STEP_MARKER=0")
        return

    milestones = _parse_milestones()
    if not milestones:
        logger.info("[step-marker] no milestones configured; no-op")
        return

    from vllm.v1.worker import gpu_model_runner as _gmr

    orig = _gmr.GPUModelRunner.execute_model
    state = {"n": 0, "started": False}
    min_new = int(os.environ.get("LAYERWISE_BENCH_MIN_NEW", "2"))

    def patched(self, scheduler_output, intermediate_tensors=None):
        # Ignore pre-bench calls (profile_run, single-req sanity) entirely —
        # only start counting once we see a prefill with ≥ min_new new reqs.
        num_new = len(scheduler_output.scheduled_new_reqs)
        if num_new >= min_new:
            state["n"] = 1
            state["started"] = True
        elif state["started"]:
            state["n"] += 1
        else:
            return orig(self, scheduler_output, intermediate_tensors)
        n = state["n"]
        if n not in milestones:
            return orig(self, scheduler_output, intermediate_tensors)

        num_reqs = (
            len(scheduler_output.scheduled_new_reqs)
            + len(scheduler_output.scheduled_cached_reqs.req_ids)
        )
        # past_kv at this step = n - 1 under isl=1 driver
        # (step 1 = prefill, past_kv=0; step k = decode, past_kv=k-1).
        past_kv = n - 1
        label = (
            f"bench_step::N{n:07d}::bs{num_reqs}::past{past_kv:06d}"
        )
        nvtx.range_push(label)
        try:
            return orig(self, scheduler_output, intermediate_tensors)
        finally:
            nvtx.range_pop()

    _gmr.GPUModelRunner.execute_model = patched
    logger.warning(
        f"[step-marker] installed GPUModelRunner.execute_model wrapper, "
        f"milestones={sorted(milestones)}"
    )


_install()
