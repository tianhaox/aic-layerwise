"""Per-step-milestone kernel attribution from an nsys sqlite.

Designed for runs produced by `vllm_step_marker.py`:
  outer NVTX range:  `bench_step::N<nnnnnnn>::bs<B>::past<pppppp>`
  inner NVTX ranges: `{'Module': '<dotted.module.path>'}` (PytHooks layerwise)

Output: one row per milestone step × Module (rollup-regex optional), showing
per-step kernel GPU time and kernel count.

Attribution approach (matches TensorRT-LLM's layer_wise_benchmarks/parse.py):
  For cuda-graph replay kernels, traditional correlationId→host-time won't
  hit any Module NVTX because host is inside a single `cudaGraphLaunch`.
  We use a two-step SQL JOIN through `CUDA_GRAPH_NODE_EVENTS` to recover
  the *stream-capture-time* timestamp of each kernel's original node:

    replay kernel.graphNodeId
      → CGE1.graphNodeId          (matches instantiate-time row)
      → CGE1.originalGraphNodeId  (stable template id)
      → CGE2.graphNodeId          (matches stream-capture-time row)
      → CGE2.start/end            (host time when NVTX stack was open)

  The JOIN is constrained on globalTid too so a merged multi-rank sqlite
  doesn't produce 8x cardinality.

  Eager kernels (graphNodeId IS NULL) take `capture_start = R.start`
  (runtime host time, which already falls inside Module NVTX).

Step attribution uses `R.start` (host time of cudaLaunchKernel/cudaGraphLaunch)
against the `bench_step::*` NVTX windows (outer marker is host-level).

Usage:
  python parse_nsys_step_sweep.py <sqlite> \\
      --rollup '(self_attn|mlp|input_layernorm|post_attention_layernorm)' \\
      --layer 3
"""
import argparse
import bisect
import re
import sqlite3
import sys
from collections import defaultdict

_BENCH_STEP_RE = re.compile(r"bench_step::N(\d+)::bs(\d+)::past(\d+)")
_MODULE_RE = re.compile(r"'Module':\s*'([^']+)'")

_DEFAULT_KERNEL_DROP = re.compile(
    r"(ncclDevKernel|ncclKernel|all2all|deep_ep|ep_fuse|nvshmem|"
    r"multimem_all_reduce|one_shot_all_reduce|two_shot_all_reduce)",
    re.IGNORECASE,
)


def _extract_module(text):
    if text is None:
        return None
    m = _MODULE_RE.search(text)
    return m.group(1) if m else None


def _extract_bench_step(text):
    if text is None:
        return None
    m = _BENCH_STEP_RE.search(text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _query_kernels(cur):
    """Yield (cid, kernel_start, kernel_end, short_name_id, tid,
              runtime_start, capture_start, capture_end).

    Uses TRT-LLM's 2-step JOIN via `originalGraphNodeId` for graph kernels.
    Constrains JOINs on globalTid so multi-rank sqlites stay 1:1 at the
    CGE1→CGE2 level.

    vLLM captures/instantiates the same graph template many times during
    warmup, so one replay kernel's graphNodeId can match hundreds of CGE1
    rows (one per instantiate). The JOIN result thus has duplicates per
    kernel. Callers must dedup by correlationId.
    """
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='CUDA_GRAPH_NODE_EVENTS'"
    )
    has_graph = cur.fetchone() is not None

    query = """
    SELECT K.correlationId, K.graphNodeId, K.start, K.end, K.shortName,
           R.globalTid, R.start AS runtime_start,
           R.start AS capture_start, R.end AS capture_end
    FROM CUPTI_ACTIVITY_KIND_KERNEL K
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME R ON K.correlationId = R.correlationId
    WHERE K.graphNodeId IS NULL
    """
    if has_graph:
        query += """
        UNION ALL
        SELECT K.correlationId, K.graphNodeId, K.start, K.end, K.shortName,
               R.globalTid, R.start AS runtime_start,
               CGE2.start AS capture_start, CGE2.end AS capture_end
        FROM CUPTI_ACTIVITY_KIND_KERNEL K
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME R ON K.correlationId = R.correlationId
        LEFT JOIN CUDA_GRAPH_NODE_EVENTS CGE1
          ON K.graphNodeId = CGE1.graphNodeId
         AND R.globalTid = CGE1.globalTid
         AND CGE1.originalGraphNodeId IS NOT NULL
        LEFT JOIN CUDA_GRAPH_NODE_EVENTS CGE2
          ON CGE1.originalGraphNodeId = CGE2.graphNodeId
         AND CGE1.globalTid = CGE2.globalTid
        WHERE K.graphNodeId IS NOT NULL
        """
    cur.execute(query)
    return cur.fetchall()


def _build_nvtx_lookups(cur):
    """Return (step_wins_by_tid, module_intervals_by_tid).

    step_wins_by_tid[tid] = sorted list of (start, end, (n, bs, past))
    module_intervals_by_tid[tid] = sorted list of (start, end, module_name)
    """
    cur.execute(
        "SELECT text, start, end, globalTid FROM NVTX_EVENTS "
        "WHERE text IS NOT NULL AND (text LIKE 'bench_step::%' OR text LIKE '%Module%')"
    )
    step_wins = defaultdict(list)
    mod_ivs = defaultdict(list)
    for text, s, e, tid in cur.fetchall():
        step = _extract_bench_step(text)
        if step is not None:
            step_wins[tid].append((s, e, step))
            continue
        mod = _extract_module(text)
        if mod:
            mod_ivs[tid].append((s, e, mod))
    for tid in step_wins:
        step_wins[tid].sort()
    for tid in mod_ivs:
        mod_ivs[tid].sort()
    return step_wins, mod_ivs


def _step_of(step_wins_for_tid, host_time):
    """Binary search host_time against per-tid step windows."""
    if not step_wins_for_tid:
        return None
    starts = [s for s, _, _ in step_wins_for_tid]
    idx = bisect.bisect_right(starts, host_time) - 1
    if idx < 0:
        return None
    s, e, step = step_wins_for_tid[idx]
    if s <= host_time < e:
        return step
    return None


def _innermost_module_at(mod_ivs_for_tid, capture_start, capture_end):
    """Find innermost Module NVTX range enclosing [capture_start, capture_end].

    "Innermost" = the one with the latest `start` among those enclosing the
    kernel's capture interval.
    """
    best = None
    best_start = -1
    # Linear scan; O(N) per kernel. N is typically small (~200-300 intervals).
    # Binary-search prune on start <= capture_start.
    starts = [s for s, _, _ in mod_ivs_for_tid]
    hi = bisect.bisect_right(starts, capture_start)
    for i in range(hi):
        s, e, name = mod_ivs_for_tid[i]
        if s <= capture_start and capture_end <= e and s > best_start:
            best = name
            best_start = s
    return best


def _sum_kernels(cur, kernel_drop_re):
    step_wins, mod_ivs = _build_nvtx_lookups(cur)

    cur.execute("SELECT id, value FROM StringIds")
    sid = {i: v for i, v in cur.fetchall()}

    # A single `cudaGraphLaunch` call fires every kernel in the graph with
    # the same host correlationId but distinct `graphNodeId` per kernel.
    # So the dedup key must be (correlationId, graphNodeId) — both to
    # avoid collapsing ~50 kernels in one graph into 1, and to dedup the
    # many CGE1/CGE2 JOIN rows coming from multiple instantiates of the
    # same template.
    kern_meta = {}  # (cid, gnid) → (ks, ke, short_id, tid, rt_start)
    kern_best_mod = {}  # (cid, gnid) → (best_mod_start, best_mod_name)
    for row in _query_kernels(cur):
        cid, gnid, ks, ke, short_id, tid, rt_start, cap_s, cap_e = row
        key = (cid, gnid)
        if key not in kern_meta:
            kern_meta[key] = (ks, ke, short_id, tid, rt_start)
        if cap_s is None:
            continue
        if key in kern_best_mod:
            continue  # already attributed via another instantiate's candidate
        mod_start_name = _innermost_module_at_with_start(
            mod_ivs.get(tid, []), cap_s, cap_e
        )
        if mod_start_name is None:
            continue
        kern_best_mod[key] = mod_start_name

    gpu_ns = defaultdict(int)
    n_k = defaultdict(int)
    unmatched_step = 0
    unmatched_module = 0
    dropped_comm = 0

    for key, (ks, ke, short_id, tid, rt_start) in kern_meta.items():
        name = sid.get(short_id, "")
        if kernel_drop_re.search(name):
            dropped_comm += 1
            continue
        step = _step_of(step_wins.get(tid, []), rt_start)
        if step is None:
            unmatched_step += 1
            continue
        mod_entry = kern_best_mod.get(key)
        if mod_entry is None:
            unmatched_module += 1
            continue
        _, mod = mod_entry
        out_key = (step, mod, tid)
        gpu_ns[out_key] += ke - ks
        n_k[out_key] += 1

    return gpu_ns, n_k, (unmatched_step, unmatched_module, dropped_comm)


def _innermost_module_at_with_start(mod_ivs_for_tid, capture_start, capture_end):
    """Like _innermost_module_at but returns (start, name) tuple."""
    best = None
    best_start = -1
    starts = [s for s, _, _ in mod_ivs_for_tid]
    hi = bisect.bisect_right(starts, capture_start)
    for i in range(hi):
        s, e, name = mod_ivs_for_tid[i]
        if s <= capture_start and capture_end <= e and s > best_start:
            best = name
            best_start = s
    return (best_start, best) if best is not None else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite_path")
    ap.add_argument("--rollup", default=r"layers\.(\d+)\.(\w+)",
                    help="Regex with capture groups; module rolled up to captures. "
                         "Default groups by (layer_idx, first sub-submodule).")
    ap.add_argument("--layer", type=int, default=None,
                    help="Filter rolled-up rows whose first-capture == this layer.")
    ap.add_argument("--keep-comm", action="store_true",
                    help="Include NCCL / all2all / deep_ep kernels in totals.")
    ap.add_argument("--per-rank", action="store_true",
                    help="Emit one row per (step, rank) instead of summing across ranks.")
    args = ap.parse_args()

    kernel_drop_re = re.compile("^$") if args.keep_comm else _DEFAULT_KERNEL_DROP

    con = sqlite3.connect(args.sqlite_path)
    cur = con.cursor()

    print("[parse] running 2-step JOIN via originalGraphNodeId ...", file=sys.stderr)
    gpu_ns, n_k, (miss_step, miss_mod, dropped) = _sum_kernels(cur, kernel_drop_re)
    print(f"[parse] kernels attributed: {sum(n_k.values())}, "
          f"outside milestone window: {miss_step}, "
          f"no Module NVTX: {miss_mod}, dropped (comm): {dropped}",
          file=sys.stderr)

    con.close()

    # Detect multi-rank.
    tids = sorted({tid for (_, _, tid) in gpu_ns})
    tid_to_rank = {tid: i for i, tid in enumerate(tids)}
    n_ranks = len(tids)
    if n_ranks > 1:
        mode = "per-rank" if args.per_rank else f"SUMMED across {n_ranks} ranks (divide for per-rank)"
        print(f"[multi-rank] {n_ranks} ranks detected — output mode: {mode}",
              file=sys.stderr)

    rollup_re = re.compile(args.rollup)
    # Aggregate by (step_meta, roll_key [, rank]).
    rolled_ns = defaultdict(int)
    rolled_k = defaultdict(int)
    step_seen = set()
    roll_keys = set()
    rank_seen = set()
    for (step_meta, mod, tid), ns in gpu_ns.items():
        m = rollup_re.search(mod or "")
        if not m:
            continue
        roll_key = m.groups() if m.groups() else (m.group(0),)
        if args.layer is not None and roll_key and str(args.layer) != str(roll_key[0]):
            continue
        rank = tid_to_rank[tid] if args.per_rank else None
        step_seen.add(step_meta)
        roll_keys.add(roll_key)
        rank_seen.add(rank)
        key = (step_meta, roll_key, rank)
        rolled_ns[key] += ns
        rolled_k[key] += n_k[(step_meta, mod, tid)]

    if not step_seen:
        print("No rows matched the rollup regex.", file=sys.stderr)
        sys.exit(1)

    sorted_steps = sorted(step_seen, key=lambda s: s[0])
    sorted_keys = sorted(roll_keys)
    sorted_ranks = sorted(rank_seen, key=lambda r: -1 if r is None else r)

    header_keys = ["|".join(map(str, k)) for k in sorted_keys]
    col_w = max(14, max(len(h) for h in header_keys) + 1)

    print()
    rank_col = f"{'rank':>5}  " if args.per_rank else ""
    print(f"{'step':>8} {'bs':>5} {'past_kv':>8}  {rank_col}"
          + "  ".join(f"{h:>{col_w}}" for h in header_keys))
    for step_meta in sorted_steps:
        for rank in sorted_ranks:
            step_n, bs, past = step_meta
            row_cells = []
            any_nonzero = False
            for k in sorted_keys:
                ns = rolled_ns.get((step_meta, k, rank), 0)
                if ns > 0:
                    any_nonzero = True
                    row_cells.append(f"{ns/1000:>{col_w-4}.1f} μs")
                else:
                    row_cells.append(f"{'-':>{col_w}}")
            if not any_nonzero:
                continue
            rank_prefix = f"{rank:>5}  " if args.per_rank and rank is not None else ""
            print(f"{step_n:>8} {bs:>5} {past:>8}  {rank_prefix}"
                  + "  ".join(row_cells))


if __name__ == "__main__":
    main()
