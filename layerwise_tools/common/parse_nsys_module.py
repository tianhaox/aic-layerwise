"""Per-module NVTX → kernel GPU time attribution, auto-detecting cuda-graph vs eager.

Consumes the `{'Module': 'model.model.layers.3.self_attn'}` NVTX format
emitted by sglang's `--enable-layerwise-nvtx-marker` / vLLM's
`--enable-layerwise-nvtx-tracing` (both use `nvtx_pytorch_hooks.PytHooks`).

Auto-detect:
  - If `CUDA_GRAPH_NODE_EVENTS` has rows → graph-mode path:
    map each graphNodeId (low 32 bits) to the innermost Module NVTX range
    open at the node's capture time, then sum replay kernel durations per
    range. Requires nsys `--cuda-graph-trace=node`.
  - Else → eager-mode path:
    for each `cudaLaunchKernel` RUNTIME row, find the innermost Module
    NVTX range at that host time; join kernels via `correlationId`.

Usage:
    python parse_nsys_module.py <nsys.sqlite>
    python parse_nsys_module.py <nsys.sqlite> --rollup 'layers\\.(\\d+)\\.(\\w+)'
    python parse_nsys_module.py <nsys.sqlite> --top 30 --keep-comm
"""
import argparse
import re
import sqlite3
import sys
from collections import defaultdict

_NODE_INDEX_MASK = 0xFFFFFFFF
_MODULE_NAME_RE = re.compile(r"'Module':\s*'([^']+)'")
_NVTX_GLOB = "*Module*"

# Default kernel-name filter: NCCL / all2all / deep_ep are blocking waits,
# not compute. Override with --keep-comm.
_DEFAULT_KERNEL_DROP = re.compile(
    r"(ncclDevKernel|ncclKernel|all2all|deep_ep|ep_fuse|nvshmem|deepep|multimem_all_reduce|one_shot_all_reduce|two_shot_all_reduce)",
    re.IGNORECASE,
)


def _extract_module(text):
    if text is None:
        return None
    m = _MODULE_NAME_RE.search(text)
    return m.group(1) if m else None


def _has_graph_nodes(cur):
    cur.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='CUDA_GRAPH_NODE_EVENTS'"
    )
    if not cur.fetchone():
        return False
    cur.execute("SELECT COUNT(*) FROM CUDA_GRAPH_NODE_EVENTS")
    return cur.fetchone()[0] > 0


# ======================================================================
# Graph-mode path (graphNodeId-based attribution)
# ======================================================================


def _build_node_to_module(cur):
    """For each graph node, find innermost enclosing Module NVTX range at
    the node's capture time. Returns {node_index_low32: module_name}.
    """
    cur.execute(
        "SELECT text, start, end, globalTid FROM NVTX_EVENTS "
        "WHERE text IS NOT NULL AND text GLOB ?",
        (_NVTX_GLOB,),
    )
    by_tid = defaultdict(list)
    for text, s, e, tid in cur.fetchall():
        name = _extract_module(text)
        if name:
            by_tid[tid].append((s, e, name))
    for tid in by_tid:
        by_tid[tid].sort()

    cur.execute(
        "SELECT DISTINCT graphNodeId, start, end, globalTid "
        "FROM CUDA_GRAPH_NODE_EVENTS WHERE graphNodeId IS NOT NULL"
    )
    node_to_mod = {}
    for gnid, nstart, nend, tid in cur.fetchall():
        # Innermost currently-open = latest-opened still-open at [nstart, nend].
        best_name, best_start = None, -1
        for s, e, name in by_tid.get(tid, []):
            if s > nstart:
                break
            if nend <= e and s > best_start:
                best_name, best_start = name, s
        if best_name is not None:
            node_to_mod[gnid & _NODE_INDEX_MASK] = best_name
    return node_to_mod


def _sum_graph_kernels(cur, node_to_mod, drop_re):
    totals_ns = defaultdict(int)
    n_kern = defaultdict(int)
    excluded_ns = defaultdict(int)

    cur.execute("SELECT id, value FROM StringIds")
    sid = {i: v for i, v in cur.fetchall()}

    cur.execute(
        "SELECT graphNodeId, start, end, shortName "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphNodeId IS NOT NULL"
    )
    for gnid, ks, ke, short in cur.fetchall():
        mod = node_to_mod.get(gnid & _NODE_INDEX_MASK)
        if mod is None:
            continue
        name = sid.get(short, "")
        dur = ke - ks
        if drop_re.search(name):
            excluded_ns[mod] += dur
            continue
        totals_ns[mod] += dur
        n_kern[mod] += 1
    return totals_ns, n_kern, excluded_ns


# ======================================================================
# Eager-mode path (correlationId-based attribution)
# ======================================================================


def _build_corr_to_module(cur):
    """For each RUNTIME row, pick the innermost enclosing Module NVTX range
    on the same thread via LIFO stack sweep. Returns {correlationId: mod}.
    """
    cur.execute(
        "SELECT text, start, end, globalTid FROM NVTX_EVENTS "
        "WHERE text LIKE '%Module%'"
    )
    by_tid = defaultdict(list)
    for text, s, e, tid in cur.fetchall():
        mod = _extract_module(text)
        if mod:
            by_tid[tid].append((s, 0, mod, e))
            by_tid[tid].append((e, 2, mod, s))

    cur.execute(
        "SELECT correlationId, start, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    )
    for cid, s, tid in cur.fetchall():
        by_tid[tid].append((s, 1, cid, None))

    corr_to_mod = {}
    for tid, evs in by_tid.items():
        evs.sort(key=lambda e: (e[0], e[1]))
        stack = []
        for _, kind, payload, aux in evs:
            if kind == 0:
                stack.append((aux, payload))
            elif kind == 2:
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i][1] == payload:
                        del stack[i]
                        break
            else:
                if stack:
                    corr_to_mod[payload] = stack[-1][1]
    return corr_to_mod


def _sum_eager_kernels(cur, corr_to_mod, drop_re):
    totals_ns = defaultdict(int)
    n_kern = defaultdict(int)
    excluded_ns = defaultdict(int)

    cur.execute("SELECT id, value FROM StringIds")
    sid = {i: v for i, v in cur.fetchall()}

    cur.execute(
        "SELECT correlationId, start, end, shortName "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL"
    )
    for cid, ks, ke, short in cur.fetchall():
        mod = corr_to_mod.get(cid)
        if mod is None:
            continue
        name = sid.get(short, "")
        dur = ke - ks
        if drop_re.search(name):
            excluded_ns[mod] += dur
            continue
        totals_ns[mod] += dur
        n_kern[mod] += 1
    return totals_ns, n_kern, excluded_ns


# ======================================================================
# Reporting
# ======================================================================


def _print_report(totals_ns, n_kern, excluded_ns, mode_label, top, rollup):
    if excluded_ns:
        total_excluded_us = sum(excluded_ns.values()) / 1000.0
        print(
            f"(excluded {total_excluded_us:.1f} μs of NCCL/collective kernel "
            f"time — blocking waits not compute)"
        )

    if not totals_ns:
        print(f"no kernels matched ({mode_label} path)", file=sys.stderr)
        sys.exit(1)

    total_all = sum(totals_ns.values())
    print(f"\nMode: {mode_label}")
    print(
        f"Total: {total_all / 1e6:.3f} ms across "
        f"{sum(n_kern.values())} kernels, {len(totals_ns)} modules\n"
    )

    rows = sorted(totals_ns.items(), key=lambda kv: -kv[1])
    if top > 0:
        rows = rows[:top]
    print(f"{'module':<70} {'gpu_us':>10} {'n_kern':>8} {'pct':>6}")
    for name, ns in rows:
        pct = 100 * ns / total_all if total_all else 0
        print(
            f"{name[:70]:<70} {ns / 1000:>10.2f} "
            f"{n_kern[name]:>8d} {pct:>5.1f}%"
        )

    if rollup:
        rollup_re = re.compile(rollup)
        grouped = defaultdict(int)
        grouped_k = defaultdict(int)
        for name, ns in totals_ns.items():
            m = rollup_re.search(name)
            if not m:
                continue
            key = "|".join(m.groups()) if m.groups() else m.group(0)
            grouped[key] += ns
            grouped_k[key] += n_kern[name]
        if grouped:
            print(f"\nRollup by {rollup!r}:")
            print(f"  {'group':<60} {'total_us':>10} {'n_kern':>8}")
            for k, ns in sorted(grouped.items(), key=lambda kv: -kv[1]):
                print(f"  {k:<60} {ns / 1000:>10.2f} {grouped_k[k]:>8d}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("sqlite_path")
    ap.add_argument(
        "--rollup",
        default=None,
        help="Regex with capture groups; sum ranges by the captured key.",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=40,
        help="Show top-N modules by GPU time (0 = all). Default 40.",
    )
    ap.add_argument(
        "--keep-comm",
        action="store_true",
        help="Keep NCCL / all2all / deep_ep kernels (default: filter out).",
    )
    args = ap.parse_args()

    drop_re = re.compile("^$") if args.keep_comm else _DEFAULT_KERNEL_DROP

    con = sqlite3.connect(args.sqlite_path)
    cur = con.cursor()

    if _has_graph_nodes(cur):
        print("[pass 1] detected CUDA_GRAPH_NODE_EVENTS → graph-mode path",
              file=sys.stderr)
        node_to_mod = _build_node_to_module(cur)
        print(f"  mapped {len(node_to_mod)} graph nodes to modules",
              file=sys.stderr)
        totals_ns, n_kern, excluded = _sum_graph_kernels(cur, node_to_mod, drop_re)
        mode_label = "cuda-graph (graphNodeId)"
    else:
        print("[pass 1] no graph nodes → eager-mode (correlationId) path",
              file=sys.stderr)
        corr_to_mod = _build_corr_to_module(cur)
        print(f"  mapped {len(corr_to_mod)} correlations to modules",
              file=sys.stderr)
        totals_ns, n_kern, excluded = _sum_eager_kernels(cur, corr_to_mod, drop_re)
        mode_label = "eager (correlationId)"

    con.close()
    _print_report(totals_ns, n_kern, excluded, mode_label, args.top, args.rollup)


if __name__ == "__main__":
    main()
