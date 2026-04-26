"""Validated per-step kernel comparison for single-GPU mock vs true EP8.

This is intentionally narrower than parse_nsys_step_sweep.py:

* It keeps the exact layer module (for MHC/HCA kernels) instead of only
  rolling up descendants like layers.0.self_attn.
* It reports both module-level rank-max totals and per-kernel rank-max rows.
  These are different reductions and must not be mixed.
* It validates the expected modules/ranks before writing final comparison
  files, so missing parser coverage fails loudly.

Example:
  python layerwise_tools/common/compare_mock_ep8_step_kernels.py \\
      --mock-sqlite /tmp/mock.sqlite \\
      --ep8-sqlite /tmp/ep8.sqlite \\
      --layer 0 --step 2 \\
      --out-dir /tmp/exp3_mock_vs_ep8_validated_parser
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from parse_nsys_step_sweep import (
    _build_nvtx_lookups,
    _innermost_module_at_with_start,
    _query_kernels,
    _step_of,
)


_COMM_RE = re.compile(
    r"(ncclDevKernel|ncclKernel|all2all|deep_ep|ep_fuse|nvshmem|deepep|"
    r"multimem_all_reduce|one_shot_all_reduce|two_shot_all_reduce|"
    r"^dispatch$|^combine$)",
    re.IGNORECASE,
)

_KNOWN_KERNELS = (
    "mhc_post_tilelang_kernel",
    "mhc_pre_big_fuse_tilelang_kernel",
    "mhc_pre_gemm_sqrsum_splitk_stage_0_kernel",
    "mhc_pre_gemm_sqrsum_splitk_stage_1_kernel",
    "flash_fwd_splitkv_mla_fp8_sparse_kernel",
    "flash_fwd_mla_combine_kernel",
    "fused_store_flashmla_cache",
    "get_mla_metadata_kernel",
    "smxx_paged_mqa_logits_metadata",
    "_init_compressed_attn_metadata_kernel",
    "deepseek_rope_kernel",
    "fused_norm_rope",
    "flash_c128_decode",
    "deepseek_v3_topk_kernel",
    "compute_masked_m_triton_kernel",
    "compute_seg_indptr_triton_kernel",
    "deepgemm_compute_src2dst_triton_kernel",
    "fill_gateup_input_triton_kernel",
    "post_reorder_triton_kernel",
    "radixSortKVInPlace",
    "per_token_group_quant_8bit_kernel",
    "silu_mul_quant_kernel",
    "sm90_fp8_gemm_1d2d_impl",
    "transpose_fp32",
    "RMSNormKernel",
    "_rms_normalize_kernel",
    "act_and_mul_kernel",
    "splitKreduce_kernel",
    "nvjet_tss_64x8_64x16_2x1_v_bz_splitK_TNT",
    "nvjet_tss_64x8_64x16_1x1_h_bz_splitK_TNT",
    "nvjet_tst_256x8_64x6_2x1_v_bz_TNT",
    "triton_poi_fused_arange_ge_index_put_lift_fresh_0",
    "index_elementwise_kernel",
    "elementwise_kernel_with_index",
    "unrolled_elementwise_kernel",
    "vectorized_elementwise_kernel",
    "elementwise_kernel",
    "dispatch",
    "combine",
)


@dataclass(frozen=True)
class KernelRecord:
    source: str
    rank: int
    step: int
    bs: int
    past: int
    module: str
    sub_module: str
    kernel: str
    is_comm: int
    gpu_ns: int


def _fmt_us(ns: int | float | None) -> str:
    if ns is None:
        return ""
    return f"{ns / 1000.0:.3f}"


def _ratio(left_ns: int | None, right_ns: int | None) -> str:
    if left_ns is None or right_ns is None or left_ns == 0:
        return ""
    return f"{right_ns / left_ns:.2f}"


def _delta_us(left_ns: int | None, right_ns: int | None) -> str:
    if right_ns is None:
        right_ns = 0
    if left_ns is None:
        left_ns = 0
    return f"{(right_ns - left_ns) / 1000.0:.3f}"


def _normalize_kernel(name: str) -> str:
    if not name:
        return ""
    for known in _KNOWN_KERNELS:
        if known in name:
            return known
    stripped = name.strip()
    if "(" in stripped:
        stripped = stripped.split("(", 1)[0].strip()
    if "::" in stripped:
        stripped = stripped.rsplit("::", 1)[-1]
    parts = stripped.split()
    return parts[-1] if parts else stripped


def _classify_submodule(module: str, base_module: str) -> str | None:
    if module == base_module:
        return "<layer>"
    prefix = base_module + "."
    if not module.startswith(prefix):
        return None
    return module[len(prefix) :]


def _load_string_ids(cur: sqlite3.Cursor) -> dict[int, str]:
    cur.execute("SELECT id, value FROM StringIds")
    return {int(i): str(v) for i, v in cur.fetchall()}


def _extract_records(
    sqlite_path: str,
    source: str,
    layer: int,
    step_filter: int,
) -> tuple[list[KernelRecord], dict[str, int]]:
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    step_wins, mod_ivs = _build_nvtx_lookups(cur)
    sid = _load_string_ids(cur)

    kern_meta: dict[tuple[int, int, int | None], tuple[int, int, int, int, int]] = {}
    kern_best_mod: dict[tuple[int, int, int | None], tuple[int, str]] = {}

    for row in _query_kernels(cur):
        cid, gnid, ks, ke, short_id, tid, rt_start, cap_s, cap_e = row
        key = (int(tid), int(cid), None if gnid is None else int(gnid))
        if key not in kern_meta:
            kern_meta[key] = (int(ks), int(ke), int(short_id), int(tid), int(rt_start))
        if cap_s is None or cap_e is None:
            continue
        if key in kern_best_mod:
            continue
        mod_start_name = _innermost_module_at_with_start(
            mod_ivs.get(tid, []), int(cap_s), int(cap_e)
        )
        if mod_start_name is not None:
            kern_best_mod[key] = mod_start_name

    tids = sorted({tid for (tid, _, _) in kern_meta})
    tid_to_rank = {tid: i for i, tid in enumerate(tids)}
    base_module = f"model.model.layers.{layer}"

    records: list[KernelRecord] = []
    stats = defaultdict(int)
    for key, (ks, ke, short_id, tid, rt_start) in kern_meta.items():
        step = _step_of(step_wins.get(tid, []), rt_start)
        if step is None:
            stats["outside_step_window"] += 1
            continue
        step_n, bs, past = step
        if step_n != step_filter:
            stats["other_step"] += 1
            continue
        mod_entry = kern_best_mod.get(key)
        if mod_entry is None:
            stats["missing_module"] += 1
            continue
        _, module = mod_entry
        sub_module = _classify_submodule(module, base_module)
        if sub_module is None:
            stats["outside_layer"] += 1
            continue
        kernel = _normalize_kernel(sid.get(short_id, ""))
        records.append(
            KernelRecord(
                source=source,
                rank=tid_to_rank[tid],
                step=step_n,
                bs=bs,
                past=past,
                module=module,
                sub_module=sub_module,
                kernel=kernel,
                is_comm=1 if _COMM_RE.search(kernel) else 0,
                gpu_ns=ke - ks,
            )
        )

    stats["ranks"] = len(tids)
    stats["raw_kernels"] = len(kern_meta)
    stats["matched_records"] = len(records)
    con.close()
    return records, dict(stats)


def _aggregate(
    records: Iterable[KernelRecord],
    key_fields: tuple[str, ...],
) -> dict[tuple, dict[str, int]]:
    out: dict[tuple, dict[str, int]] = {}
    for rec in records:
        key = tuple(getattr(rec, f) for f in key_fields)
        row = out.setdefault(key, {"gpu_ns": 0, "n": 0})
        row["gpu_ns"] += rec.gpu_ns
        row["n"] += 1
    return out


def _rankmax(
    per_rank: dict[tuple, dict[str, int]],
    key_len_without_rank: int,
) -> dict[tuple, dict[str, int]]:
    out: dict[tuple, dict[str, int]] = {}
    for key, stats in per_rank.items():
        group_key = key[:key_len_without_rank]
        rank = key[key_len_without_rank]
        current = out.get(group_key)
        if current is None or stats["gpu_ns"] > current["gpu_ns"]:
            out[group_key] = {
                "gpu_ns": stats["gpu_ns"],
                "n": stats["n"],
                "rank": rank,
            }
    return out


def _validate_records(
    name: str,
    records: list[KernelRecord],
    expected_ranks: int,
    require_mhc: bool,
) -> None:
    if not records:
        raise RuntimeError(f"{name}: no records matched requested layer/step")
    ranks = sorted({r.rank for r in records})
    if len(ranks) != expected_ranks:
        raise RuntimeError(
            f"{name}: expected {expected_ranks} ranks, found {len(ranks)} ({ranks})"
        )
    submods = {r.sub_module for r in records}
    required = {
        "<layer>",
        "input_layernorm",
        "post_attention_layernorm",
        "mlp",
        "mlp.experts",
        "mlp.gate",
        "self_attn",
        "self_attn.wqkv_a",
        "self_attn.wq_b",
        "self_attn.wo_b",
    }
    missing = sorted(required - submods)
    if missing:
        raise RuntimeError(f"{name}: missing required submodules: {', '.join(missing)}")
    if require_mhc:
        layer_kernels = {r.kernel for r in records if r.sub_module == "<layer>"}
        required_mhc = {
            "mhc_pre_big_fuse_tilelang_kernel",
            "mhc_pre_gemm_sqrsum_splitk_stage_0_kernel",
            "mhc_pre_gemm_sqrsum_splitk_stage_1_kernel",
            "mhc_post_tilelang_kernel",
        }
        missing_mhc = sorted(required_mhc - layer_kernels)
        if missing_mhc:
            raise RuntimeError(f"{name}: missing MHC kernels: {', '.join(missing_mhc)}")


def _write_kernel_compare(
    path: Path,
    mock_records: list[KernelRecord],
    ep8_records: list[KernelRecord],
) -> None:
    mock = _aggregate(mock_records, ("sub_module", "kernel", "is_comm"))
    ep8_per_rank = _aggregate(ep8_records, ("sub_module", "kernel", "is_comm", "rank"))
    ep8 = _rankmax(ep8_per_rank, 3)
    keys = sorted(set(mock) | set(ep8))
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sub_module",
                "kernel",
                "is_comm",
                "mock_us",
                "mock_n",
                "ep8_rankmax_us",
                "ep8_n",
                "ep8_max_rank",
                "delta_ep8_minus_mock_us",
                "ratio_ep8_over_mock",
            ]
        )
        for key in keys:
            m = mock.get(key)
            e = ep8.get(key)
            m_ns = m["gpu_ns"] if m else None
            e_ns = e["gpu_ns"] if e else None
            writer.writerow(
                [
                    key[0],
                    key[1],
                    key[2],
                    _fmt_us(m_ns),
                    m["n"] if m else "",
                    _fmt_us(e_ns),
                    e["n"] if e else "",
                    e["rank"] if e else "",
                    _delta_us(m_ns, e_ns),
                    _ratio(m_ns, e_ns),
                ]
            )


def _write_module_summary(
    path: Path,
    mock_records: list[KernelRecord],
    ep8_records: list[KernelRecord],
) -> None:
    mock = _aggregate(mock_records, ("sub_module", "is_comm"))
    ep8_per_rank = _aggregate(ep8_records, ("sub_module", "is_comm", "rank"))
    ep8 = _rankmax(ep8_per_rank, 2)
    keys = sorted(set(mock) | set(ep8))
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sub_module",
                "is_comm",
                "mock_us",
                "mock_n",
                "ep8_rankmax_us",
                "ep8_n",
                "ep8_max_rank",
                "delta_ep8_minus_mock_us",
                "ratio_ep8_over_mock",
            ]
        )
        for key in keys:
            m = mock.get(key)
            e = ep8.get(key)
            m_ns = m["gpu_ns"] if m else None
            e_ns = e["gpu_ns"] if e else None
            writer.writerow(
                [
                    key[0],
                    key[1],
                    _fmt_us(m_ns),
                    m["n"] if m else "",
                    _fmt_us(e_ns),
                    e["n"] if e else "",
                    e["rank"] if e else "",
                    _delta_us(m_ns, e_ns),
                    _ratio(m_ns, e_ns),
                ]
            )


def _write_ep8_per_rank(
    path: Path,
    ep8_records: list[KernelRecord],
) -> None:
    per_rank = _aggregate(ep8_records, ("sub_module", "is_comm", "rank"))
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sub_module", "is_comm", "rank", "gpu_us", "n_kernels"])
        for key, stats in sorted(per_rank.items()):
            writer.writerow([key[0], key[1], key[2], _fmt_us(stats["gpu_ns"]), stats["n"]])


def _write_readme(
    path: Path,
    args: argparse.Namespace,
    mock_records: list[KernelRecord],
    ep8_records: list[KernelRecord],
    mock_stats: dict[str, int],
    ep8_stats: dict[str, int],
) -> None:
    mock_modules = _aggregate(mock_records, ("sub_module", "is_comm"))
    ep8_modules = _rankmax(_aggregate(ep8_records, ("sub_module", "is_comm", "rank")), 2)

    def mock_total_for(comm_filter: int | None) -> int:
        return sum(
            r.gpu_ns
            for r in mock_records
            if comm_filter is None or r.is_comm == comm_filter
        )

    def ep8_rankmax_total_for(comm_filter: int | None) -> tuple[int, int]:
        filtered = [
            r
            for r in ep8_records
            if comm_filter is None or r.is_comm == comm_filter
        ]
        if not filtered:
            return -1, 0
        per_rank = _aggregate(filtered, ("rank",))
        return max(
            ((rank[0], stats["gpu_ns"]) for rank, stats in per_rank.items()),
            key=lambda item: item[1],
        )

    focus_keys = [
        ("<layer>", 0),
        ("self_attn", 0),
        ("self_attn.wqkv_a", 0),
        ("self_attn.wq_b", 0),
        ("self_attn.wo_b", 0),
        ("mlp.experts", 0),
        ("mlp.experts", 1),
        ("mlp.shared_experts.gate_up_proj", 0),
        ("mlp.shared_experts.down_proj", 0),
    ]

    lines = [
        "# Mock vs EP8 Validated Parser",
        "",
        f"Mock trace: `{args.mock_sqlite}`",
        "",
        f"EP8 trace: `{args.ep8_sqlite}`",
        "",
        f"Layer: `{args.layer}`",
        "",
        f"Step: `{args.step}`",
        "",
        "This output is generated by `layerwise_tools/common/compare_mock_ep8_step_kernels.py`.",
        "Module totals use per-rank aggregation first, then rank-max for EP8.",
        "Kernel rows use per-kernel rank-max for EP8 and are diagnostic when the max rank differs by kernel.",
        "",
        "## Validation",
        "",
        f"- Mock ranks: {mock_stats.get('ranks')} expected {args.mock_ranks}",
        f"- EP8 ranks: {ep8_stats.get('ranks')} expected {args.ep8_ranks}",
        f"- Mock matched kernels: {mock_stats.get('matched_records')}",
        f"- EP8 matched kernels: {ep8_stats.get('matched_records')}",
        f"- Exact layer module and MHC kernels: present",
        "",
        "## Layer Total",
        "",
        "| metric | mock us | ep8 rankmax us | ep8 rank | delta us | ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for label, comm_filter in (
        ("layer attributed compute kernels", 0),
        ("layer attributed communication kernels", 1),
        ("layer attributed all kernels", None),
    ):
        mock_total = mock_total_for(comm_filter)
        ep8_total_rank, ep8_total = ep8_rankmax_total_for(comm_filter)
        lines.append(
            f"| {label} | {_fmt_us(mock_total)} | {_fmt_us(ep8_total)} | "
            f"{'' if ep8_total_rank < 0 else ep8_total_rank} | "
            f"{_delta_us(mock_total, ep8_total)} | {_ratio(mock_total, ep8_total)} |"
        )

    lines.extend(
        [
            "",
            "Compute and communication totals are reduced separately. The all-kernel row is for wall-critical diagnosis only.",
            "",
            "## Focus Module Summary",
            "",
            "| sub_module | comm | mock us | ep8 rankmax us | ep8 rank | delta us | ratio |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for key in focus_keys:
        m = mock_modules.get(key)
        e = ep8_modules.get(key)
        m_ns = m["gpu_ns"] if m else None
        e_ns = e["gpu_ns"] if e else None
        lines.append(
            f"| {key[0]} | {key[1]} | {_fmt_us(m_ns)} | {_fmt_us(e_ns)} | "
            f"{e['rank'] if e else ''} | {_delta_us(m_ns, e_ns)} | {_ratio(m_ns, e_ns)} |"
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `module_summary.csv`: module totals; EP8 is rank-max after module aggregation.",
            "- `kernel_compare.csv`: per-kernel table; EP8 is rank-max per kernel.",
            "- `ep8_per_rank_module_summary.csv`: EP8 module totals by rank before rank-max reduction.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mock-sqlite", required=True)
    ap.add_argument("--ep8-sqlite", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mock-ranks", type=int, default=1)
    ap.add_argument("--ep8-ranks", type=int, default=8)
    ap.add_argument("--allow-missing-mhc", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    for p in (args.mock_sqlite, args.ep8_sqlite):
        if not os.path.exists(p):
            print(f"missing sqlite: {p}", file=sys.stderr)
            return 2

    mock_records, mock_stats = _extract_records(
        args.mock_sqlite, "mock", args.layer, args.step
    )
    ep8_records, ep8_stats = _extract_records(args.ep8_sqlite, "ep8", args.layer, args.step)
    require_mhc = not args.allow_missing_mhc
    _validate_records("mock", mock_records, args.mock_ranks, require_mhc)
    _validate_records("ep8", ep8_records, args.ep8_ranks, require_mhc)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_kernel_compare(out_dir / "kernel_compare.csv", mock_records, ep8_records)
    _write_module_summary(out_dir / "module_summary.csv", mock_records, ep8_records)
    _write_ep8_per_rank(out_dir / "ep8_per_rank_module_summary.csv", ep8_records)
    _write_readme(
        out_dir / "README.md",
        args,
        mock_records,
        ep8_records,
        mock_stats,
        ep8_stats,
    )

    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
