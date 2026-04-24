#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-nsys_dsv4pro_tp8mock}"
TOP="${TOP:-40}"
SORT="${SORT:-time}"
KERNEL_SORT="${KERNEL_SORT:-execution}"
KERNEL_TOP="${KERNEL_TOP:-0}"
MODE="${MODE:-smoke}"
CAPTURE_RANGE="${CAPTURE_RANGE:-auto}"

if [[ "${CAPTURE_RANGE}" == "auto" ]]; then
  # For decode with CUDA graph, capture the whole process so graph capture
  # events and replay kernels are both present for graph-node attribution.
  if [[ "${MODE}" == "decode" ]]; then
    CAPTURE_RANGE=0
  else
    CAPTURE_RANGE=1
  fi
fi

if [[ "${CAPTURE_RANGE}" == "1" ]]; then
  export BENCH_PROFILE="${BENCH_PROFILE:-1}"
fi

NSYS_ARGS=(
  profile
  --cuda-graph-trace=node
  --force-overwrite=true
  -o "${OUT}"
)
if [[ "${CAPTURE_RANGE}" == "1" ]]; then
  NSYS_ARGS+=(--capture-range=cudaProfilerApi --capture-range-end=stop)
fi

MODE="${MODE}" nsys "${NSYS_ARGS[@]}" "${SCRIPT_DIR}/run_dsv4pro_tp8_mock.sh"

nsys export \
  --type sqlite \
  --force-overwrite=true \
  --output "${OUT}.sqlite" \
  "${OUT}.nsys-rep"

PARSE_ARGS=(
  "${OUT}.sqlite"
  --sort "${SORT}"
  --top "${TOP}"
)
if [[ "${SHOW_KERNELS:-0}" == "1" ]]; then
  PARSE_ARGS+=(--show-kernels --kernel-sort "${KERNEL_SORT}" --kernel-top "${KERNEL_TOP}")
  if [[ -n "${MODULE_REGEX:-}" ]]; then
    PARSE_ARGS+=(--module-regex "${MODULE_REGEX}")
  fi
fi

python "${SCRIPT_DIR}/../common/parse_nsys_module.py" "${PARSE_ARGS[@]}"
