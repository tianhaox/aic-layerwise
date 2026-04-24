#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V4-Pro}"
GPU="${GPU:-0}"
MODE="${MODE:-smoke}"
TARGET_LAYERS="${TARGET_LAYERS:-0}"
NUM_LAYERS="${NUM_LAYERS:-1}"
case "${MODE}" in
  prefill)
    BATCH_SIZE="${BATCH_SIZE:-1}"
    INPUT_LEN="${INPUT_LEN:-8192}"
    OUTPUT_LEN="${OUTPUT_LEN:-1}"
    MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
    LAYERWISE_SKIP_PREFILL="${LAYERWISE_SKIP_PREFILL:-0}"
    DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"
    PROFILE_STAGE="${PROFILE_STAGE:-prefill}"
    DEFAULT_MAX_TOTAL_TOKENS=$((BATCH_SIZE * (INPUT_LEN + OUTPUT_LEN) * 12))
    if (( DEFAULT_MAX_TOTAL_TOKENS < 98304 )); then
      DEFAULT_MAX_TOTAL_TOKENS=98304
    fi
    ;;
  decode)
    BATCH_SIZE="${BATCH_SIZE:-128}"
    INPUT_LEN="${INPUT_LEN:-8192}"
    OUTPUT_LEN="${OUTPUT_LEN:-2}"
    MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
    LAYERWISE_SKIP_PREFILL="${LAYERWISE_SKIP_PREFILL:-1}"
    DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
    PROFILE_STAGE="${PROFILE_STAGE:-decode}"
    DEFAULT_MAX_TOTAL_TOKENS=$((BATCH_SIZE * (INPUT_LEN + OUTPUT_LEN)))
    ;;
  smoke)
    BATCH_SIZE="${BATCH_SIZE:-1}"
    INPUT_LEN="${INPUT_LEN:-16}"
    OUTPUT_LEN="${OUTPUT_LEN:-2}"
    MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
    LAYERWISE_SKIP_PREFILL="${LAYERWISE_SKIP_PREFILL:-1}"
    DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"
    PROFILE_STAGE="${PROFILE_STAGE:-all}"
    DEFAULT_MAX_TOTAL_TOKENS=8192
    ;;
  *)
    echo "Unsupported MODE=${MODE}; use smoke, prefill, or decode" >&2
    exit 2
    ;;
esac
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-${DEFAULT_MAX_TOTAL_TOKENS}}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
RUN_NAME="${RUN_NAME:-dsv4pro_tp8mock_${MODE}}"
RESULT_FILENAME="${RESULT_FILENAME:-result.jsonl}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-${BATCH_SIZE}}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-${BATCH_SIZE}}"
BENCH_PROFILE="${BENCH_PROFILE:-0}"

if [[ -z "${CONFIG_OVERRIDES:-}" ]]; then
  CONFIG_OVERRIDES="{\"num_hidden_layers\":${NUM_LAYERS},\"num_hash_layers\":0,\"n_hash_layers\":0,\"num_attention_heads\":16,\"o_groups\":2,\"moe_intermediate_size\":384,\"vocab_size\":10000,\"max_position_embeddings\":8192,\"rope_scaling.original_max_position_embeddings\":512}"
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_HACK_OVERRIDE_TOPK_IDS_RANDOM=1
export LAYERWISE_FLASHMLA_PAD_HQ="${LAYERWISE_FLASHMLA_PAD_HQ:-1}"
export LAYERWISE_TARGET_LAYERS="${TARGET_LAYERS}"
export LAYERWISE_SKIP_PREFILL

CUDA_GRAPH_ARGS=(--cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}")
if [[ "${DISABLE_CUDA_GRAPH}" == "1" ]]; then
  CUDA_GRAPH_ARGS+=(--disable-cuda-graph)
else
  CUDA_GRAPH_ARGS+=(--cuda-graph-bs "${CUDA_GRAPH_BS}")
fi

PROFILE_ARGS=()
if [[ "${BENCH_PROFILE}" == "1" ]]; then
  PROFILE_ARGS+=(--profile --profile-activities CUDA_PROFILER --profile-stage "${PROFILE_STAGE}")
fi

python "${SCRIPT_DIR}/run_bench_skip.py" \
  --model-path "${MODEL_PATH}" \
  --config-overrides "${CONFIG_OVERRIDES}" \
  --trust-remote-code \
  --load-format dummy \
  --tp-size 1 \
  --enable-layerwise-nvtx-marker \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --max-total-tokens "${MAX_TOTAL_TOKENS}" \
  "${CUDA_GRAPH_ARGS[@]}" \
  --batch-size "${BATCH_SIZE}" \
  --input-len "${INPUT_LEN}" \
  --output-len "${OUTPUT_LEN}" \
  --run-name "${RUN_NAME}" \
  --result-filename "${RESULT_FILENAME}" \
  "${PROFILE_ARGS[@]}"
