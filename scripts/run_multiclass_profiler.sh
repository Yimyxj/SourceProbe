#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATA_ROOT="${DATA_ROOT:-ai_source_attribution/external/Profiler/Dataset}"
TEST_DATA_ROOT="${TEST_DATA_ROOT:-}"
PROFILER_SPLIT_ROOT="${PROFILER_SPLIT_ROOT:-}"
PROFILER_SPLIT_SEED="${PROFILER_SPLIT_SEED:-42}"
MODEL_PATH="${MODEL_PATH:-../../LLMs/QWEN/Qwen3-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ai_source_attribution/outputs}"
PRECOMPUTED_OUTPUT_ROOT="${PRECOMPUTED_OUTPUT_ROOT:-}"
SEEDS="${SEEDS:-42,43,44}"
DOMAINS="${DOMAINS:-Arxiv Creative Essay Yelp GCJ Code}"
K="${K:-8}"
METHOD="${METHOD:-energy}"
METHOD_OUTPUT_NAME="${METHOD_OUTPUT_NAME:-$METHOD}"
PRECOMPUTED_METHOD="${PRECOMPUTED_METHOD:-$METHOD_OUTPUT_NAME}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_TRAIN_PER_LABEL="${MAX_TRAIN_PER_LABEL:-}"
MAX_TEST_PER_LABEL="${MAX_TEST_PER_LABEL:-}"
SKIP_DONE="${SKIP_DONE:-1}"
RUN_AI_ONLY="${RUN_AI_ONLY:-0}"
LIMIT_ARGS=()
if [ -n "$MAX_TRAIN_PER_LABEL" ]; then LIMIT_ARGS+=(--max_train_per_label "$MAX_TRAIN_PER_LABEL"); fi
if [ -n "$MAX_TEST_PER_LABEL" ]; then LIMIT_ARGS+=(--max_test_per_label "$MAX_TEST_PER_LABEL"); fi
TEST_ROOT_ARGS=()
if [ -n "$TEST_DATA_ROOT" ]; then TEST_ROOT_ARGS+=(--test_data_root "$TEST_DATA_ROOT"); fi
SPLIT_ARGS=(--train_split none --test_split none)
if [ -n "$PROFILER_SPLIT_ROOT" ]; then
  SPLIT_ARGS=(--train_split train --test_split test --profiler_split_root "$PROFILER_SPLIT_ROOT" --profiler_split_seed "$PROFILER_SPLIT_SEED")
fi

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/results/multiclass"

INCLUDE_HUMAN_VALUES="true"
if [ "$RUN_AI_ONLY" = "1" ]; then
  INCLUDE_HUMAN_VALUES="true false"
fi

for include_human in $INCLUDE_HUMAN_VALUES; do
  for domain in $DOMAINS; do
    out_dir="$OUTPUT_ROOT/results/multiclass/${METHOD_OUTPUT_NAME}/${domain}_include_human_${include_human}"
    log_file="$OUTPUT_ROOT/logs/multiclass_${METHOD_OUTPUT_NAME}_${domain}_include_human_${include_human}.log"
    if [ "$SKIP_DONE" = "1" ] && [ -f "$out_dir/summary.json" ]; then
      echo "[SKIP] $out_dir already has summary.json"
      continue
    fi
    echo "[RUN] multiclass method=$METHOD domain=$domain include_human=$include_human"
    echo "[LOG] $log_file"
    human_flag="--include_human"
    if [ "$include_human" = "false" ]; then
      human_flag="--no-include_human"
    fi
    PRECOMPUTED_ARGS=()
    if [ -n "$PRECOMPUTED_OUTPUT_ROOT" ]; then
      precomputed_dir="$PRECOMPUTED_OUTPUT_ROOT/results/multiclass/${PRECOMPUTED_METHOD}/${domain}_include_human_${include_human}"
      if [ -d "$precomputed_dir" ]; then
        PRECOMPUTED_ARGS+=(--precomputed_feature_dir "$precomputed_dir")
      fi
    fi
    python -u -m ai_source_attribution.run_experiment \
      --dataset profiler \
      --data_root "$DATA_ROOT" \
      "${TEST_ROOT_ARGS[@]}" \
      "${SPLIT_ARGS[@]}" \
      --domain "$domain" \
      $human_flag \
      --model_path "$MODEL_PATH" \
      --method "$METHOD" \
      --k "$K" \
      --batch_size "$BATCH_SIZE" \
      --max_length "$MAX_LENGTH" \
      "${PRECOMPUTED_ARGS[@]}" \
      "${LIMIT_ARGS[@]}" \
      --seeds "$SEEDS" \
      --output_dir "$out_dir" > "$log_file" 2>&1
    echo "[DONE] multiclass method=$METHOD domain=$domain include_human=$include_human"
  done
done
