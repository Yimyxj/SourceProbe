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
METHODS="${METHODS:-hidden_probe profiler_rf_ova}"
METHOD_OUTPUT_SUFFIX="${METHOD_OUTPUT_SUFFIX:-}"
PRECOMPUTED_METHOD_SUFFIX="${PRECOMPUTED_METHOD_SUFFIX:-$METHOD_OUTPUT_SUFFIX}"
TARGET_LABELS="${TARGET_LABELS:-gpt-3.5-turbo gpt-4-turbo-preview gemini-1.0-pro claude-3-sonnet claude-3-opus human}"
K="${K:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_TRAIN_PER_LABEL="${MAX_TRAIN_PER_LABEL:-}"
MAX_TEST_PER_LABEL="${MAX_TEST_PER_LABEL:-}"
SKIP_DONE="${SKIP_DONE:-1}"

LIMIT_ARGS=()
if [ -n "$MAX_TRAIN_PER_LABEL" ]; then LIMIT_ARGS+=(--max_train_per_label "$MAX_TRAIN_PER_LABEL"); fi
if [ -n "$MAX_TEST_PER_LABEL" ]; then LIMIT_ARGS+=(--max_test_per_label "$MAX_TEST_PER_LABEL"); fi
TEST_ROOT_ARGS=()
if [ -n "$TEST_DATA_ROOT" ]; then TEST_ROOT_ARGS+=(--test_data_root "$TEST_DATA_ROOT"); fi
SPLIT_ARGS=(--train_split none --test_split none)
if [ -n "$PROFILER_SPLIT_ROOT" ]; then
  SPLIT_ARGS=(--train_split train --test_split test --profiler_split_root "$PROFILER_SPLIT_ROOT" --profiler_split_seed "$PROFILER_SPLIT_SEED")
fi

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/results/one_vs_all"

for method in $METHODS; do
  result_method="${method}${METHOD_OUTPUT_SUFFIX}"
  precomputed_method="${method}${PRECOMPUTED_METHOD_SUFFIX}"
  for domain in $DOMAINS; do
    for target in $TARGET_LABELS; do
      safe_target="${target//\//_}"
      out_dir="$OUTPUT_ROOT/results/one_vs_all/${result_method}/${domain}/${safe_target}"
      log_file="$OUTPUT_ROOT/logs/one_vs_all_${result_method}_${domain}_${safe_target}.log"
      if [ "$SKIP_DONE" = "1" ] && [ -f "$out_dir/summary.json" ]; then
        echo "[SKIP] $out_dir already has summary.json"
        continue
      fi
      echo "[RUN] one_vs_all method=$method domain=$domain target=$target"
      PRECOMPUTED_ARGS=()
      if [ -n "$PRECOMPUTED_OUTPUT_ROOT" ]; then
        precomputed_dir="$PRECOMPUTED_OUTPUT_ROOT/results/multiclass/${precomputed_method}/${domain}_include_human_true"
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
        --include_human \
        --one_vs_all_label "$target" \
        --model_path "$MODEL_PATH" \
        --method "$method" \
        --k "$K" \
        --batch_size "$BATCH_SIZE" \
        --max_length "$MAX_LENGTH" \
        "${PRECOMPUTED_ARGS[@]}" \
        "${LIMIT_ARGS[@]}" \
        --seeds "$SEEDS" \
        --output_dir "$out_dir" > "$log_file" 2>&1
      echo "[DONE] one_vs_all method=$method domain=$domain target=$target"
    done
  done
done
