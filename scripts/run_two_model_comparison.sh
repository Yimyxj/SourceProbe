#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LEAKAGE_FREE_ROOT="${LEAKAGE_FREE_ROOT:-ai_source_attribution/outputs/temperature_sweep/leakage_free_seed42_fresh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$LEAKAGE_FREE_ROOT/outputs}"
DATA_ROOT="${DATA_ROOT:-ai_source_attribution/external/Profiler/Dataset}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-../../LLMs/QWEN/Qwen3-8B}"
LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-../../LLMs/LLAMA/llama3.1-8b}"
QWEN25_MODEL_PATH="${QWEN25_MODEL_PATH:-../../LLMs/QWEN/Qwen2.5-7B}"
LLAMA2_MODEL_PATH="${LLAMA2_MODEL_PATH:-../../LLMs/LLAMA/LLaMA-2-7B}"
MULTI_MODEL_PATHS="${MULTI_MODEL_PATHS:-$QWEN_MODEL_PATH,$LLAMA_MODEL_PATH,$QWEN25_MODEL_PATH,$LLAMA2_MODEL_PATH}"
MULTI_MODEL_TAGS="${MULTI_MODEL_TAGS:-qwen3,llama31,qwen25,llama2}"
DOMAINS="${DOMAINS:-Arxiv Creative Essay Yelp}"
TARGET_LABELS="${TARGET_LABELS:-gpt-3.5-turbo gpt-4-turbo-preview gemini-1.0-pro claude-3-sonnet claude-3-opus human}"
SEEDS="${SEEDS:-42,43,44}"
TRAIN_SIZES="${TRAIN_SIZES:-5 10 30 50 100 200 300}"

LLAMA_GPU="${LLAMA_GPU:-1}"
DUAL_GPUS="${DUAL_GPUS:-0,1}"
BASELINE_GPU="${BASELINE_GPU:-0}"

RUN_EXPORT_DATA="${RUN_EXPORT_DATA:-1}"
RUN_LLAMA_SINGLE="${RUN_LLAMA_SINGLE:-1}"
RUN_DUAL_OFFICIAL_FEATURES="${RUN_DUAL_OFFICIAL_FEATURES:-1}"
RUN_DUAL_OFFICIAL_BASELINES="${RUN_DUAL_OFFICIAL_BASELINES:-1}"
RUN_DUAL_PROFILER="${RUN_DUAL_PROFILER:-1}"
RUN_FEWSHOT="${RUN_FEWSHOT:-1}"

DUAL_FEATURE_ROOT="${DUAL_FEATURE_ROOT:-$LEAKAGE_FREE_ROOT/outputs/official_baselines/features_4lm}"
DUAL_SUFFIX="${DUAL_SUFFIX:-_4lm_official}"
DUAL_PROFILER_METHOD="${DUAL_PROFILER_METHOD:-profiler_rf_ova_4lm}"
OFFICIAL_REPEAT_EACH_MODEL="${OFFICIAL_REPEAT_EACH_MODEL:-1}"
OFFICIAL_EXPECTED_CHANNELS="${OFFICIAL_EXPECTED_CHANNELS:-4}"

if [ "$RUN_EXPORT_DATA" = "1" ]; then
  echo "[STEP] Export fixed official-baseline data from the leakage-free split."
  LEAKAGE_FREE_ROOT="$LEAKAGE_FREE_ROOT" \
  DATA_ROOT="$DATA_ROOT" \
  DOMAINS="$DOMAINS" \
  SEEDS="$SEEDS" \
  bash ai_source_attribution/scripts/export_official_baseline_data.sh
fi

if [ "$RUN_LLAMA_SINGLE" = "1" ]; then
  echo "[STEP] Run Llama-only hidden_probe and PROFILER on the same fixed split."
  CUDA_VISIBLE_DEVICES="$LLAMA_GPU" \
  DATA_ROOT="$DATA_ROOT" \
  PROFILER_SPLIT_ROOT="$LEAKAGE_FREE_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  MODEL_PATH="$LLAMA_MODEL_PATH" \
  DOMAINS="$DOMAINS" \
  SEEDS="$SEEDS" \
  METHOD=hidden_probe \
  METHOD_OUTPUT_NAME=hidden_probe_llama \
  bash ai_source_attribution/scripts/run_multiclass_profiler.sh

  CUDA_VISIBLE_DEVICES="$LLAMA_GPU" \
  DATA_ROOT="$DATA_ROOT" \
  PROFILER_SPLIT_ROOT="$LEAKAGE_FREE_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  MODEL_PATH="$LLAMA_MODEL_PATH" \
  DOMAINS="$DOMAINS" \
  SEEDS="$SEEDS" \
  METHOD=profiler_rf_ova \
  METHOD_OUTPUT_NAME=profiler_rf_ova_llama \
  bash ai_source_attribution/scripts/run_multiclass_profiler.sh

  CUDA_VISIBLE_DEVICES="$LLAMA_GPU" \
  DATA_ROOT="$DATA_ROOT" \
  PROFILER_SPLIT_ROOT="$LEAKAGE_FREE_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  PRECOMPUTED_OUTPUT_ROOT="$OUTPUT_ROOT" \
  MODEL_PATH="$LLAMA_MODEL_PATH" \
  DOMAINS="$DOMAINS" \
  TARGET_LABELS="$TARGET_LABELS" \
  SEEDS="$SEEDS" \
  METHODS="hidden_probe profiler_rf_ova" \
  METHOD_OUTPUT_SUFFIX="_llama" \
  bash ai_source_attribution/scripts/run_one_vs_all_profiler.sh
fi

if [ "$RUN_DUAL_OFFICIAL_FEATURES" = "1" ]; then
  echo "[STEP] Extract multi-model official PPL features for Sniffer and SeqXGPT."
  CUDA_VISIBLE_DEVICES="$DUAL_GPUS" \
  LEAKAGE_FREE_ROOT="$LEAKAGE_FREE_ROOT" \
  OFFICIAL_FEATURE_ROOT="$DUAL_FEATURE_ROOT" \
  MODEL_PATHS="$MULTI_MODEL_PATHS" \
  FEATURE_CHANNELS=4 \
  REPEAT_EACH_MODEL="$OFFICIAL_REPEAT_EACH_MODEL" \
  EXPECTED_CHANNELS="$OFFICIAL_EXPECTED_CHANNELS" \
  MAX_LENGTH="${OFFICIAL_MAX_LENGTH:-512}" \
  DOMAINS="$DOMAINS" \
  bash ai_source_attribution/scripts/extract_official_baseline_features.sh
fi

if [ "$RUN_DUAL_OFFICIAL_BASELINES" = "1" ]; then
  echo "[STEP] Train/evaluate strict official Sniffer and SeqXGPT with multi-model features."
  CUDA_VISIBLE_DEVICES="$BASELINE_GPU" \
  LEAKAGE_FREE_ROOT="$LEAKAGE_FREE_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  OFFICIAL_FEATURE_ROOT="$DUAL_FEATURE_ROOT" \
  DOMAINS="$DOMAINS" \
  SEEDS="$SEEDS" \
  METHODS="sniffer seqxgpt" \
  METHOD_OUTPUT_SUFFIX="$DUAL_SUFFIX" \
  bash ai_source_attribution/scripts/run_official_multiclass_profiler.sh

  CUDA_VISIBLE_DEVICES="$BASELINE_GPU" \
  LEAKAGE_FREE_ROOT="$LEAKAGE_FREE_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  OFFICIAL_FEATURE_ROOT="$DUAL_FEATURE_ROOT" \
  DOMAINS="$DOMAINS" \
  TARGET_LABELS="$TARGET_LABELS" \
  SEEDS="$SEEDS" \
  METHODS="sniffer seqxgpt" \
  METHOD_OUTPUT_SUFFIX="$DUAL_SUFFIX" \
  bash ai_source_attribution/scripts/run_official_one_vs_all_profiler.sh
fi

if [ "$RUN_DUAL_PROFILER" = "1" ]; then
  echo "[STEP] Run multi-model PROFILER by concatenating local PROFILER features."
  CUDA_VISIBLE_DEVICES="$DUAL_GPUS" \
  LEAKAGE_FREE_ROOT="$LEAKAGE_FREE_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  DATA_ROOT="$DATA_ROOT" \
  MODEL_PATHS="$MULTI_MODEL_PATHS" \
  MODEL_TAGS="$MULTI_MODEL_TAGS" \
  FEATURE_CACHE_ROOT="$DUAL_FEATURE_ROOT" \
  RESULT_METHOD="$DUAL_PROFILER_METHOD" \
  DOMAINS="$DOMAINS" \
  SEEDS="$SEEDS" \
  bash ai_source_attribution/scripts/run_dual_profiler_multiclass.sh

  CUDA_VISIBLE_DEVICES="$DUAL_GPUS" \
  LEAKAGE_FREE_ROOT="$LEAKAGE_FREE_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  DATA_ROOT="$DATA_ROOT" \
  MODEL_PATHS="$MULTI_MODEL_PATHS" \
  MODEL_TAGS="$MULTI_MODEL_TAGS" \
  FEATURE_CACHE_ROOT="$DUAL_FEATURE_ROOT" \
  RESULT_METHOD="$DUAL_PROFILER_METHOD" \
  DOMAINS="$DOMAINS" \
  TARGET_LABELS="$TARGET_LABELS" \
  SEEDS="$SEEDS" \
  bash ai_source_attribution/scripts/run_dual_profiler_one_vs_all.sh
fi

if [ "$RUN_FEWSHOT" = "1" ]; then
  echo "[STEP] Run few-shot comparisons using cached full-split features."
  CUDA_VISIBLE_DEVICES="$LLAMA_GPU" \
  LEAKAGE_FREE_ROOT="$LEAKAGE_FREE_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT/fewshot" \
  DATA_ROOT="$DATA_ROOT" \
  PROFILER_SPLIT_ROOT="$LEAKAGE_FREE_ROOT" \
  PRECOMPUTED_OUTPUT_ROOT="$OUTPUT_ROOT" \
  MODEL_PATH="$QWEN_MODEL_PATH" \
  DOMAINS="$DOMAINS" \
  SEEDS="$SEEDS" \
  TRAIN_SIZES="$TRAIN_SIZES" \
  METHODS="hidden_probe profiler_rf_ova" \
  bash ai_source_attribution/scripts/run_fewshot_profiler.sh

  CUDA_VISIBLE_DEVICES="$LLAMA_GPU" \
  LEAKAGE_FREE_ROOT="$LEAKAGE_FREE_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT/fewshot" \
  DATA_ROOT="$DATA_ROOT" \
  PROFILER_SPLIT_ROOT="$LEAKAGE_FREE_ROOT" \
  PRECOMPUTED_OUTPUT_ROOT="$OUTPUT_ROOT" \
  MODEL_PATH="$LLAMA_MODEL_PATH" \
  DOMAINS="$DOMAINS" \
  SEEDS="$SEEDS" \
  TRAIN_SIZES="$TRAIN_SIZES" \
  METHODS="hidden_probe profiler_rf_ova" \
  METHOD_OUTPUT_SUFFIX="_llama" \
  bash ai_source_attribution/scripts/run_fewshot_profiler.sh

  CUDA_VISIBLE_DEVICES="$BASELINE_GPU" \
  LEAKAGE_FREE_ROOT="$LEAKAGE_FREE_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT/fewshot" \
  OFFICIAL_FEATURE_ROOT="$DUAL_FEATURE_ROOT" \
  DOMAINS="$DOMAINS" \
  SEEDS="$SEEDS" \
  TRAIN_SIZES="$TRAIN_SIZES" \
  METHODS="sniffer seqxgpt" \
  METHOD_OUTPUT_SUFFIX="$DUAL_SUFFIX" \
  bash ai_source_attribution/scripts/run_official_fewshot_profiler.sh

  CUDA_VISIBLE_DEVICES="$DUAL_GPUS" \
  LEAKAGE_FREE_ROOT="$LEAKAGE_FREE_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT/fewshot" \
  DATA_ROOT="$DATA_ROOT" \
  MODEL_PATHS="$MULTI_MODEL_PATHS" \
  MODEL_TAGS="$MULTI_MODEL_TAGS" \
  FEATURE_CACHE_ROOT="$DUAL_FEATURE_ROOT" \
  RESULT_METHOD="$DUAL_PROFILER_METHOD" \
  DOMAINS="$DOMAINS" \
  SEEDS="$SEEDS" \
  TRAIN_SIZES="$TRAIN_SIZES" \
  bash ai_source_attribution/scripts/run_dual_profiler_fewshot.sh
fi

echo "[DONE] two-model comparison pipeline finished."
