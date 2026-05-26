# SourceProbe: Qwen3 Hidden-Probe Experiments

This document describes how to run **SourceProbe**, i.e. the Qwen3-based `hidden_probe` method, in a standalone folder without relying on the `ai_source_attribution` package name.

The commands below assume that the uploaded folder contains the required Python files directly in the same directory:

```text
sourceprobe/
  README.md
  run_experiment.py
  encoder.py
  data.py
  metrics.py
```


All commands in this README use:

```bash
python run_experiment.py
```


## 1. Method Overview

SourceProbe corresponds to:

```text
method = hidden_probe
surrogate LLM = Qwen3-8B
```

For each input text, SourceProbe extracts a hidden representation from Qwen3 and trains a lightweight linear probe for source attribution.

The default representation is:

```text
layer_pos = last
pooling = last_token
max_length = 512
```

The probe classifier is:

```python
StandardScaler()
LogisticRegression(max_iter=2000, class_weight="balanced", multi_class="auto")
```

## 2. Required Inputs

Set the paths according to your server:

```bash
export DATA_ROOT=/path/to/Profiler/Dataset
export LEAK_ROOT=/path/to/leakage_free_seed42_fresh
export OUTPUT_ROOT="$LEAK_ROOT/outputs"
export QWEN_MODEL_PATH=/path/to/Qwen3-8B
```

Expected split files:

```text
$LEAK_ROOT/splits/Arxiv_seed42.json
$LEAK_ROOT/splits/Creative_seed42.json
$LEAK_ROOT/splits/Essay_seed42.json
$LEAK_ROOT/splits/Yelp_seed42.json
```

## 3. Install Dependencies

Typical dependencies are:

```bash
pip install torch transformers scikit-learn numpy joblib tqdm
```

Use the PyTorch build that matches your CUDA version.

## 4. Run Multiclass SourceProbe

The multiclass setting trains one closed-set classifier over all source labels.

```bash
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/results/multiclass"

for domain in Arxiv Creative Essay Yelp; do
  out_dir="$OUTPUT_ROOT/results/multiclass/hidden_probe/${domain}_include_human_true"
  log_file="$OUTPUT_ROOT/logs/multiclass_hidden_probe_${domain}_include_human_true.log"

  CUDA_VISIBLE_DEVICES=0 \
  python -u run_experiment.py \
    --dataset profiler \
    --data_root "$DATA_ROOT" \
    --train_split train \
    --test_split test \
    --profiler_split_root "$LEAK_ROOT" \
    --profiler_split_seed 42 \
    --domain "$domain" \
    --include_human \
    --model_path "$QWEN_MODEL_PATH" \
    --method hidden_probe \
    --batch_size 1 \
    --max_length 512 \
    --seeds 42,43,44 \
    --output_dir "$out_dir" \
    > "$log_file" 2>&1
done
```

Outputs are saved to:

```text
$OUTPUT_ROOT/results/multiclass/hidden_probe/<Domain>_include_human_true/
```

For example:

```text
$OUTPUT_ROOT/results/multiclass/hidden_probe/Arxiv_include_human_true/
```

## 5. Run One-vs-All SourceProbe

The one-vs-all setting trains a binary classifier for each target source label. The target label is treated as positive, and all other labels are merged into `other`.

This stage can reuse the hidden features already saved by the multiclass run through `--precomputed_feature_dir`.

```bash
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/results/one_vs_all"

TARGET_LABELS=(
  "gpt-3.5-turbo"
  "gpt-4-turbo-preview"
  "gemini-1.0-pro"
  "claude-3-sonnet"
  "claude-3-opus"
  "human"
)

for domain in Arxiv Creative Essay Yelp; do
  precomputed_dir="$OUTPUT_ROOT/results/multiclass/hidden_probe/${domain}_include_human_true"

  for target in "${TARGET_LABELS[@]}"; do
    out_dir="$OUTPUT_ROOT/results/one_vs_all/hidden_probe/${domain}/${target}"
    log_file="$OUTPUT_ROOT/logs/one_vs_all_hidden_probe_${domain}_${target}.log"

    CUDA_VISIBLE_DEVICES=0 \
    python -u run_experiment.py \
      --dataset profiler \
      --data_root "$DATA_ROOT" \
      --train_split train \
      --test_split test \
      --profiler_split_root "$LEAK_ROOT" \
      --profiler_split_seed 42 \
      --domain "$domain" \
      --include_human \
      --one_vs_all_label "$target" \
      --model_path "$QWEN_MODEL_PATH" \
      --method hidden_probe \
      --batch_size 1 \
      --max_length 512 \
      --seeds 42,43,44 \
      --precomputed_feature_dir "$precomputed_dir" \
      --output_dir "$out_dir" \
      > "$log_file" 2>&1
  done
done
```

Outputs are saved to:

```text
$OUTPUT_ROOT/results/one_vs_all/hidden_probe/<Domain>/<target_label>/
```

For example:

```text
$OUTPUT_ROOT/results/one_vs_all/hidden_probe/Arxiv/gpt-3.5-turbo/
```

## 6. Main Output Files

Each experiment directory contains:

```text
summary.json
label_mapping.json
seed_42/
seed_43/
seed_44/
```

Each seed directory contains:

```text
train_hidden_features.npy
test_hidden_features.npy
classifier.joblib
test_metrics.json
test_confusion_matrix.npy
test_confusion_matrix.csv
```

Important files:

- `summary.json`: mean and standard deviation over seeds.
- `test_metrics.json`: metrics for one seed.
- `train_hidden_features.npy`: Qwen3 hidden features for the training split.
- `test_hidden_features.npy`: Qwen3 hidden features for the test split.
- `classifier.joblib`: trained SourceProbe classifier.

## 7. Metrics

The evaluation reports:

```text
accuracy
macro-F1
weighted-F1
AUROC
confusion matrix
classification report
```

For multiclass experiments, AUROC is computed with one-vs-rest macro averaging.

For one-vs-all experiments, AUROC is computed for the target label against `other`.

## 8. Logs

Logs are saved under:

```text
$OUTPUT_ROOT/logs/
```

Examples:

```bash
tail -f "$OUTPUT_ROOT/logs/multiclass_hidden_probe_Arxiv_include_human_true.log"
tail -f "$OUTPUT_ROOT/logs/one_vs_all_hidden_probe_Arxiv_gpt-3.5-turbo.log"
```

python run_experiment.py
```
