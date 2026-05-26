# Black-box AI-generated Text Source Attribution

This directory contains a clean, standalone implementation for black-box
AI-generated text source attribution. It is built on top of the representation
and energy ideas from CoSur, but it removes all cognitive editing and all
IPP-style yes/no prompting.

The original repository files are not modified. All new code lives under:

```text
ai_source_attribution/
```

## Task

Given a public surrogate model `F`, such as Qwen, Llama, DeepSeek, GPT-2, or
RoBERTa, the goal is to predict the origin/source label of an input text.

The surrogate model is used only as a forensic encoder:

- It extracts hidden representations or PROFILER-style context-loss features.
- It is never asked to generate a yes/no answer.
- Its weights are never modified.
- Candidate origin LLM parameters, logits, and hidden states are never accessed.

Training uses a small number of labeled reference samples per source. Testing
uses only raw text and outputs a source prediction.

## Implemented Methods

The main entry point is:

```bash
python -m ai_source_attribution.run_experiment
```

Supported methods:

- `hidden_probe`: multiclass Logistic Regression on surrogate hidden features.
- `energy`: direct prediction by maximum projection energy over source
  subspaces.
- `linear_probe_on_energy`: an optional hybrid ablation that trains multiclass
  Logistic Regression on source-energy features. This is not the main method
  from the original CoSur-style code.
- `profiler_baseline`: PROFILER-style features with multiclass Logistic
  Regression.
- `profiler_rf_ova`: PROFILER-style features with the official one-vs-all
  RandomForest classifier style.
- `origin_tracing`: logit/probability distribution features in the style of
  origin tracing and detecting methods.
- `sniffer`: reserved for strict official Sniffer features/runs. It is no
  longer mapped to the `origin_tracing` approximation.
- `seqxgpt`: reserved for strict official SeqXGPT runs with the official
  convolution/self-attention network. It is no longer mapped to the resized
  log-probability + Logistic Regression approximation.

## Code Layout

```text
ai_source_attribution/
  data.py                  Unified CoSur/PROFILER/generic data loader
  encoder.py               Surrogate hidden representation extraction
  energy.py                SVD/PCA source spaces and projection energy
  profiler_baseline.py     PROFILER-style context-aware features
  metrics.py               Metrics and output serialization
  run_experiment.py        Unified experiment CLI
  scripts/                 Shell scripts for experiment suites
  external/Profiler/       Official PROFILER repository
  NOTE.md                  Chinese implementation notes
```

## Data Interfaces

### CoSur-style JSONL

Use `--dataset cosur` with fields such as:

```text
human_answers,chatgpt_answers,qwen_answers,llama_answers,deepseek_answers
```

Example:

```bash
python -m ai_source_attribution.run_experiment \
  --dataset cosur \
  --data_root data \
  --train_split train \
  --test_split test \
  --source_fields human_answers,chatgpt_answers,qwen_answers,llama_answers,deepseek_answers \
  --model_path ../../LLMs/QWEN/Qwen3-8B \
  --method linear_probe_on_energy \
  --output_dir ai_source_attribution/outputs/results/cosur_demo
```

### PROFILER Official Dataset

The official PROFILER repository is cloned into:

```text
ai_source_attribution/external/Profiler
```

Example with a reproducible stratified split:

```bash
python -m ai_source_attribution.run_experiment \
  --dataset profiler \
  --data_root ai_source_attribution/external/Profiler/Dataset \
  --train_split none \
  --test_split none \
  --domain Arxiv \
  --model_path ../../LLMs/QWEN/Qwen3-8B \
  --method linear_probe_on_energy \
  --output_dir ai_source_attribution/outputs/results/profiler_demo
```

Example for normal-train / paraphrased-test OOD evaluation:

```bash
python -m ai_source_attribution.run_experiment \
  --dataset profiler \
  --data_root ai_source_attribution/external/Profiler/Dataset \
  --test_data_root ai_source_attribution/external/Profiler/Paraphrased_Dataset \
  --train_split none \
  --test_split none \
  --domain Arxiv \
  --no-include_human \
  --model_path ../../LLMs/QWEN/Qwen3-8B \
  --method linear_probe_on_energy \
  --output_dir ai_source_attribution/outputs/results/ood_demo
```

## Metrics and Outputs

Each experiment saves:

- Accuracy
- Macro and weighted F1
- AUROC
- Per-class AUROC when available
- Classification report
- Confusion matrix as `.npy` and `.csv`
- Label mapping
- Hidden or PROFILER features as `.npy`
- Energy features as `.npy` for energy-based methods
- Mean and standard deviation across multiple seeds

Typical output files:

```text
label_mapping.json
summary.json
seed_42/test_metrics.json
seed_42/test_confusion_matrix.npy
seed_42/test_confusion_matrix.csv
seed_42/train_hidden_or_profiler_features.npy
seed_42/test_hidden_or_profiler_features.npy
seed_42/train_energy_features.npy
seed_42/test_energy_features.npy
```

## Experiment Scripts

All scripts are in:

```text
ai_source_attribution/scripts/
```

They use `set -e`, support environment variables, and write stdout/stderr to
`ai_source_attribution/outputs/logs/`.

Supported environment variables:

```bash
DATA_ROOT
MODEL_PATH
OUTPUT_ROOT
CUDA_VISIBLE_DEVICES
SEEDS
```

Scripts:

- `run_binary_profiler.sh`: Human vs AI binary classification on PROFILER.
- `run_multiclass_profiler.sh`: multiclass attribution with and without human.
- `run_compare_profiler_methods.sh`: run `hidden_probe`, `energy`, and the
  PROFILER baseline on PROFILER for direct comparison.
- `run_temperature_sweep.sh`: generate gpt-3.5-turbo rewrites from PROFILER
  human texts over generation length and temperature sweeps, evaluate
  `hidden_probe`, `profiler_rf_ova`, `origin_tracing`, and `seqxgpt`, and
  export an `.xlsx` summary plus ROC points.
- `run_fewshot_profiler.sh`: compare `hidden_probe` and PROFILER when each
  source/class contributes only `5, 10, 30, 50, 100, 200, 300` training
  samples. Strict Sniffer/SeqXGPT runs use their official repositories under
  `external/` and the exported fixed-split official-baseline data.
- `run_ood_paraphrase.sh`: normal train, paraphrased test.
- `run_k_ablation.sh`: sweep `k = 1, 2, 4, 8, 16, 32, 64`.
- `run_pooling_ablation.sh`: sweep `last_token`, `mean_pooling`,
  `last_k_mean`.
- `run_surrogate_ablation.sh`: sweep surrogate model paths.
- `run_all.sh`: run the main experiment scripts sequentially.
- `run_binary_cosur.sh`: Human vs AI binary classification on the local CoSur data.
- `run_multiclass_cosur.sh`: multiclass attribution on the local CoSur data.
- `run_k_ablation_cosur.sh`: k ablation on the local CoSur data.
- `run_pooling_ablation_cosur.sh`: pooling ablation on the local CoSur data.
- `run_surrogate_ablation_cosur.sh`: surrogate ablation on the local CoSur data.
- `run_all_cosur.sh`: run the CoSur-data scripts sequentially.

Example:

```bash
DATA_ROOT=ai_source_attribution/external/Profiler/Dataset \
MODEL_PATH=../../LLMs/QWEN/Qwen3-8B \
OUTPUT_ROOT=ai_source_attribution/outputs \
SEEDS=42,43,44 \
bash ai_source_attribution/scripts/run_binary_profiler.sh
```

### Temperature Sweep

The end-to-end script uses Arxiv `abs` and Creative `essay` human fields by
default and prompts `gpt-3.5-turbo` with:

```text
Enhance word choices to make the sentence sound more like a human.
 text:{}
```

Example:

```bash
export OPENAI_API_KEY=...
MODEL_PATH=../../LLMs/QWEN/Qwen3-8B \
OUTPUT_ROOT=ai_source_attribution/outputs \
DOMAINS="Arxiv Creative" \
TEMPERATURES=0.1,0.3,0.5,0.7,1.0 \
MAX_COMPLETION_TOKENS=128,256,512 \
N_SAMPLES=200 \
bash ai_source_attribution/scripts/run_temperature_sweep.sh
```

Useful switches:

```bash
RUN_GENERATION=0   # reuse existing temperature JSONL files
RUN_TRAINING=0     # reuse saved classifiers under outputs/results
METHODS="hidden_probe profiler_rf_ova origin_tracing"
```

### Few-Shot Training Size Sweep

Run the low-resource comparison with:

```bash
DATA_ROOT=ai_source_attribution/external/Profiler/Dataset \
MODEL_PATH=../../LLMs/QWEN/Qwen3-8B \
OUTPUT_ROOT=ai_source_attribution/outputs/fewshot \
DOMAINS="Arxiv Creative Essay Yelp" \
METHODS="hidden_probe profiler_rf_ova" \
TRAIN_SIZES="5 10 30 50 100 200 300" \
SEEDS=42,43,44 \
bash ai_source_attribution/scripts/run_fewshot_profiler.sh
```

To reuse features from an existing full run, point `PRECOMPUTED_OUTPUT_ROOT` to
the directory that contains `results/multiclass/...`. If the existing run used
an explicit PROFILER split, pass the same split root as well:

```bash
PRECOMPUTED_OUTPUT_ROOT=ai_source_attribution/outputs/temperature_sweep/leakage_free_seed42_fresh/outputs \
PROFILER_SPLIT_ROOT=ai_source_attribution/outputs/temperature_sweep/leakage_free_seed42_fresh \
MODEL_PATH=../../LLMs/QWEN/Qwen3-8B \
bash ai_source_attribution/scripts/run_fewshot_profiler.sh
```

The reuse path supports both full matrices such as `all_hidden_features.npy`
and older split matrices such as `seed_42/train_logit_features.npy` plus
`seed_42/test_logit_features.npy`. It also recognizes legacy
`all_hidden_or_profiler_features.npy` files. The few-shot script defaults to the
leakage-free `temperature_sweep/leakage_free_seed42_fresh` split and keeps the
test set fixed while only subsampling the training split. It also passes
`--require_precomputed_features` by default, so missing reusable features cause
an error instead of silently re-extracting model features.

The script writes per-run files under `outputs/fewshot/multiclass/` and a
compact CSV summary to:

```text
ai_source_attribution/outputs/fewshot/fewshot_summary.csv
```

Here `TRAIN_SIZES` is per class/source. For example, `TRAIN_SIZES=100` means
100 training samples for each label, not 100 total samples. Each seed directory
also writes `train_selection.json`, which records the selected row indices
relative to the fixed full training split and the actual per-label counts.

To prepare the same fixed train/test split for strict official Sniffer and
SeqXGPT reproduction, export raw JSONL files with:

```bash
bash ai_source_attribution/scripts/export_official_baseline_data.sh
```

This writes full and few-shot `train.jsonl` / `test.jsonl` files under
`ai_source_attribution/outputs/official_baselines/data/`. These files keep the
leakage-free test split fixed and subsample only the training split.

Main outputs:

```text
outputs/temperature_sweep/data/<Domain>/<Domain>_gpt-3.5-turbo_temp_<T>_len_<L>.jsonl
outputs/temperature_sweep/eval/temperature_metrics_by_seed.csv
outputs/temperature_sweep/eval/temperature_metrics_summary.csv
outputs/temperature_sweep/eval/<method>/<domain>/temp_<T>/len_<L>/seed_<S>/sample_scores.csv
outputs/temperature_sweep/eval/<method>/<domain>/temp_<T>/len_<L>/seed_<S>/roc_curve.csv
outputs/temperature_sweep/temperature_sweep_results.xlsx
```

## Reproducibility

Use comma-separated seeds:

```bash
--seeds 42,43,44
```

Random splits are stratified and reproducible. The final `summary.json` reports
mean and standard deviation across seeds.

## Important Constraints

This implementation intentionally avoids:

- IPP prompts
- yes/no answer generation
- string matching for correctness
- hidden-state editing
- logits editing
- surrogate weight updates
- access to origin LLM internals

The surrogate model is only used as a forensic encoder.

