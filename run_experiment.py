import argparse
import hashlib
import json
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ai_source_attribution.data import load_dataset, stratified_split
from ai_source_attribution.encoder import extract_hidden_representations, load_surrogate
from ai_source_attribution.energy import build_source_spaces, energy_matrix, predict_by_max_energy
from ai_source_attribution.logit_baselines import extract_logit_features
from ai_source_attribution.metrics import aggregate_runs, ensure_dir, evaluate_and_save, save_label_mapping
from ai_source_attribution.profiler_baseline import extract_profiler_features


def maybe_binary(samples, enabled: bool):
    if not enabled:
        return samples
    for sample in samples:
        sample.label = "human" if sample.label.lower() in {"human", "human_answers"} else "ai"
    return samples


def maybe_one_vs_all(samples, target_label: str):
    if not target_label:
        return samples
    for sample in samples:
        sample.label = target_label if sample.label == target_label else "other"
    return samples


def parse_seeds(value: str) -> List[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def source_fields(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def feature_kind(method: str) -> str:
    if method in {"profiler_baseline", "profiler_rf_ova"}:
        return "profiler"
    if method == "origin_tracing":
        return "logit"
    if method == "sniffer":
        return "sniffer"
    if method == "seqxgpt":
        return "seqxgpt"
    return "hidden"


def limit_per_label(samples, max_per_label: int):
    if max_per_label is None or max_per_label <= 0:
        return samples
    counts: Dict[str, int] = {}
    kept = []
    for sample in samples:
        count = counts.get(sample.label, 0)
        if count < max_per_label:
            kept.append(sample)
            counts[sample.label] = count + 1
    return kept


def stratified_limited_indices(labels: Sequence[str], max_per_label: int, seed: int) -> np.ndarray:
    if max_per_label is None or max_per_label <= 0:
        return np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    by_label: Dict[str, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[str(label)].append(idx)
    selected = []
    for label in sorted(by_label):
        label_indices = np.asarray(by_label[label], dtype=np.int64)
        rng.shuffle(label_indices)
        selected.extend(label_indices[:max_per_label].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def count_by_label(labels: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label in labels:
        counts[str(label)] = counts.get(str(label), 0) + 1
    return counts


def selected_indices_by_label(labels: Sequence[str], indices: Sequence[int]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for idx in indices:
        label = str(labels[int(idx)])
        out.setdefault(label, []).append(int(idx))
    return out


def fit_classifier(x_train, y_train, seed: int):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", multi_class="auto", random_state=seed),
    )
    clf.fit(x_train, y_train)
    return clf


def extract_features(args, model, tokenizer, texts):
    if args.method in {"profiler_baseline", "profiler_rf_ova"}:
        return extract_profiler_features(
            model, tokenizer, texts, args.batch_size, args.max_length,
            context_window=args.profiler_context_window, sample_clip=args.sample_clip, prompt=args.profiler_prompt
        )
    if args.method == "origin_tracing":
        logit_method = args.method
        return extract_logit_features(
            model, tokenizer, texts, logit_method, args.batch_size, args.max_length,
            sequence_length=args.logit_sequence_length
        )
    if args.method == "sniffer":
        raise NotImplementedError(
            "Strict Sniffer is not a single-surrogate logit baseline. "
            "Provide precomputed official Sniffer features via --precomputed_feature_dir "
            "or --precomputed_full_features. Expected files are named all_sniffer_features.npy "
            "or seed_*/train_sniffer_features.npy and seed_*/test_sniffer_features.npy."
        )
    if args.method == "seqxgpt":
        raise NotImplementedError(
            "Strict SeqXGPT uses the official CNN/self-attention network over aligned multi-model "
            "token log-probability lists. Use the official wrapper/script, or provide compatible "
            "official SeqXGPT features. The previous resized-logit + LogisticRegression baseline "
            "is not strict SeqXGPT."
        )
    return extract_hidden_representations(
        model, tokenizer, texts, args.layer_pos, args.pooling, args.last_k, args.batch_size, args.max_length
    )


def feature_cache_key(args, texts: Sequence[str]) -> str:
    h = hashlib.sha256()
    logit_method = args.method
    config = {
        "method": args.method,
        "logit_method": logit_method,
        "feature_kind": feature_kind(args.method),
        "model_path": args.model_path,
        "layer_pos": args.layer_pos,
        "pooling": args.pooling,
        "last_k": args.last_k,
        "max_length": args.max_length,
        "profiler_context_window": args.profiler_context_window,
        "sample_clip": args.sample_clip,
        "profiler_prompt": args.profiler_prompt,
        "logit_sequence_length": args.logit_sequence_length,
        "causal_lm": args.causal_lm,
        "n_texts": len(texts),
    }
    h.update(json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    for text in texts:
        h.update(b"\0")
        h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:24]


def load_or_extract_features(args, model, tokenizer, texts: Sequence[str], output_dir: str):
    ensure_dir(output_dir)
    kind = feature_kind(args.method)
    if not args.use_feature_cache:
        return extract_features(args, model, tokenizer, list(texts))

    ensure_dir(args.feature_cache_dir)
    key = feature_cache_key(args, texts)
    cache_path = os.path.join(args.feature_cache_dir, f"{kind}_{key}.npy")
    meta_path = os.path.join(args.feature_cache_dir, f"{kind}_{key}.json")
    if os.path.exists(cache_path):
        print(f"[Feature Cache] Loading {cache_path}")
        return np.load(cache_path)

    legacy_names = [
        f"all_{kind}_features.npy",
        "all_hidden_or_profiler_features.npy",
    ]
    legacy_dirs = [output_dir]
    out_path = Path(output_dir)
    parts = out_path.parts
    if kind == "hidden":
        for method_name in ("energy", "linear_probe_on_energy"):
            if method_name in parts:
                method_idx = parts.index(method_name)
                hidden_probe_parts = list(parts)
                hidden_probe_parts[method_idx] = "hidden_probe"
                legacy_dirs.append(str(Path(*hidden_probe_parts)))
    if "multiclass" in parts:
        try:
            idx = parts.index("multiclass")
            method = parts[idx + 1]
            leaf = parts[idx + 2]
            if leaf.endswith("_include_human_true"):
                domain = leaf[: -len("_include_human_true")]
                binary_dir = Path(*parts[:idx]) / "binary" / method / domain
                legacy_dirs.append(str(binary_dir))
                if kind == "hidden" and method in {"energy", "linear_probe_on_energy"}:
                    hidden_binary_dir = Path(*parts[:idx]) / "binary" / "hidden_probe" / domain
                    legacy_dirs.append(str(hidden_binary_dir))
        except (ValueError, IndexError):
            pass

    for legacy_dir in legacy_dirs:
        for legacy_name in legacy_names:
            legacy_path = os.path.join(legacy_dir, legacy_name)
            if not os.path.exists(legacy_path):
                continue
            legacy = np.load(legacy_path)
            if legacy.shape[0] == len(texts):
                print(f"[Feature Cache] Migrating legacy features from {legacy_path}")
                np.save(cache_path, legacy)
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "feature_kind": kind,
                            "model_path": args.model_path,
                            "n_texts": len(texts),
                            "shape": list(legacy.shape),
                            "migrated_from": legacy_path,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                return legacy
            print(
                f"[Feature Cache] Ignoring {legacy_path}: "
                f"row count {legacy.shape[0]} != expected {len(texts)}"
            )

    feats = extract_features(args, model, tokenizer, list(texts))
    np.save(cache_path, feats)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_kind": kind,
                "model_path": args.model_path,
                "n_texts": len(texts),
                "shape": list(feats.shape),
                "max_length": args.max_length,
                "layer_pos": args.layer_pos,
                "pooling": args.pooling,
                            "profiler_context_window": args.profiler_context_window,
                            "logit_sequence_length": args.logit_sequence_length,
                        },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[Feature Cache] Saved {cache_path}")
    return feats


def filter_profiler_short_samples(samples, tokenizer, args, name: str):
    if args.method not in {"profiler_baseline", "profiler_rf_ova"}:
        return samples
    kept = []
    dropped = 0
    for sample in samples:
        text = sample.text
        if args.sample_clip and len(text) > args.sample_clip:
            text = text[: args.sample_clip]
        text_ids = tokenizer(text, truncation=True, max_length=args.max_length).input_ids
        # profiler_baseline.py uses tokenizer(...).input_ids[:, 1:], so mirror
        # that exact effective length here.
        effective_text_tokens = max(0, len(text_ids) - 1)
        if effective_text_tokens > args.profiler_context_window:
            kept.append(sample)
        else:
            dropped += 1
    if dropped:
        print(f"[Data Filter] Dropped {dropped} too-short samples from {name} for PROFILER features.")
    if not kept:
        raise ValueError(f"All samples in {name} are too short for PROFILER features.")
    return kept


def classifier_scores(clf, x, labels):
    if hasattr(clf[-1], "predict_proba"):
        probs = clf.predict_proba(x)
        clf_labels = list(clf[-1].classes_)
        out = np.zeros((x.shape[0], len(labels)), dtype=np.float32)
        for j, label in enumerate(labels):
            if label in clf_labels:
                out[:, j] = probs[:, clf_labels.index(label)]
        return out
    return None


def resolve_precomputed_full_features(args, kind: str):
    candidates = []
    if args.precomputed_full_features:
        candidates.append(Path(args.precomputed_full_features))
    candidate_dirs = []
    if args.precomputed_feature_dir:
        candidate_dirs.append(Path(args.precomputed_feature_dir))
    if args.auto_precomputed_feature_search and args.domain and args.domain != "all":
        legacy_root = Path(args.auto_precomputed_output_root)
        include_human = str(args.include_human).lower()
        method_dirs = [args.method]
        for method_name in method_dirs:
            candidate_dirs.append(
                legacy_root / "results" / "multiclass" / method_name / f"{args.domain}_include_human_{include_human}"
            )
        if args.one_vs_all_label and include_human != "true":
            for method_name in method_dirs:
                candidate_dirs.append(
                    legacy_root / "results" / "multiclass" / method_name / f"{args.domain}_include_human_true"
                )
    for root in candidate_dirs:
        candidates.extend(
            [
                root / f"all_{kind}_features.npy",
                root / "all_hidden_or_profiler_features.npy",
                root / "all_hidden_features.npy",
                root / "all_profiler_features.npy",
                root / "all_logit_features.npy",
                root / "all_sniffer_features.npy",
                root / "all_seqxgpt_features.npy",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_precomputed_run_dir(args):
    candidates = []
    if args.precomputed_feature_dir:
        candidates.append(Path(args.precomputed_feature_dir))
    if args.auto_precomputed_feature_search and args.domain and args.domain != "all":
        legacy_root = Path(args.auto_precomputed_output_root)
        include_human = str(args.include_human).lower()
        method_dirs = [args.method]
        for method_name in method_dirs:
            candidates.append(
                legacy_root / "results" / "multiclass" / method_name / f"{args.domain}_include_human_{include_human}"
            )
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    return None


def feature_file_names(kind: str):
    names = [(f"train_{kind}_features.npy", f"test_{kind}_features.npy")]
    if kind == "hidden":
        names.append(("train_hidden_features.npy", "test_hidden_features.npy"))
        names.append(("train_hidden_or_profiler_features.npy", "test_hidden_or_profiler_features.npy"))
    if kind == "profiler":
        names.append(("train_profiler_features.npy", "test_profiler_features.npy"))
        names.append(("train_hidden_or_profiler_features.npy", "test_hidden_or_profiler_features.npy"))
    if kind == "logit":
        names.append(("train_logit_features.npy", "test_logit_features.npy"))
    if kind == "sniffer":
        names.append(("train_sniffer_features.npy", "test_sniffer_features.npy"))
    if kind == "seqxgpt":
        names.append(("train_seqxgpt_features.npy", "test_seqxgpt_features.npy"))
    return names


def load_precomputed_split_features(args, kind: str, train_samples, test_samples):
    run_dir = resolve_precomputed_run_dir(args)
    if not run_dir:
        return None
    seed_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("seed_"))
    if not seed_dirs:
        return None
    preferred_seed = parse_seeds(args.seeds)[0] if args.seeds else None
    if preferred_seed is not None:
        preferred_dir = run_dir / f"seed_{preferred_seed}"
        if preferred_dir.exists():
            seed_dirs = [preferred_dir] + [p for p in seed_dirs if p != preferred_dir]

    expected_dim = expected_feature_dim(args, kind)
    for seed_dir in seed_dirs:
        for train_name, test_name in feature_file_names(kind):
            train_path = seed_dir / train_name
            test_path = seed_dir / test_name
            if not train_path.exists() or not test_path.exists():
                continue
            x_train = np.load(train_path)
            x_test = np.load(test_path)
            if x_train.shape[0] != len(train_samples) or x_test.shape[0] != len(test_samples):
                print(
                    f"[Precomputed Features] Ignoring {seed_dir}: "
                    f"train/test rows {(x_train.shape[0], x_test.shape[0])} != expected {(len(train_samples), len(test_samples))}"
                )
                continue
            if expected_dim is not None and x_train.ndim == 2 and x_train.shape[1] != expected_dim:
                print(
                    f"[Precomputed Features] Ignoring {seed_dir}: "
                    f"feature dim {x_train.shape[1]} != expected {expected_dim} for method={args.method}"
                )
                continue
            print(f"[Precomputed Features] Reusing split features from {seed_dir}")
            return x_train, x_test
    return None


def _feature_index_key(sample):
    return sample.text, sample.label


def _lookup_feature_indices(full_samples, selected_samples):
    by_key = defaultdict(deque)
    by_text = defaultdict(deque)
    for idx, sample in enumerate(full_samples):
        by_key[_feature_index_key(sample)].append(idx)
        by_text[sample.text].append(idx)

    indices = []
    for sample in selected_samples:
        key = _feature_index_key(sample)
        if by_key[key]:
            indices.append(by_key[key].popleft())
        elif by_text[sample.text]:
            indices.append(by_text[sample.text].popleft())
        else:
            raise KeyError("Could not align a split sample to the precomputed full feature matrix.")
    return np.asarray(indices, dtype=np.int64)


def expected_feature_dim(args, kind: str):
    if args.method == "origin_tracing":
        return 73
    if args.method == "sniffer":
        n_models = int(getattr(args, "sniffer_num_known_models", 0) or 0)
        if n_models > 1:
            return n_models + 3 * ((n_models * (n_models - 1)) // 2)
        return None
    if args.method == "seqxgpt":
        return None
    if kind == "profiler":
        context_window = int(args.profiler_context_window)
        return context_window * 10 + (context_window * (context_window - 1)) // 2
    return None


def load_precomputed_fixed_features(args, kind: str, train_samples, test_samples, tokenizer):
    split_features = load_precomputed_split_features(args, kind, train_samples, test_samples)
    if split_features is not None:
        return split_features

    feature_path = resolve_precomputed_full_features(args, kind)
    if not feature_path:
        return None
    full_features = np.load(feature_path)
    expected_dim = expected_feature_dim(args, kind)
    if expected_dim is not None and full_features.ndim == 2 and full_features.shape[1] != expected_dim:
        print(
            f"[Precomputed Features] Ignoring {feature_path}: "
            f"feature dim {full_features.shape[1]} != expected {expected_dim} for method={args.method}"
        )
        return None
    fields = source_fields(args.source_fields)
    full_samples = load_dataset(
        args.dataset,
        args.data_root,
        None,
        fields,
        args.domain,
        args.include_human,
        paraphrase=None,
        profiler_split_root=None,
        profiler_split_seed=args.profiler_split_seed,
    )
    full_samples = maybe_binary(full_samples, args.binary_human_ai)
    full_samples = maybe_one_vs_all(full_samples, args.one_vs_all_label)
    full_samples = filter_profiler_short_samples(full_samples, tokenizer, args, "precomputed_full_samples")
    if full_features.shape[0] != len(full_samples):
        print(
            f"[Precomputed Features] Ignoring {feature_path}: "
            f"row count {full_features.shape[0]} != full sample count {len(full_samples)}"
        )
        return None
    train_idx = _lookup_feature_indices(full_samples, train_samples)
    test_idx = _lookup_feature_indices(full_samples, test_samples)
    print(f"[Precomputed Features] Reusing {feature_path}")
    return full_features[train_idx], full_features[test_idx]


def main():
    parser = argparse.ArgumentParser(description="Black-box AI-generated text source attribution.")
    parser.add_argument("--dataset", choices=["cosur", "profiler", "generic"], default="cosur")
    parser.add_argument("--test_dataset_type", choices=["cosur", "profiler", "generic"], default=None)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--test_data_root", default=None)
    parser.add_argument("--profiler_split_root", default=None)
    parser.add_argument("--profiler_split_seed", type=int, default=42)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--domain", default="all")
    parser.add_argument("--include_human", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--binary_human_ai", action="store_true")
    parser.add_argument("--one_vs_all_label", default="")
    parser.add_argument("--paraphrase_test", action="store_true")
    parser.add_argument("--source_fields", default="human_answers,chatgpt_answers,qwen_answers,llama_answers,deepseek_answers")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", choices=["hidden_probe", "energy", "linear_probe_on_energy", "profiler_baseline", "profiler_rf_ova", "origin_tracing", "sniffer", "seqxgpt"], default="linear_probe_on_energy")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--basis_method", choices=["svd", "pca"], default="svd")
    parser.add_argument("--layer_pos", default="last")
    parser.add_argument("--pooling", choices=["last_token", "mean_pooling", "last_k_mean"], default="last_token")
    parser.add_argument("--last_k", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--profiler_context_window", type=int, default=6)
    parser.add_argument("--sample_clip", type=int, default=4000)
    parser.add_argument("--profiler_prompt", default="Complete the following text: ")
    parser.add_argument("--logit_sequence_length", type=int, default=256)
    parser.add_argument(
        "--sniffer_num_known_models",
        type=int,
        default=0,
        help="Optional dimensionality check for official Sniffer features: n + 3*C(n,2).",
    )
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--max_train_per_label", type=int, default=None)
    parser.add_argument("--max_test_per_label", type=int, default=None)
    parser.add_argument(
        "--shuffle_train_limit_by_seed",
        action="store_true",
        help="When --max_train_per_label is set on fixed train/test splits, draw a stratified random subset per seed.",
    )
    parser.add_argument("--feature_cache_dir", default="ai_source_attribution/outputs/feature_cache")
    parser.add_argument("--precomputed_feature_dir", default="")
    parser.add_argument("--precomputed_full_features", default="")
    parser.add_argument("--auto_precomputed_feature_search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto_precomputed_output_root", default="ai_source_attribution/outputs")
    parser.add_argument(
        "--require_precomputed_features",
        action="store_true",
        help="Fail instead of extracting features when no compatible precomputed feature matrix is found.",
    )
    parser.add_argument("--use_feature_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--causal_lm", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    fields = source_fields(args.source_fields)
    seeds = parse_seeds(args.seeds)
    test_dataset_type = args.test_dataset_type or args.dataset
    use_random_split = (
        (args.test_split == "" or args.test_split.lower() == "none")
        and not args.test_data_root
        and not args.profiler_split_root
    )
    if use_random_split:
        all_samples = load_dataset(
            args.dataset, args.data_root, None, fields, args.domain, args.include_human, paraphrase=None,
            profiler_split_root=args.profiler_split_root, profiler_split_seed=args.profiler_split_seed
        )
        all_samples = maybe_binary(all_samples, args.binary_human_ai)
        all_samples = maybe_one_vs_all(all_samples, args.one_vs_all_label)
        train_samples, test_samples = stratified_split(all_samples, args.train_ratio, seeds[0])
        if not args.shuffle_train_limit_by_seed:
            train_samples = limit_per_label(train_samples, args.max_train_per_label)
        test_samples = limit_per_label(test_samples, args.max_test_per_label)
    else:
        train_split = None if args.train_split.lower() == "none" else args.train_split
        test_split = None if args.test_split.lower() == "none" else args.test_split
        train_samples = load_dataset(
            args.dataset, args.data_root, train_split, fields, args.domain, args.include_human, paraphrase=False,
            profiler_split_root=args.profiler_split_root, profiler_split_seed=args.profiler_split_seed
        )
        test_root = args.test_data_root or args.data_root
        test_samples = load_dataset(
            test_dataset_type, test_root, test_split, fields, args.domain, args.include_human, paraphrase=args.paraphrase_test or None,
            profiler_split_root=args.profiler_split_root, profiler_split_seed=args.profiler_split_seed
        )
        train_samples = maybe_binary(train_samples, args.binary_human_ai)
        test_samples = maybe_binary(test_samples, args.binary_human_ai)
        train_samples = maybe_one_vs_all(train_samples, args.one_vs_all_label)
        test_samples = maybe_one_vs_all(test_samples, args.one_vs_all_label)
        if not args.shuffle_train_limit_by_seed:
            train_samples = limit_per_label(train_samples, args.max_train_per_label)
        test_samples = limit_per_label(test_samples, args.max_test_per_label)

    labels = ["other", args.one_vs_all_label] if args.one_vs_all_label else sorted(set(s.label for s in train_samples))
    missing = sorted(set(s.label for s in test_samples) - set(labels))
    if missing:
        raise ValueError(f"Test labels not present in training references: {missing}")

    model, tokenizer = load_surrogate(args.model_path, causal_lm=args.causal_lm)
    if use_random_split:
        all_samples = filter_profiler_short_samples(all_samples, tokenizer, args, "all_samples")
        train_samples, test_samples = stratified_split(all_samples, args.train_ratio, seeds[0])
        train_samples = limit_per_label(train_samples, args.max_train_per_label)
        test_samples = limit_per_label(test_samples, args.max_test_per_label)
    else:
        train_samples = filter_profiler_short_samples(train_samples, tokenizer, args, "train_samples")
        test_samples = filter_profiler_short_samples(test_samples, tokenizer, args, "test_samples")
    labels = ["other", args.one_vs_all_label] if args.one_vs_all_label else sorted(set(s.label for s in train_samples))
    missing = sorted(set(s.label for s in test_samples) - set(labels))
    if missing:
        raise ValueError(f"Test labels not present in training references after filtering: {missing}")
    save_label_mapping(labels, args.output_dir)
    kind = feature_kind(args.method)

    if use_random_split:
        id_to_index = {id(sample): i for i, sample in enumerate(all_samples)}
        split_plan = {}
        selected_indices = set()
        for seed in seeds:
            split_train, split_test = stratified_split(all_samples, args.train_ratio, seed)
            split_train = limit_per_label(split_train, args.max_train_per_label)
            split_test = limit_per_label(split_test, args.max_test_per_label)
            train_idx = np.asarray([id_to_index[id(s)] for s in split_train])
            test_idx = np.asarray([id_to_index[id(s)] for s in split_test])
            split_plan[seed] = (split_train, split_test, train_idx, test_idx)
            selected_indices.update(train_idx.tolist())
            selected_indices.update(test_idx.tolist())
        selected_indices = np.asarray(sorted(selected_indices))
        selected_texts = [all_samples[i].text for i in selected_indices]
        if args.require_precomputed_features:
            raise ValueError(
                "--require_precomputed_features is set, but this run is using a random split. "
                "Use a fixed --train_split/--test_split, preferably with --profiler_split_root, "
                "so precomputed train/test features can be reused without changing the test set."
            )
        selected_features = load_or_extract_features(args, model, tokenizer, selected_texts, args.output_dir)
        row_by_original_idx = {int(original_idx): row for row, original_idx in enumerate(selected_indices.tolist())}
        np.save(os.path.join(args.output_dir, f"all_{kind}_features.npy"), selected_features)
    else:
        train_texts = [s.text for s in train_samples]
        test_texts = [s.text for s in test_samples]
        precomputed = load_precomputed_fixed_features(args, kind, train_samples, test_samples, tokenizer)
        if precomputed is None:
            if args.require_precomputed_features:
                raise FileNotFoundError(
                    "No compatible precomputed split/full feature matrix found, and --require_precomputed_features is set."
                )
            x_train_fixed = load_or_extract_features(args, model, tokenizer, train_texts, args.output_dir)
            x_test_fixed = load_or_extract_features(args, model, tokenizer, test_texts, args.output_dir)
        else:
            x_train_fixed, x_test_fixed = precomputed
        y_train_fixed = np.asarray([s.label for s in train_samples])
        y_test_fixed = np.asarray([s.label for s in test_samples])

    run_metrics = []
    per_seed_train_counts = {}
    per_seed_train_label_counts = {}
    for seed in seeds:
        run_dir = os.path.join(args.output_dir, f"seed_{seed}")
        ensure_dir(run_dir)
        train_selection_idx = None
        if use_random_split:
            train_samples, test_samples, train_idx, test_idx = split_plan[seed]
            x_train = selected_features[np.asarray([row_by_original_idx[int(i)] for i in train_idx])]
            x_test = selected_features[np.asarray([row_by_original_idx[int(i)] for i in test_idx])]
            y_train = np.asarray([s.label for s in train_samples])
            y_test = np.asarray([s.label for s in test_samples])
            train_selection_idx = np.arange(len(y_train), dtype=np.int64)
        else:
            x_train = x_train_fixed
            x_test = x_test_fixed
            y_train = y_train_fixed
            y_test = y_test_fixed
            if args.shuffle_train_limit_by_seed:
                train_subset_idx = stratified_limited_indices(y_train_fixed, args.max_train_per_label, seed)
                x_train = x_train_fixed[train_subset_idx]
                y_train = y_train_fixed[train_subset_idx]
                train_selection_idx = train_subset_idx
            else:
                train_selection_idx = np.arange(len(y_train_fixed), dtype=np.int64)

        per_seed_train_counts[str(seed)] = int(len(y_train))
        per_seed_train_label_counts[str(seed)] = count_by_label(y_train)

        np.save(os.path.join(run_dir, f"train_{kind}_features.npy"), x_train)
        np.save(os.path.join(run_dir, f"test_{kind}_features.npy"), x_test)
        with open(os.path.join(run_dir, "train_selection.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "max_train_per_label": args.max_train_per_label,
                    "shuffle_train_limit_by_seed": args.shuffle_train_limit_by_seed,
                    "n_train_total": int(len(y_train)),
                    "n_train_by_label": count_by_label(y_train),
                    "selected_indices_relative_to_full_train_split": [int(i) for i in train_selection_idx],
                    "selected_indices_by_label_relative_to_full_train_split": selected_indices_by_label(
                        y_train_fixed if not use_random_split else y_train,
                        train_selection_idx,
                    ),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        if args.method == "profiler_rf_ova":
            ova_scores = np.zeros((x_test.shape[0], len(labels)), dtype=np.float32)
            source_auroc = {}
            for j, label in enumerate(labels):
                y_bin = (y_train == label).astype(int)
                clf = RandomForestClassifier(
                    n_estimators=200,
                    random_state=seed,
                    bootstrap=False,
                    criterion="entropy",
                    max_depth=7,
                    n_jobs=8,
                )
                clf.fit(x_train, y_bin)
                ova_scores[:, j] = clf.predict_proba(x_test)[:, 1]
                joblib.dump(clf, os.path.join(run_dir, f"profiler_ova_{label}.joblib"))
            y_pred = np.asarray(labels)[np.argmax(ova_scores, axis=1)]
            scores = ova_scores
            with open(os.path.join(run_dir, "profiler_ova_note.json"), "w", encoding="utf-8") as f:
                json.dump({"classifier": "RandomForest one-vs-all", "matches_official_profiler_classifier": True}, f, indent=2)
        elif args.method in {"hidden_probe", "profiler_baseline", "origin_tracing", "sniffer", "seqxgpt"}:
            clf = fit_classifier(x_train, y_train, seed)
            y_pred = clf.predict(x_test)
            scores = classifier_scores(clf, x_test, labels)
            joblib.dump(clf, os.path.join(run_dir, "classifier.joblib"))
        else:
            spaces = build_source_spaces(x_train, y_train, args.k, args.basis_method)
            train_energy = energy_matrix(x_train, spaces)
            test_energy = energy_matrix(x_test, spaces)
            np.save(os.path.join(run_dir, "train_energy_features.npy"), train_energy)
            np.save(os.path.join(run_dir, "test_energy_features.npy"), test_energy)
            np.savez(os.path.join(run_dir, "source_spaces.npz"), **{f"{k}_basis": v.basis for k, v in spaces.items()})
            if args.method == "energy":
                y_pred = predict_by_max_energy(x_test, spaces)
                scores = test_energy
            else:
                clf = fit_classifier(train_energy, y_train, seed)
                y_pred = clf.predict(test_energy)
                scores = classifier_scores(clf, test_energy, labels)
                joblib.dump(clf, os.path.join(run_dir, "energy_classifier.joblib"))
        metrics = evaluate_and_save(y_test, y_pred, labels, run_dir, scores=scores)
        run_metrics.append(metrics)

    summary = {
        "config": vars(args),
        "n_train": int(np.mean(list(per_seed_train_counts.values()))) if per_seed_train_counts else len(train_samples),
        "n_train_per_seed": per_seed_train_counts,
        "n_train_by_label_per_seed": per_seed_train_label_counts,
        "n_test": len(test_samples),
        "labels": labels,
        "mean_std": aggregate_runs(run_metrics),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary["mean_std"], indent=2))


if __name__ == "__main__":
    main()
