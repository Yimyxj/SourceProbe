import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass
class TextSample:
    text: str
    label: str
    domain: str = "all"
    split: Optional[str] = None
    variant: str = "normal"


def _clean_text(value) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    if value is None:
        return ""
    return str(value).strip()


def _record_text(item: dict):
    for key in ("text", "content", "answer", "generation", "abs", "essay"):
        if key in item:
            return item[key]
    if "prompt" in item and "completion" in item:
        return f"{item['prompt']}{item['completion']}"
    return None


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        yield from obj
    elif isinstance(obj, dict):
        for key in ("data", "samples", "records", "examples"):
            if isinstance(obj.get(key), list):
                yield from obj[key]
                return
        yield obj
    else:
        raise ValueError(f"Unsupported JSON root in {path}: {type(obj).__name__}")


def _read_csv(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def _iter_records(path: Path):
    if path.suffix.lower() == ".jsonl":
        yield from _read_jsonl(path)
    elif path.suffix.lower() == ".json":
        yield from _read_json(path)
    elif path.suffix.lower() == ".csv":
        yield from _read_csv(path)


def _candidate_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Data path does not exist: {root}")
    exts = {".jsonl", ".json", ".csv"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def load_profiler_group_indices(
    split_root: Optional[str],
    domain: Optional[str],
    split: Optional[str],
    seed: int = 42,
) -> Optional[Set[int]]:
    if not split_root or not domain or domain == "all" or not split:
        return None
    split_name = split.lower()
    if split_name in {"none", ""}:
        return None
    root = Path(split_root)
    manifest_path = root
    if root.is_dir():
        direct = root / f"{domain}_seed{seed}.json"
        nested = root / "splits" / f"{domain}_seed{seed}.json"
        if direct.exists():
            manifest_path = direct
        elif nested.exists():
            manifest_path = nested
        else:
            matches = sorted(root.rglob(f"{domain}_seed*.json"))
            if not matches:
                raise FileNotFoundError(f"No split manifest found for domain={domain} under {root}")
            manifest_path = matches[0]
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    key = "train_indices" if split_name == "train" else "test_indices"
    if key not in manifest:
        raise ValueError(f"Split manifest {manifest_path} does not contain {key}")
    return set(int(i) for i in manifest[key])


def load_cosur_jsonl(
    data_root: str,
    split: str,
    source_fields: Sequence[str],
    domain: Optional[str] = None,
    include_human: bool = True,
    max_samples_per_source: Optional[int] = None,
) -> List[TextSample]:
    root = Path(data_root)
    file_map = {
        "train": root / "train_all.jsonl",
        "test": root / "test_all.jsonl",
        "val": root / "val_all.jsonl",
        "train_ch": root / "train_ch.jsonl",
        "test_ch": root / "test_ch.jsonl",
        "val_ch": root / "val_ch.jsonl",
    }
    path = file_map.get(split, root / split)
    if not path.exists():
        raise FileNotFoundError(f"CoSur split file not found: {path}")

    counts: Dict[str, int] = {}
    samples: List[TextSample] = []
    for item in _read_jsonl(path):
        item_domain = str(item.get("source", item.get("domain", "all")))
        if domain and domain != "all" and item_domain.lower() != domain.lower():
            continue
        for field in source_fields:
            if not include_human and field.lower().startswith("human"):
                continue
            if field not in item:
                continue
            if max_samples_per_source is not None and counts.get(field, 0) >= max_samples_per_source:
                continue
            text = _clean_text(item[field])
            if not text:
                continue
            samples.append(TextSample(text=text, label=field, domain=item_domain, split=split))
            counts[field] = counts.get(field, 0) + 1
    _validate_samples(samples, f"CoSur {path}")
    return samples


def load_profiler_like(
    data_root: str,
    split: Optional[str],
    domain: Optional[str] = None,
    include_human: bool = True,
    paraphrase: Optional[bool] = None,
    text_key: str = "text",
    label_key: str = "label",
    group_indices: Optional[Set[int]] = None,
) -> List[TextSample]:
    root = Path(data_root)
    samples: List[TextSample] = []
    for path in _candidate_files(root):
        inferred_domain = path.parent.name if path.parent != root else "all"
        inferred_label = path.stem
        if "_" in inferred_label:
            inferred_label = inferred_label.split("_", 1)[1]
        file_lower = path.name.lower()
        for item_index, item in enumerate(_iter_records(path)):
            if group_indices is not None and item_index not in group_indices:
                continue
            if isinstance(item, str):
                item = {"text": item, "label": inferred_label, "domain": inferred_domain}
            elif isinstance(item, list):
                item = {"text": "".join(_clean_text(part) for part in item), "label": inferred_label, "domain": inferred_domain}
            if not isinstance(item, dict):
                continue
            item_split = str(item.get("split", "") or "")
            if split and item_split and item_split.lower() != split.lower():
                continue
            item_domain = str(item.get("domain", item.get("source_domain", inferred_domain)) or inferred_domain)
            if domain and domain != "all" and item_domain.lower() != domain.lower() and domain.lower() not in file_lower:
                continue
            variant = str(item.get("variant", item.get("attack", item.get("setting", "normal"))) or "normal")
            is_para = "para" in variant.lower() or "paraphrase" in file_lower
            if paraphrase is not None and is_para != paraphrase:
                continue
            text = _clean_text(item.get(text_key) if text_key in item else _record_text(item))
            label = _clean_text(item.get(label_key, item.get("source", item.get("model", item.get("origin", inferred_label)))))
            if not include_human and label.lower() in {"human", "human_answers"}:
                continue
            if not text or not label:
                continue
            samples.append(TextSample(text=text, label=label, domain=item_domain, split=item_split or split, variant=variant))
    _validate_samples(samples, f"Profiler-like {root}")
    return samples


def load_dataset(
    dataset: str,
    data_root: str,
    split: Optional[str],
    source_fields: Sequence[str],
    domain: Optional[str],
    include_human: bool,
    paraphrase: Optional[bool] = None,
    profiler_split_root: Optional[str] = None,
    profiler_split_seed: int = 42,
) -> List[TextSample]:
    if dataset == "cosur":
        if split is None:
            raise ValueError("CoSur loader requires --train_split/--test_split.")
        return load_cosur_jsonl(data_root, split, source_fields, domain, include_human)
    if dataset in {"profiler", "generic"}:
        group_indices = load_profiler_group_indices(profiler_split_root, domain, split, profiler_split_seed)
        return load_profiler_like(data_root, split, domain, include_human, paraphrase, group_indices=group_indices)
    raise ValueError(f"Unknown dataset '{dataset}'. Expected cosur, profiler, or generic.")


def stratified_split(
    samples: Sequence[TextSample],
    train_ratio: float,
    seed: int,
) -> Tuple[List[TextSample], List[TextSample]]:
    by_label: Dict[str, List[TextSample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)
    rng = random.Random(seed)
    train, test = [], []
    for label, items in sorted(by_label.items()):
        rng.shuffle(items)
        n_train = max(1, int(round(len(items) * train_ratio)))
        if n_train >= len(items) and len(items) > 1:
            n_train = len(items) - 1
        train.extend(items[:n_train])
        test.extend(items[n_train:])
    return train, test


def _validate_samples(samples: Sequence[TextSample], name: str) -> None:
    if not samples:
        raise ValueError(f"No usable samples loaded from {name}. Check data path, split, domain, labels, and text fields.")
    empty = [i for i, s in enumerate(samples) if not s.text or not s.label]
    if empty:
        raise ValueError(f"{name} contains empty text or label at indices: {empty[:10]}")
