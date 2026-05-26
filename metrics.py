import json
import os
from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_label_mapping(labels: Iterable[str], output_dir: str) -> Dict[str, int]:
    mapping = {label: i for i, label in enumerate(sorted(set(labels)))}
    with open(os.path.join(output_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    return mapping


def evaluate_and_save(
    y_true,
    y_pred,
    labels,
    output_dir: str,
    scores: Optional[np.ndarray] = None,
    prefix: str = "test",
) -> Dict[str, object]:
    ensure_dir(output_dir)
    labels = list(labels)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    auroc = None
    per_class_auroc = {}
    if scores is not None and len(labels) > 1:
        try:
            if len(labels) == 2:
                positive_col = 1 if scores.ndim == 2 else None
                auroc = roc_auc_score(np.asarray(y_true) == labels[1], scores[:, positive_col])
                per_class_auroc[labels[1]] = float(auroc)
            else:
                y_bin = label_binarize(y_true, classes=labels)
                auroc = roc_auc_score(y_bin, scores, average="macro", multi_class="ovr")
                for i, label in enumerate(labels):
                    per_class_auroc[label] = float(roc_auc_score(y_bin[:, i], scores[:, i]))
        except ValueError:
            auroc = None

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    result = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "auroc": None if auroc is None else float(auroc),
        "per_class_auroc": per_class_auroc,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    with open(os.path.join(output_dir, f"{prefix}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    np.save(os.path.join(output_dir, f"{prefix}_confusion_matrix.npy"), cm)
    with open(os.path.join(output_dir, f"{prefix}_confusion_matrix.csv"), "w", encoding="utf-8") as f:
        f.write("," + ",".join(labels) + "\n")
        for label, row in zip(labels, cm):
            f.write(label + "," + ",".join(str(int(x)) for x in row) + "\n")
    return result


def aggregate_runs(run_metrics):
    keys = ["accuracy", "f1_macro", "f1_weighted", "auroc"]
    out = {}
    for key in keys:
        values = [m[key] for m in run_metrics if m.get(key) is not None]
        if values:
            out[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return out
