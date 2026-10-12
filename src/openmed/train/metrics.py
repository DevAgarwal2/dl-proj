"""Multi-label evaluation metrics."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from openmed.data.labels import NIH_LABELS


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: Sequence[str] = NIH_LABELS,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Per-class AUROC / AP plus macro means and exact-match accuracy."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    label_names = list(label_names)

    per_class: Dict[str, float] = {}
    aurocs: List[float] = []
    for index, name in enumerate(label_names):
        truth = y_true[:, index]
        score = y_prob[:, index]
        if truth.min() == truth.max():  # no positives or no negatives
            auroc = float("nan")
        else:
            auroc = float(roc_auc_score(truth, score))
        per_class[name] = auroc
        aurocs.append(auroc)

    valid = [value for value in aurocs if not np.isnan(value)]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "mean_auroc": float(np.mean(valid)) if valid else float("nan"),
        "macro_auroc": float(np.nanmean(aurocs)),
        "mean_ap": float(
            np.mean(
                [
                    average_precision_score(y_true[:, i], y_prob[:, i])
                    if y_true[:, i].min() != y_true[:, i].max()
                    else float("nan")
                    for i in range(len(label_names))
                ]
            )
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "exact_match": float(np.mean(np.all(y_pred == y_true, axis=1))),
    }
    for name, value in per_class.items():
        metrics[f"auroc/{name}"] = value
    return metrics
