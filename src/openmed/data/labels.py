"""NIH ChestX-ray14 label vocabulary and helpers."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

# The 14 pathology labels used by NIH ChestX-ray14.
NIH_LABELS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

LABEL_TO_INDEX = {label: index for index, label in enumerate(NIH_LABELS)}

# Free-text aliases -> canonical label, for grounding RAG queries.
LABEL_ALIASES = {
    "pleural effusion": "Effusion",
    "effusion": "Effusion",
    "enlarged heart": "Cardiomegaly",
    "cardiac enlargement": "Cardiomegaly",
    "lung collapse": "Atelectasis",
    "collapsed lung": "Atelectasis",
    "pleural thickening": "Pleural_Thickening",
    "infiltrate": "Infiltration",
    "infiltrates": "Infiltration",
    "opacity": "Infiltration",
    "pneumothorax": "Pneumothorax",
    "pneumonia": "Pneumonia",
    "consolidation": "Consolidation",
    "pulmonary edema": "Edema",
    "edema": "Edema",
}


def parse_label_string(raw: str, separator: str = "|") -> List[str]:
    """Parse a NIH ``Finding Labels`` cell into canonical labels.

    ``"No Finding"`` (or empty) yields an empty list.
    """
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text.lower() in {"no finding", "normal", "nan"}:
        return []
    labels = [part.strip() for part in text.split(separator) if part.strip()]
    canonical = []
    for label in labels:
        canonical.append(LABEL_ALIASES.get(label.lower(), label))
    return canonical


def labels_to_vector(
    labels: Iterable[str], label_names: Sequence[str] = NIH_LABELS
) -> np.ndarray:
    """Multi-hot encode ``labels`` against ``label_names``."""
    vector = np.zeros(len(label_names), dtype=np.float32)
    index = {name: i for i, name in enumerate(label_names)}
    for label in labels:
        if label in index:
            vector[index[label]] = 1.0
    return vector


def normalize_label(name: str) -> str:
    """Map a free-text finding to its canonical NIH label when possible."""
    if name in LABEL_TO_INDEX:
        return name
    return LABEL_ALIASES.get(name.strip().lower(), name)
