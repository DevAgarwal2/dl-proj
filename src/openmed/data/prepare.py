"""Build patient-level train/val/test splits from NIH ChestX-ray14 metadata.

Usage:
    python -m openmed.cli prepare-data --config configs/chestxray14.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from openmed.data.labels import NIH_LABELS, parse_label_string
from openmed.utils import ensure_dir, get_logger

LOGGER = get_logger(__name__)

META_COLUMNS = {
    "Image Index": "image",
    "Finding Labels": "finding_labels",
    "Patient ID": "patient_id",
    "Patient Age": "patient_age",
    "Patient Gender": "patient_gender",
    "View Position": "view_position",
}


def load_metadata(metadata_csv: str | Path) -> pd.DataFrame:
    """Read ``Data_Entry_2017.csv`` and normalise its columns."""
    frame = pd.read_csv(metadata_csv)
    frame = frame.rename(columns={k: v for k, v in META_COLUMNS.items() if k in frame.columns})
    if "image" not in frame.columns:
        raise ValueError(f"{metadata_csv} has no 'Image Index' column")
    if "finding_labels" not in frame.columns:
        raise ValueError(f"{metadata_csv} has no 'Finding Labels' column")
    if "patient_id" not in frame.columns:
        # Fall back to filename prefix (e.g. 00000001_000.png -> patient 1).
        frame["patient_id"] = frame["image"].str.split("_").str[0]
    return frame


def binarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the 14 binary label columns and a canonical ``labels`` string."""
    out = frame.copy()
    parsed = out["finding_labels"].apply(parse_label_string)
    out["labels"] = parsed.apply(lambda items: "|".join(items))
    for label in NIH_LABELS:
        out[label] = parsed.apply(lambda items, _l=label: int(_l in items)).astype(np.int8)
    return out


def make_splits(
    frame: pd.DataFrame,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    patient_level: bool = True,
    seed: int = 42,
    official_train_val_list: Optional[str | Path] = None,
    official_test_list: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Assign a ``split`` column of ``train`` / ``val`` / ``test``.

    When the official NIH test list is provided it is used verbatim; the
    remaining studies are split by patient into train/val.
    """
    out = frame.copy()
    rng = np.random.default_rng(seed)

    if official_test_list and Path(official_test_list).exists():
        test_images = {
            line.strip()
            for line in Path(official_test_list).read_text().splitlines()
            if line.strip()
        }
        out["split"] = np.where(out["image"].isin(test_images), "test", "train")
        pool = out[out["split"] == "train"]
    else:
        pool = out

    group_key = "patient_id" if patient_level else "image"
    groups = np.array(sorted(pool[group_key].astype(str).unique()))
    rng.shuffle(groups)

    n_test = int(round(len(groups) * test_fraction)) if official_test_list is None else 0
    n_val = int(round(len(groups) * val_fraction))
    val_groups = set(groups[:n_val])
    test_groups = set(groups[n_val : n_val + n_test])

    def assign(pid: str) -> str:
        pid = str(pid)
        if pid in test_groups:
            return "test"
        if pid in val_groups:
            return "val"
        return "train"

    if official_test_list and Path(official_test_list).exists():
        remaining = out["split"] == "train"
        out.loc[remaining, "split"] = out.loc[remaining, group_key].astype(str).map(
            lambda pid: "val" if pid in val_groups else "train"
        )
    else:
        out["split"] = out[group_key].astype(str).map(assign)
    return out


def prepare_nih(
    metadata_csv: str | Path,
    output_csv: str | Path,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    patient_level: bool = True,
    seed: int = 42,
    official_train_val_list: Optional[str | Path] = None,
    official_test_list: Optional[str | Path] = None,
) -> pd.DataFrame:
    frame = load_metadata(metadata_csv)
    frame = binarize(frame)
    frame = make_splits(
        frame,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        patient_level=patient_level,
        seed=seed,
        official_train_val_list=official_train_val_list,
        official_test_list=official_test_list,
    )
    keep = [
        "image",
        "labels",
        "patient_id",
        "patient_age",
        "patient_gender",
        "view_position",
        "split",
        *NIH_LABELS,
    ]
    frame = frame[[c for c in keep if c in frame.columns]]
    ensure_dir(Path(output_csv).parent)
    frame.to_csv(output_csv, index=False)
    counts = frame["split"].value_counts().to_dict()
    LOGGER.info("Wrote %s (%d studies) splits=%s", output_csv, len(frame), counts)
    return frame


def prepare_from_config(cfg) -> pd.DataFrame:
    data = cfg.data
    return prepare_nih(
        metadata_csv=data.metadata_csv,
        output_csv=data.split_csv,
        val_fraction=float(data.get("val_fraction", 0.10)),
        test_fraction=float(data.get("test_fraction", 0.10)),
        patient_level=bool(data.get("patient_level_split", True)),
        seed=int(cfg.project.get("seed", 42)),
        official_train_val_list=data.get("official_train_val_list"),
        official_test_list=data.get("official_test_list"),
    )
