"""Dataset construction shared by training and evaluation.

``data.dataset`` selects the source:

* ``files``    - a processed CSV + image directory (NIH ChestX-ray14 layout)
* ``medmnist`` - MedMNIST ChestMNIST (NIH-derived, official splits)
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from openmed.data.dataset import ChestXrayDataset
from openmed.data.labels import NIH_LABELS
from openmed.data.transforms import build_transforms


def build_dataset(
    cfg,
    split: str,
    train: bool = False,
    label_names: Optional[Sequence[str]] = None,
    image_size: Optional[int] = None,
    apply_transform: bool = True,
):
    data = cfg.data
    label_names = list(label_names or NIH_LABELS)
    image_size = int(image_size or data.get("image_size", 224))
    transform = build_transforms(image_size, train, data.mean, data.std) if apply_transform else None

    kind = str(data.get("dataset", "files")).lower()
    if kind == "medmnist":
        from openmed.data.medmnist import MedMNISTDataset

        dataset = MedMNISTDataset(
            split=split,
            root=data.get("medmnist_root", "data/raw/medmnist"),
            size=int(data.get("medmnist_size", image_size)),
            transform=transform,
            label_names=label_names,
            as_rgb=True,
        )
    elif kind == "files":
        frame = pd.read_csv(data.split_csv)
        subset = frame[frame["split"] == split].reset_index(drop=True)
        if subset.empty:
            raise ValueError(f"No rows for split={split!r} in {data.split_csv}")
        dataset = ChestXrayDataset(
            subset,
            images_dir=data.images_dir,
            label_names=label_names,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown data.dataset: {kind!r} (use 'files' or 'medmnist')")

    if data.get("max_samples"):
        import torch
        from torch.utils.data import Subset

        limit = min(int(data.max_samples), len(dataset))
        dataset = Subset(dataset, list(range(limit)))
    return dataset


def build_dataloaders(cfg, label_names: Optional[Sequence[str]] = None):
    """Return ``{split: (DataLoader, dataset)}`` for train/val/test."""
    import torch
    from torch.utils.data import DataLoader

    data = cfg.data
    loaders = {}
    for split, is_train in (("train", True), ("val", False), ("test", False)):
        dataset = build_dataset(cfg, split, train=is_train, label_names=label_names)
        loaders[split] = (
            DataLoader(
                dataset,
                batch_size=int(data.get("batch_size", 32)),
                shuffle=is_train,
                num_workers=int(data.get("num_workers", 0)),
                pin_memory=False,
                drop_last=False,
            ),
            dataset,
        )
    return loaders


def labels_of(dataset) -> Optional[np.ndarray]:
    """Best-effort label matrix for class-weight computation."""
    import torch
    from torch.utils.data import Subset

    if isinstance(dataset, Subset):
        base = dataset.dataset
        labels = getattr(base, "labels", None)
        if labels is None:
            return None
        return np.asarray(labels)[list(dataset.indices)]
    if isinstance(dataset, ChestXrayDataset):
        if dataset._has_label_columns:
            return dataset.frame[dataset.label_names].to_numpy(dtype=np.float32)
        return None
    labels = getattr(dataset, "labels", None)
    return np.asarray(labels) if labels is not None else None
