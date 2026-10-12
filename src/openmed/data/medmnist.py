"""MedMNIST ChestMNIST adapter.

ChestMNIST is derived from the NIH ChestX-ray14 dataset and uses the same 14
findings, but ships as a ~50 MB npz with official train/val/test splits instead
of 45 GB of PNGs. Handy for training a real model quickly and reproducibly.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from openmed.data.labels import NIH_LABELS

VALID_SPLITS = ("train", "val", "test")


class MedMNISTDataset:
    """Duck-typed torch Dataset yielding the OpenMed sample dict."""

    def __init__(
        self,
        split: str = "train",
        root: str | Path = "data/raw/medmnist",
        size: int = 64,
        transform=None,
        label_names: Sequence[str] = NIH_LABELS,
        as_rgb: bool = True,
        download: bool = True,
    ) -> None:
        from medmnist import ChestMNIST

        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
        self.split = split
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.label_names = list(label_names)
        self.transform = transform
        self.as_rgb = as_rgb
        self.size = size
        self.ds = ChestMNIST(
            split=split,
            root=str(self.root),
            download=download,
            size=size,
            as_rgb=as_rgb,
        )
        if self.ds.labels.shape[1] != len(self.label_names):
            raise ValueError(
                f"ChestMNIST has {self.ds.labels.shape[1]} labels, "
                f"expected {len(self.label_names)}"
            )

    def __len__(self) -> int:
        return len(self.ds)

    @property
    def labels(self) -> np.ndarray:
        """Multi-hot labels of shape (n, 14) as float32."""
        return np.asarray(self.ds.labels, dtype=np.float32)

    def __getitem__(self, index: int):
        import torch

        image, target = self.ds[index]
        if self.transform is not None:
            image = self.transform(image)
        target = torch.as_tensor(np.asarray(target).reshape(-1), dtype=torch.float32)
        return {"image": image, "target": target, "image_id": f"{self.split}_{index:06d}"}


def dataset_labels(dataset) -> Optional[np.ndarray]:
    """Return the (n, 14) label matrix for a MedMNIST dataset or Subset."""
    import torch
    from torch.utils.data import Subset

    if isinstance(dataset, Subset):
        base = dataset.dataset
        labels = getattr(base, "labels", None)
        if labels is None:
            return None
        return np.asarray(labels)[list(dataset.indices)]
    labels = getattr(dataset, "labels", None)
    return np.asarray(labels) if labels is not None else None
