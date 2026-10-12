"""Multi-label chest X-ray dataset backed by a processed CSV.

Expected CSV columns:
  - ``image`` : filename (resolved against ``images_dir``) or an absolute path
  - either one binary column per label in ``label_names``
  - or a ``labels`` column with ``|``-separated finding names
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from openmed.data.labels import NIH_LABELS, labels_to_vector, parse_label_string


class ChestXrayDataset:
    """Torch ``Dataset`` (duck-typed to avoid a hard torch import at module load)."""

    def __init__(
        self,
        frame: pd.DataFrame,
        images_dir: str | Path,
        label_names: Sequence[str] = NIH_LABELS,
        transform=None,
        image_col: str = "image",
        label_col: str = "labels",
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.label_names = list(label_names)
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col
        self._has_label_columns = all(name in self.frame.columns for name in self.label_names)

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve_path(self, value: str) -> Path:
        path = Path(str(value))
        if path.is_absolute() and path.exists():
            return path
        candidate = self.images_dir / path
        if candidate.exists():
            return candidate
        return candidate

    def __getitem__(self, index: int):
        import torch

        row = self.frame.iloc[index]
        image_path = self._resolve_path(row[self.image_col])
        image = Image.open(image_path).convert("RGB")

        if self._has_label_columns:
            target = row[self.label_names].to_numpy(dtype=np.float32)
        else:
            raw = row[self.label_col] if self.label_col in row else ""
            target = labels_to_vector(parse_label_string(raw), self.label_names)

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "target": torch.as_tensor(target, dtype=torch.float32),
            "image_id": str(row[self.image_col]),
        }
