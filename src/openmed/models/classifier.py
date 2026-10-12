"""Multi-label chest X-ray classifier built on timm backbones."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from openmed.data.labels import NIH_LABELS


def build_model(
    backbone: str = "densenet121",
    num_classes: int = len(NIH_LABELS),
    pretrained: bool = True,
    dropout: float = 0.0,
) -> nn.Module:
    """Create a timm model with a fresh ``num_classes`` head."""
    import timm

    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )
    return model


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    backbone: str,
    image_size: int,
    label_names=None,
    epoch: Optional[int] = None,
    metrics: Optional[dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "backbone": backbone,
            "image_size": image_size,
            "label_names": list(label_names or NIH_LABELS),
            "epoch": epoch,
            "metrics": metrics or {},
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu", build: bool = True):
    """Load a checkpoint. Returns ``(model, meta)`` when ``build=True`` else ``meta``."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    meta = {
        "backbone": checkpoint.get("backbone", "densenet121"),
        "image_size": checkpoint.get("image_size", 320),
        "label_names": checkpoint.get("label_names", list(NIH_LABELS)),
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics", {}),
    }
    if not build:
        return meta
    model = build_model(meta["backbone"], num_classes=len(meta["label_names"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, meta
