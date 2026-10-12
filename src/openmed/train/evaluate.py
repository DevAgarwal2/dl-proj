"""Evaluate a trained checkpoint on a split, optionally exporting Grad-CAM grids."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from openmed.data.factory import build_dataset
from openmed.data.labels import NIH_LABELS
from openmed.models.classifier import load_checkpoint
from openmed.models.gradcam import GradCAM, find_last_conv, overlay_cam
from openmed.train.engine import evaluate
from openmed.train.metrics import compute_metrics
from openmed.utils import ensure_dir, get_device, get_logger, save_json

LOGGER = get_logger(__name__)


def run_evaluation(cfg, checkpoint: str, split: str = "test") -> dict:
    device = get_device(cfg.train.get("device", "auto"))
    model, meta = load_checkpoint(checkpoint, device=device)
    label_names = list(meta["label_names"])

    dataset = build_dataset(
        cfg, split, train=False, label_names=label_names, image_size=int(meta["image_size"])
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(cfg.data.batch_size))

    criterion = torch.nn.BCEWithLogitsLoss()
    _, probs, targets = evaluate(model, loader, criterion, device)
    metrics = compute_metrics(targets, probs, label_names)

    output_dir = ensure_dir(cfg.project.output_dir)
    save_json(metrics, Path(output_dir) / f"{split}_metrics.json")

    per_class = {
        "label": label_names,
        "auroc": [metrics.get(f"auroc/{name}", float("nan")) for name in label_names],
    }
    pd.DataFrame(per_class).to_csv(Path(output_dir) / f"{split}_per_class.csv", index=False)

    LOGGER.info("mean AUROC %.4f | macro F1 %.4f", metrics["mean_auroc"], metrics["macro_f1"])
    return metrics


def export_gradcam_grid(cfg, checkpoint: str, split: str = "test", n: int = 8) -> Path:
    """Save a grid of Grad-CAM overlays for the top predicted finding per image."""
    from PIL import Image

    device = get_device(cfg.train.get("device", "auto"))
    model, meta = load_checkpoint(checkpoint, device=device)
    label_names = list(meta["label_names"])
    from openmed.data.transforms import build_transforms

    transform = build_transforms(int(meta["image_size"]), False, cfg.data.mean, cfg.data.std)
    dataset = build_dataset(
        cfg, split, train=False, label_names=label_names,
        image_size=int(meta["image_size"]), apply_transform=False,
    )

    cam = GradCAM(model, find_last_conv(model))
    tiles = []
    try:
        for index in range(min(n, len(dataset))):
            sample = dataset[index]
            pil = sample["image"].convert("RGB")
            tensor = transform(pil)
            heatmap, probs = cam(tensor.to(device), class_idx=None)
            top = int(np.argmax(probs))
            tiles.append(overlay_cam(pil, heatmap))
            LOGGER.info("%s -> %s %.3f", sample["image_id"], label_names[top], probs[top])
    finally:
        cam.remove()

    output = Path(ensure_dir(cfg.project.output_dir)) / f"gradcam_{split}.png"
    if tiles:
        width, height = tiles[0].size
        grid = Image.new("RGB", (width * len(tiles), height))
        for i, tile in enumerate(tiles):
            grid.paste(tile.resize((width, height)), (i * width, 0))
        grid.save(output)
    LOGGER.info("Saved Grad-CAM grid -> %s", output)
    return output
