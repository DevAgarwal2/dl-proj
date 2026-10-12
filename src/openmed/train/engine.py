"""Training / evaluation loops shared by train.py and evaluate.py."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from openmed.utils import get_logger

LOGGER = get_logger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: str,
    grad_clip: float | None = None,
) -> float:
    model.train()
    running = 0.0
    seen = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=False)
        targets = batch["target"].to(device, non_blocking=False)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running += loss.item() * images.size(0)
        seen += images.size(0)
    return running / max(seen, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: str,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return ``(loss, probabilities, targets)``."""
    model.eval()
    running = 0.0
    seen = 0
    probs, targets_all = [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=False)
        targets = batch["target"].to(device, non_blocking=False)
        logits = model(images)
        loss = criterion(logits, targets)

        running += loss.item() * images.size(0)
        seen += images.size(0)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targets_all.append(targets.cpu().numpy())

    probs = np.concatenate(probs, axis=0) if probs else np.zeros((0, 0))
    targets_all = np.concatenate(targets_all, axis=0) if targets_all else np.zeros((0, 0))
    return running / max(seen, 1), probs, targets_all


def build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, name: str, epochs: int, warmup_epochs: int = 0):
    name = (name or "none").lower()
    if name in {"none", "constant"}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.1)
    raise ValueError(f"Unknown scheduler: {name}")


def pos_weight_from_labels(labels: np.ndarray, device: str):
    """``pos_weight = negatives / positives`` per class for BCEWithLogitsLoss."""
    counts = np.asarray(labels, dtype=np.float64)
    positives = counts.sum(axis=0)
    negatives = counts.shape[0] - positives
    weights = np.where(positives > 0, negatives / np.maximum(positives, 1), 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def class_pos_weight(frame, label_names, device: str):
    """Backwards-compatible wrapper around :func:`pos_weight_from_labels`."""
    return pos_weight_from_labels(
        frame[list(label_names)].to_numpy(dtype=np.float64), device
    )
