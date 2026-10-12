"""Fine-tune a pretrained vision backbone on chest X-rays (Apple Silicon MPS)."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from openmed.config import load_config
from openmed.data.factory import build_dataloaders, labels_of
from openmed.data.labels import NIH_LABELS
from openmed.models.classifier import build_model, save_checkpoint
from openmed.train.engine import (
    build_optimizer,
    build_scheduler,
    evaluate,
    pos_weight_from_labels,
    train_one_epoch,
)
from openmed.train.metrics import compute_metrics
from openmed.utils import ensure_dir, get_device, get_logger, save_json, seed_everything

LOGGER = get_logger(__name__)


def run_training(cfg) -> dict:
    seed_everything(int(cfg.project.get("seed", 42)))
    device = get_device(cfg.train.get("device", "auto"))
    LOGGER.info("Device: %s", device)

    output_dir = ensure_dir(cfg.project.output_dir)
    label_names = list(NIH_LABELS)
    loaders = build_dataloaders(cfg, label_names)
    LOGGER.info(
        "Splits: %s",
        {split: len(loader.dataset) for split, (loader, _) in loaders.items()},
    )

    model = build_model(
        cfg.model.backbone,
        num_classes=len(label_names),
        pretrained=bool(cfg.model.pretrained),
        dropout=float(cfg.model.get("dropout", 0.0)),
    ).to(device)

    pos_weight = None
    if bool(cfg.train.get("pos_weight", False)) and "train" in loaders:
        train_labels = labels_of(loaders["train"][1])
        if train_labels is not None:
            pos_weight = pos_weight_from_labels(train_labels, device)
            LOGGER.info(
                "Using pos_weight: %s", [round(float(x), 2) for x in pos_weight.cpu()]
            )
        else:
            LOGGER.warning("pos_weight requested but labels unavailable; skipping")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = build_optimizer(
        model, cfg.train.optimizer, float(cfg.train.lr), float(cfg.train.weight_decay)
    )
    scheduler = build_scheduler(
        optimizer, cfg.train.get("scheduler", "none"), int(cfg.train.epochs),
        int(cfg.train.get("warmup_epochs", 0)),
    )

    best_score = float("-inf")
    best_path = Path(output_dir) / "best.pt"
    patience = int(cfg.train.get("early_stopping_patience", 0))
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, int(cfg.train.epochs) + 1):
        start = time.time()
        train_loss = train_one_epoch(
            model, loaders["train"][0], optimizer, criterion, device,
            grad_clip=cfg.train.get("grad_clip"),
        )
        val_loss, val_probs, val_targets = evaluate(
            model, loaders["val"][0], criterion, device
        )
        metrics = compute_metrics(val_targets, val_probs, label_names)
        score = metrics.get(cfg.train.get("monitor", "mean_auroc"), metrics["mean_auroc"])
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - start, 1),
            **{k: v for k, v in metrics.items() if not k.startswith("auroc/")},
        }
        history.append(row)
        LOGGER.info(
            "epoch %d | train %.4f | val %.4f | AUROC %.4f | %.1fs",
            epoch, train_loss, val_loss, metrics["mean_auroc"], row["seconds"],
        )

        if score == score and score > best_score:  # not NaN and improved
            best_score = score
            epochs_without_improvement = 0
            save_checkpoint(
                best_path, model, cfg.model.backbone, int(cfg.data.image_size),
                label_names, epoch=epoch, metrics=metrics,
            )
            LOGGER.info("  saved new best (%s=%.4f)", cfg.train.get("monitor", "mean_auroc"), score)
        else:
            epochs_without_improvement += 1
            if patience and epochs_without_improvement >= patience:
                LOGGER.info("Early stopping at epoch %d", epoch)
                break

    pd.DataFrame(history).to_csv(Path(output_dir) / "history.csv", index=False)

    result = {"best_score": best_score, "checkpoint": str(best_path), "history": history}
    if "test" in loaders:
        from openmed.models.classifier import load_checkpoint

        best_model, meta = load_checkpoint(best_path, device=device)
        test_loss, test_probs, test_targets = evaluate(
            best_model, loaders["test"][0], criterion, device
        )
        test_metrics = compute_metrics(test_targets, test_probs, meta["label_names"])
        result["test"] = test_metrics
        save_json(test_metrics, Path(output_dir) / "test_metrics.json")
        LOGGER.info("Test mean AUROC: %.4f", test_metrics["mean_auroc"])

    save_json(result, Path(output_dir) / "train_summary.json")
    return result


def main(config_path: str) -> None:
    run_training(load_config(config_path))


if __name__ == "__main__":  # pragma: no cover
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "configs/chestxray14.yaml")
