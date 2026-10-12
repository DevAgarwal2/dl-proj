#!/usr/bin/env python3
"""Export real held-out ChestMNIST X-rays with ground truth and predictions.

Creates:
    samples/images/*.png          upscaled 224px upload-ready X-rays
    samples/gradcam/*.png         Grad-CAM overlays
    samples/predictions.csv       ground truth vs top-3 predictions
    samples/predictions.json      all 14 probabilities per image

Usage:
    python scripts/export_samples.py --checkpoint runs/chestmnist/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openmed.data.labels import NIH_LABELS  # noqa: E402
from openmed.data.medmnist import MedMNISTDataset  # noqa: E402
from openmed.inference import Predictor  # noqa: E402


def pick_indices(labels: np.ndarray) -> list[tuple[int, str]]:
    counts = labels.sum(axis=1)
    picks: list[tuple[int, str]] = []
    # One clean single-finding example per interesting pathology.
    for name in ["Cardiomegaly", "Effusion", "Pneumonia", "Pneumothorax",
                 "Emphysema", "Nodule", "Edema", "Hernia"]:
        index = NIH_LABELS.index(name)
        candidates = np.where((labels[:, index] == 1) & (counts == 1))[0]
        if len(candidates):
            picks.append((int(candidates[0]), "single"))
    # Two multi-label examples and two normals.
    multi = np.where(counts >= 3)[0]
    for index in multi[:2]:
        picks.append((int(index), "multi"))
    normal = np.where(counts == 0)[0]
    for index in normal[:2]:
        picks.append((int(index), "normal"))
    return picks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/chestmnist/best.pt")
    parser.add_argument("--out", default="samples")
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "gradcam").mkdir(parents=True, exist_ok=True)

    dataset = MedMNISTDataset("test", size=64, transform=None)
    labels = dataset.labels
    picks = pick_indices(labels)
    predictor = Predictor(args.checkpoint)

    rows = []
    detail = []
    for index, kind in picks:
        sample = dataset[index]
        pil = sample["image"].convert("RGB")
        name = f"{kind}_{index:06d}.png"
        upscaled = pil.resize((args.size, args.size))
        upscaled.save(out / "images" / name)

        cam = predictor.predict_with_cam(pil, top_k=1)[0]
        cam["overlay"].resize((args.size, args.size)).save(out / "gradcam" / name)

        findings = predictor.predict(pil, top_k=14)
        truth = [NIH_LABELS[i] for i in np.where(sample["target"].numpy() == 1)[0]] or ["Normal"]
        top3 = findings[:3]

        rows.append(
            {
                "file": f"images/{name}",
                "kind": kind,
                "ground_truth": "|".join(truth),
                "pred_top1": top3[0]["label"],
                "prob_top1": f"{top3[0]['probability']:.3f}",
                "pred_top2": top3[1]["label"],
                "prob_top2": f"{top3[1]['probability']:.3f}",
                "pred_top3": top3[2]["label"],
                "prob_top3": f"{top3[2]['probability']:.3f}",
                "top1_in_truth": int(top3[0]["label"] in truth),
            }
        )
        detail.append(
            {
                "file": f"images/{name}",
                "ground_truth": truth,
                "probabilities": {f["label"]: round(f["probability"], 4) for f in findings},
            }
        )
        print(f"{name:26s} truth={truth} -> {top3[0]['label']} {top3[0]['probability']:.2f}")

    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out / "predictions.json").write_text(
        json.dumps(detail, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {len(rows)} samples to {out}/ (images, gradcam, predictions.csv/json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
