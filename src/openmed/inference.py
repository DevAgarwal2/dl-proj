"""Inference helper shared by the CLI, the API and the RAG pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from openmed.data.labels import NIH_LABELS
from openmed.data.transforms import build_transforms
from openmed.models.classifier import load_checkpoint
from openmed.models.gradcam import GradCAM, find_last_conv, overlay_cam
from openmed.utils import get_device


class Predictor:
    """Scores chest X-ray images from a checkpoint or a bare backbone.

    Passing ``checkpoint=None`` builds an untrained/pretrained backbone with a
    random classification head. That mode is only useful for exercising the UI
    pipeline end to end; its predictions are meaningless.
    """

    def __init__(
        self,
        checkpoint: Optional[str | Path] = None,
        device: str = "auto",
        image_size: Optional[int] = None,
        mean=(0.5056, 0.5056, 0.5056),
        std=(0.252, 0.252, 0.252),
        backbone: Optional[str] = None,
        pretrained: bool = False,
        label_names: Optional[List[str]] = None,
    ) -> None:
        self.device = get_device(device)
        if checkpoint is not None and not Path(checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        self.is_trained = checkpoint is not None

        if self.is_trained:
            self.model, meta = load_checkpoint(checkpoint, device=self.device)
            self.backbone = meta.get("backbone", backbone or "densenet121")
            self.label_names = list(meta.get("label_names", label_names or NIH_LABELS))
            self.image_size = int(image_size or meta.get("image_size", 320))
            self.meta = meta
        else:
            from openmed.models.classifier import build_model

            self.backbone = backbone or "densenet121"
            self.label_names = list(label_names or NIH_LABELS)
            self.image_size = int(image_size or 320)
            self.model = build_model(
                self.backbone,
                num_classes=len(self.label_names),
                pretrained=pretrained,
            ).to(self.device)
            self.model.eval()
            self.meta = {"backbone": self.backbone, "image_size": self.image_size}

        self.transform = build_transforms(self.image_size, train=False, mean=mean, std=std)

    def _tensor(self, image):
        import torch

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        return self.transform(image).unsqueeze(0).to(self.device), image

    def predict(self, image, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[Dict]:
        """Return findings sorted by probability (descending)."""
        import torch

        tensor, _ = self._tensor(image)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(tensor))[0].cpu().numpy()
        findings = [
            {"label": name, "probability": float(probs[i])}
            for i, name in enumerate(self.label_names)
        ]
        findings.sort(key=lambda item: item["probability"], reverse=True)
        if threshold is not None:
            findings = [f for f in findings if f["probability"] >= threshold]
        if top_k is not None:
            findings = findings[:top_k]
        return findings

    def predict_with_cam(self, image, top_k: int = 3):
        """Return the top findings plus Grad-CAM overlays for each."""
        import torch

        tensor, pil = self._tensor(image)
        cam = GradCAM(self.model, find_last_conv(self.model))
        try:
            with torch.no_grad():
                probs = torch.sigmoid(self.model(tensor))[0].cpu().numpy()
            order = np.argsort(probs)[::-1][:top_k]
            results = []
            for class_idx in order:
                heatmap, _ = cam(tensor, class_idx=int(class_idx))
                results.append(
                    {
                        "label": self.label_names[int(class_idx)],
                        "probability": float(probs[int(class_idx)]),
                        "cam": heatmap,
                        "overlay": overlay_cam(pil, heatmap),
                    }
                )
            return results
        finally:
            cam.remove()
