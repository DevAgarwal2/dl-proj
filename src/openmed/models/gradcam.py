"""Dependency-light Grad-CAM for any CNN classifier.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def find_last_conv(model: torch.nn.Module) -> torch.nn.Module:
    """Return the last ``Conv2d`` module in the model (typical CAM target)."""
    last = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last = module
    if last is None:
        raise ValueError("No Conv2d layer found; pass target_layer explicitly.")
    return last


class GradCAM:
    """Computes a class activation map for a single image.

    Usage:
        cam = GradCAM(model, find_last_conv(model))
        heatmap, probs = cam(tensor, class_idx=5)
        cam.remove()
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inputs, output):
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    @torch.enable_grad()
    def __call__(
        self, image: torch.Tensor, class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        self.model.eval()
        image = image.requires_grad_(True)

        logits = self.model(image)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

        if class_idx is None:
            class_idx = int(np.argmax(probs))
        score = logits[0, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward()

        activations = self._activations[0]  # (C, h, w)
        gradients = self._gradients[0]  # (C, h, w)
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        cam = torch.relu((weights * activations).sum(dim=0))  # (h, w)

        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()

        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam.astype(np.float32), probs.astype(np.float32)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.remove()


def overlay_cam(
    image,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
):
    """Blend a CAM heatmap over a PIL image. Returns a PIL RGB image."""
    from PIL import Image
    from matplotlib import colormaps

    base = image.convert("RGB") if hasattr(image, "convert") else Image.fromarray(np.asarray(image))
    base = base.resize((cam.shape[1], cam.shape[0]))
    base_arr = np.asarray(base).astype(np.float32) / 255.0

    heat = colormaps[colormap](cam)[..., :3].astype(np.float32)
    blended = (1.0 - alpha) * base_arr + alpha * heat
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)
