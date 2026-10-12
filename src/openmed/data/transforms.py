"""Image transforms for chest X-ray classification."""

from __future__ import annotations

from typing import Sequence


def build_transforms(
    image_size: int,
    train: bool,
    mean: Sequence[float] = (0.5056, 0.5056, 0.5056),
    std: Sequence[float] = (0.252, 0.252, 0.252),
):
    """Return a torchvision transform pipeline.

    Imports torchvision lazily so the rest of OpenMed works without it.
    """
    from torchvision import transforms as T

    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=7),
                T.ColorJitter(brightness=0.10, contrast=0.10),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    resize = int(round(image_size * 1.14))
    return T.Compose(
        [
            T.Resize(resize),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
