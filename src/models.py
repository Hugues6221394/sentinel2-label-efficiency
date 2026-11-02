"""Models compared in the study: a spectral random-forest baseline and ResNet-18 variants."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from torchvision.models import ResNet18_Weights, resnet18

from data import RGB

B02, B03, B04, B08, B11 = 1, 2, 3, 7, 11


def _index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + 1e-6)


def spectral_features(x: np.ndarray, rgb_only: bool = False) -> np.ndarray:
    """Hand-crafted per-image features: band statistics plus vegetation/water/built-up indices.

    x: (N, 13, 64, 64) uint16 reflectance. With rgb_only, only B04/B03/B02 statistics are used,
    which is what a colour-imagery pipeline could compute.
    """
    xf = x.astype(np.float32)
    bands = xf[:, RGB] if rgb_only else xf
    flat = bands.reshape(len(bands), bands.shape[1], -1)
    feats = [flat.mean(-1), flat.std(-1), np.percentile(flat, 10, axis=-1), np.percentile(flat, 90, axis=-1)]
    if not rgb_only:
        for idx in (_index(xf[:, B08], xf[:, B04]),   # NDVI: vegetation
                    _index(xf[:, B03], xf[:, B08]),   # NDWI: water
                    _index(xf[:, B11], xf[:, B08])):  # NDBI: built-up
            v = idx.reshape(len(idx), -1)
            feats += [v.mean(-1, keepdims=True), v.std(-1, keepdims=True)]
    return np.concatenate(feats, axis=1)


def random_forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)


def resnet18_for(in_channels: int, pretrained: bool, n_classes: int = 10) -> nn.Module:
    """ResNet-18 for 64x64 Sentinel-2 patches.

    For 13-band input with ImageNet weights, the first convolution copies the pretrained RGB filters
    onto B04/B03/B02 and initialises every other band with their mean, rescaled so the summed
    activation keeps the pretrained magnitude.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    if in_channels != 3:
        old = model.conv1
        new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        if pretrained:
            with torch.no_grad():
                w = old.weight
                mean_w = w.mean(dim=1, keepdim=True)
                neww = mean_w.repeat(1, in_channels, 1, 1)
                for rgb_pos, band in enumerate(RGB):
                    neww[:, band] = w[:, rgb_pos]
                new.weight.copy_(neww * 3.0 / in_channels)
        model.conv1 = new
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model
