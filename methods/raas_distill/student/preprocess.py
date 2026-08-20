"""Frame in, normalised tensor out.

The same resize-and-normalise appeared four times in this method's original
form -- in the inference entry point, the metric script, the profiler and the
evaluation stage -- which is three chances for them to drift apart and score
the model on slightly different inputs. It lives here once instead.

``short_side`` is the one knob that matters, and the two SMIYC tracks pull in
opposite directions: anomalies on AnomalyTrack are large, so downscaling removes
noise and helps, while obstacles are tens of pixels across and downscaling
destroys them. 1024 is best for small obstacles, 736 is the real-time point, and
below 544 obstacles start disappearing.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SIZE_MULTIPLE = 32
"""SegFormer's patch grid; both sides are rounded up to a multiple of it."""

SEMANTIC_STRIDE = 4
"""The decode head predicts at 1/4 of the input resolution."""


def target_size(height: int, width: int, short_side: int) -> tuple[int, int]:
    """The size ``short_side`` implies, rounded up to the patch grid."""
    scale = short_side / min(height, width)
    return (
        int(np.ceil(height * scale / SIZE_MULTIPLE) * SIZE_MULTIPLE),
        int(np.ceil(width * scale / SIZE_MULTIPLE) * SIZE_MULTIPLE),
    )


def normalise(image_rgb: np.ndarray) -> np.ndarray:
    """ImageNet statistics, on an HWC uint8 RGB array."""
    return (image_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def to_tensor(
    image_bgr: np.ndarray,
    short_side: int,
    device: torch.device | str = "cpu",
    half: bool = False,
) -> torch.Tensor:
    """A BGR frame as a normalised ``[1, 3, H, W]`` tensor on ``device``."""
    height, width = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb, target_size(height, width, short_side)[::-1], interpolation=cv2.INTER_LINEAR
    )
    array = normalise(resized).transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(array))[None].to(
        device, dtype=torch.half if half else torch.float
    )
