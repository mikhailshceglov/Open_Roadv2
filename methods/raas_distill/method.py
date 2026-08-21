"""RAAS-Distill: the released student, wired into the shared CLI.

This is the whole of the method's public surface. ``open_road.registry`` finds
it by walking ``methods/``, imports it, and reads ``METHOD``; nothing central
had to be edited to make that happen.

The heavy imports (torch, transformers) sit inside ``AnomalySegmenter`` rather
than at module level, so building the spec -- which ``open-road methods`` does
just to list it -- stays free even where this method's pins are not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from open_road.method import MethodSpec

HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = HERE / "weights" / "student_final.pt"


class AnomalySegmenter:
    """Frame in (BGR, as OpenCV reads it), anomaly probability out.

    The output is a ranking, not a calibrated probability. AUPR and FPR@95
    depend only on the ordering of pixels, and nothing in training pushed the
    values towards being read literally -- so choose a threshold on your own
    validation data rather than trusting 0.5.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        import torch

        from open_road.device import resolve_device

        from .student.model import Student

        settings = dict(config or {})
        self.short_side = int(settings.get("short_side", 1024))
        self.device = torch.device(resolve_device(str(settings.get("device", "auto"))))
        # Half precision is emulated on CPU, where it is slower rather than faster.
        self.half = bool(settings.get("half", False)) and self.device.type != "cpu"

        checkpoint = Path(settings.get("checkpoint") or DEFAULT_CHECKPOINT)
        if not checkpoint.is_absolute():
            checkpoint = HERE / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(f"no student checkpoint at {checkpoint}")

        # pretrained=False: the architecture comes from the vendored config and
        # the checkpoint replaces every weight, so the pretrained backbone would
        # only be downloaded to be overwritten. Verified under HF_HUB_OFFLINE=1.
        self.model = Student(pretrained=False).to(self.device).eval()
        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state.get("model", state))
        if self.half:
            self.model.half()

    def score(self, image_bgr: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        from .student.preprocess import to_tensor

        height, width = image_bgr.shape[:2]
        tensor = to_tensor(image_bgr, self.short_side, self.device, self.half)

        with torch.no_grad():
            _semantic, anomaly = self.model(tensor)
            # The head predicts at 1/4 scale; the caller gets its own frame back.
            anomaly = F.interpolate(
                anomaly.float(), size=(height, width), mode="bilinear", align_corners=False
            )
            return torch.sigmoid(anomaly)[0, 0].cpu().numpy().astype(np.float32)


METHOD = MethodSpec(
    name="raas_distill",
    description="SegFormer-B0 student distilled from RAAS (Mask2Former Swin-L + CLIP), 3.72M params",
    build=AnomalySegmenter,
    # A sigmoid, so the preview needs no rescaling at all.
    score_range=(0.0, 1.0),
    # Not swept on RoadAnomaly -- this method's numbers were measured on SMIYC
    # with threshold-free metrics. Treat 0.5 as a starting point and sweep it on
    # your own validation data before trusting any masks it produces.
    default_threshold=0.5,
    # Dropping small components is the single post-processing choice that moves
    # component F1 by tens of points on this benchmark. 200 px is deliberately
    # lower than RbA's 2000: this student is at its best on small obstacles.
    default_min_area=200,
    defaults={"short_side": 1024, "device": "auto", "half": False},
)
