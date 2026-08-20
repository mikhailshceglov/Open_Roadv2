"""RAAS wired into the shared CLI.

Three variants share every line of this code and differ only in what runs after
the soft mask:

``maskomaly``      the soft mask alone
``maskomaly_id``   plus road-hole candidates arbitrated by CLIP over 19 known prompts
``maskomaly_ood``  the same, with 3 open-ended OOD prompts and a margin rule

They are a config key rather than three methods because they are one pipeline
with a switch, and because that is how the original organised them.

Heavy imports live inside the scorer: building the spec must stay free, since
``open-road methods`` builds it just to print one line — and this method's
dependencies (detectron2, a patched Mask2Former, a checkpoint with no published
URL) are absent far more often than they are present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from open_road.method import MethodSpec

from . import clip_filter, road_aware, soft_mask

VARIANTS = ("maskomaly", "maskomaly_id", "maskomaly_ood")


class RAASScorer:
    """Frame in (BGR), anomaly map out, internals left in ``last_debug``."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        from open_road.device import resolve_device

        from .backbone import QueryPredictor

        settings = dict(config or {})
        self.variant = str(settings.get("variant", "maskomaly"))
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")

        self.device = resolve_device(str(settings.get("device", "auto")))
        self.save_maps = bool(settings.get("save_maps", True))

        self.anomaly_queries = soft_mask.resolve_ranking(
            settings.get("analysis_file"), int(settings.get("masks", 4))
        )
        self.road_query = int(settings.get("road_query", road_aware.ROAD_QUERY))
        self.rejection_weight = float(settings.get("rejection_weight", soft_mask.REJECTION_WEIGHT))

        self.predictor = QueryPredictor(
            config_file=settings.get("config_file"),
            checkpoint=settings.get("checkpoint"),
            device=self.device,
        )
        self._clip = None if self.variant == "maskomaly" else _CLIP(self.variant, self.device)
        self.last_debug: dict[str, Any] | None = None

    def score(self, image_bgr: np.ndarray) -> np.ndarray:
        import cv2

        height, width = image_bgr.shape[:2]
        mask_cls, mask_pred, semantic = self.predictor(image_bgr)

        coarse = soft_mask.soft_mask(
            mask_cls,
            mask_pred,
            anomaly_queries=self.anomaly_queries,
            rejection_weight=self.rejection_weight,
        )
        # INTER_AREA, as the original: the map is downsampled far more often
        # than upsampled, and area averaging is what keeps thin objects alive.
        score = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_AREA)

        debug: dict[str, Any] = {"01_soft_mask": score.copy(), "semantic": semantic}
        decisions: list[dict[str, Any]] = []

        if self._clip is not None:
            holes = road_aware.road_holes(mask_pred, height, width, query=self.road_query)
            candidates = road_aware.candidate_components(holes)
            debug["02_road_holes"] = holes
            for component, box in candidates:
                x, y, w, h = box
                patch = image_bgr[y : y + h, x : x + w]
                if patch.size == 0:
                    continue
                decision = self._clip.decide(patch)
                clip_filter.apply_decision(score, component, box, decision)
                decisions.append(
                    {
                        "bbox": [x, y, x + w, y + h],
                        "anomalous": decision.anomalous,
                        "label": decision.label,
                        "id_probability": round(decision.id_probability, 4),
                        "ood_probability": (
                            round(decision.ood_probability, 6)
                            if decision.ood_probability is not None
                            else None
                        ),
                    }
                )
            debug["03_after_clip"] = score.copy()

        if self.save_maps:
            self.last_debug = {
                **debug,
                "stages": {
                    "variant": self.variant,
                    "anomaly_queries": list(self.anomaly_queries),
                    "candidates": len(decisions),
                    "kept_as_anomaly": sum(1 for item in decisions if item["anomalous"]),
                    "decisions": decisions,
                },
            }
        else:
            self.last_debug = None

        return np.clip(score, 0.0, 1.0).astype(np.float32)


class _CLIP:
    """CLIP ViT-B/32 over candidate patches, with the variant's decision rule."""

    def __init__(self, variant: str, device: str) -> None:
        try:
            import clip
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "openai/CLIP is required by the maskomaly_id and maskomaly_ood variants: "
                "pip install git+https://github.com/openai/CLIP.git "
                "(uninstall the unrelated PyPI package named `clip` first). "
                "The `maskomaly` variant needs none of this."
            ) from error
        import torch

        self.torch = torch
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()

        self.variant = variant
        prompts = list(clip_filter.ID_PROMPTS)
        if variant == "maskomaly_ood":
            prompts += list(clip_filter.OOD_PROMPTS)
        self.text = clip.tokenize(prompts).to(device)

    def decide(self, patch_bgr: np.ndarray) -> clip_filter.Decision:
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            logits, _ = self.model(tensor, self.text)
            probabilities = logits.softmax(dim=-1).cpu().numpy().flatten()

        if self.variant == "maskomaly_ood":
            return clip_filter.decide_ood(probabilities)
        return clip_filter.decide_id(probabilities)


METHOD = MethodSpec(
    name="raas",
    description="Maskomaly: anomaly by elimination over Mask2Former queries, optionally road-aware + CLIP",
    build=RAASScorer,
    # The map is a blend of two [0, 1] maps and is clipped, so no rescaling.
    score_range=(0.0, 1.0),
    # Not swept here. The published numbers are threshold-free (AUPR, FPR@95);
    # the CLIP variants write hard 0.05 / 1.0 constants into candidate regions,
    # so almost any cut between them separates the same pixels. Sweep on your
    # own validation data before trusting a mask.
    default_threshold=0.5,
    # The original applies no minimum component size at all. Left at zero to
    # match it; raise it if the render is speckled.
    default_min_area=0,
    defaults={"variant": "maskomaly", "device": "auto"},
)
