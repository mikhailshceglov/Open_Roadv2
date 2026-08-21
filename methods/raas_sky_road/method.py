"""RAAS + Objectomaly + semantic fusion, wired into the shared CLI.

One stage longer than `raas_objectomaly`, and the stage goes in the middle:

    RAAS ──► OASC ──► global fusion ──► MBP ──► refined

Between calibration and boundary sharpening is the only place it fits. OASC must
run first because fusion measures regions against a calibrated score; MBP must
run last because it rewrites a thin band around every boundary and would smear
the restorations fusion just made.

Fusion needs the Cityscapes semantic map, which the backbone already returns
from the same forward pass that produces the query masks — the source branch
added a hook for exactly this, to avoid running a second backbone.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from open_road.method import MethodSpec

from . import clip_filter, road_aware, soft_mask

VARIANTS = ("maskomaly", "maskomaly_id", "maskomaly_ood")


class RAASSkyRoadScorer:
    """Frame in (BGR), semantically-fused anomaly map out."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        from open_road.device import resolve_device

        from .backbone import QueryPredictor
        from .clip_regions import build_validator
        from .refine import Refiner

        settings = dict(config or {})
        self.variant = str(settings.get("variant", "maskomaly"))
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")

        self.device = resolve_device(str(settings.get("device", "auto")))
        self.save_maps = bool(settings.get("save_maps", True))
        self.settings = settings

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

        self.refiner = Refiner(settings, device=self.device)
        self.fusion_config = dict(settings.get("global_fusion", {}))
        self.fusion_enabled = bool(self.fusion_config.get("enabled", True))
        self._validator = (
            build_validator(self.fusion_config, self.device) if self.fusion_enabled else None
        )
        self.last_debug: dict[str, Any] | None = None

    def coarse(self, image_bgr: np.ndarray):
        """The RAAS half: soft mask, then the optional road-hole CLIP pass."""
        import cv2

        height, width = image_bgr.shape[:2]
        mask_cls, mask_pred, semantic = self.predictor(image_bgr)

        blob = soft_mask.soft_mask(
            mask_cls, mask_pred,
            anomaly_queries=self.anomaly_queries,
            rejection_weight=self.rejection_weight,
        )
        score = cv2.resize(blob, (width, height), interpolation=cv2.INTER_AREA)

        holes = None
        if self._clip is not None:
            holes = road_aware.road_holes(mask_pred, height, width, query=self.road_query)
            for component, box in road_aware.candidate_components(holes):
                x, y, w, h = box
                patch = image_bgr[y : y + h, x : x + w]
                if patch.size:
                    clip_filter.apply_decision(
                        score, component, box, self._clip.decide(patch)
                    )
        return np.clip(score, 0.0, 1.0).astype(np.float32), semantic, holes

    def score(self, image_bgr: np.ndarray) -> np.ndarray:
        from .fusion import apply_global_fusion

        coarse, semantic, holes = self.coarse(image_bgr)

        bundle = self.refiner.masks(image_bgr)
        calibrated = self.refiner.calibrate(coarse, bundle)

        debug: dict[str, Any] = {
            "01_coarse": coarse.copy(),
            "02_calibrated": np.asarray(calibrated, dtype=np.float32).copy(),
            "semantic": semantic,
        }
        if holes is not None:
            debug["03_road_holes"] = holes

        diagnostics: dict[str, Any] = {"enabled": False}
        fused = calibrated
        if self.fusion_enabled:
            fused, protected, diagnostics = apply_global_fusion(
                calibrated, semantic, image_bgr, bundle,
                self.fusion_config, self._validator,
            )
            diagnostics["enabled"] = True
            debug["04_fused"] = fused.copy()
            debug["05_protected"] = protected
            # What the semantic attenuation actually cost, before MBP hides it.
            debug["06_fusion_delta"] = fused - np.asarray(calibrated, dtype=np.float32)

        refined = self.refiner.sharpen(fused, image_bgr, bundle)
        refined = np.clip(np.asarray(refined, dtype=np.float32), 0.0, 1.0)
        debug["07_refined"] = refined.copy()
        debug["08_total_delta"] = refined - coarse

        if self.save_maps:
            self.last_debug = {
                **debug,
                "stages": {
                    "variant": self.variant,
                    "sam_masks": int(getattr(bundle, "n", 0)),
                    "fusion": diagnostics,
                },
            }
        else:
            self.last_debug = None
        return refined


class _CLIP:
    """CLIP ViT-B/32 over road-hole patches, with the variant's decision rule."""

    def __init__(self, variant: str, device: str) -> None:
        try:
            import clip
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "openai/CLIP is required by the maskomaly_id and maskomaly_ood variants: "
                "pip install git+https://github.com/openai/CLIP.git"
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
    name="raas_sky_road",
    description="RAAS + Objectomaly with semantic class attenuation and SAM-region restoration",
    build=RAASSkyRoadScorer,
    score_range=(0.0, 1.0),
    default_threshold=0.5,
    default_min_area=0,
    defaults={"variant": "maskomaly", "device": "auto"},
)
