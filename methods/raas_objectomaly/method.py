"""RAAS refined by Objectomaly, wired into the shared CLI.

Two stages, one process: RAAS produces a coarse map, SAM-guided refinement
sharpens its boundaries. The original split these across two conda environments
(python 3.8 / torch 1.9 for RAAS, 3.10 / 2.1 for SAM) and passed float32 maps
between them through a bespoke `.npy` cache with its own manifest schema.

That cache is not ported. Its job was to move one array between two processes,
and the skeleton already does exactly that through `score_raw/*.npy`. The
environment split itself was a consequence of old pins rather than of the
architecture — detectron2 builds against modern torch — so this runs as one
scorer. If a single environment proves impossible on some machine, the fallback
is to run `raas` on its own and feed its `score_raw/` here; that path is
documented in the README but not implemented, because implementing it before
anyone needs it would be guessing.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from open_road.method import MethodSpec

from . import clip_filter, road_aware, soft_mask

VARIANTS = ("maskomaly", "maskomaly_id", "maskomaly_ood")


class RAASObjectomalyScorer:
    """Frame in (BGR), boundary-refined anomaly map out."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        from open_road.device import resolve_device

        from .backbone import QueryPredictor
        from .refine import Refiner

        settings = dict(config or {})
        self.variant = str(settings.get("variant", "maskomaly"))
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")

        self.device = resolve_device(str(settings.get("device", "auto")))
        self.save_maps = bool(settings.get("save_maps", True))
        self.refine_enabled = bool(settings.get("refine", True))

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
        self.refiner = Refiner(settings, device=self.device) if self.refine_enabled else None
        self.last_debug: dict[str, Any] | None = None

    def coarse(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray | None]:
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
        decisions: list[dict[str, Any]] = []
        if self._clip is not None:
            holes = road_aware.road_holes(mask_pred, height, width, query=self.road_query)
            for component, box in road_aware.candidate_components(holes):
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
                    }
                )
        return np.clip(score, 0.0, 1.0).astype(np.float32), semantic, decisions, holes

    def score(self, image_bgr: np.ndarray) -> np.ndarray:
        coarse, semantic, decisions, holes = self.coarse(image_bgr)

        debug: dict[str, Any] = {"01_coarse": coarse.copy(), "semantic": semantic}
        if holes is not None:
            debug["02_road_holes"] = holes

        refined = coarse
        n_masks = 0
        if self.refiner is not None:
            refined, bundle = self.refiner(coarse, image_bgr)
            n_masks = int(getattr(bundle, "n", 0))
            debug["03_refined"] = refined.copy()
            # The interesting picture: where refinement disagreed with RAAS.
            debug["04_delta"] = refined - coarse

        if self.save_maps:
            self.last_debug = {
                **debug,
                "stages": {
                    "variant": self.variant,
                    "refined": self.refiner is not None,
                    "sam_masks": n_masks,
                    "candidates": len(decisions),
                    "decisions": decisions,
                },
            }
        else:
            self.last_debug = None

        return refined


class _CLIP:
    """CLIP ViT-B/32 over candidate patches, with the variant's decision rule."""

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
    name="raas_objectomaly",
    description="RAAS coarse map sharpened by SAM-guided Objectomaly refinement (OASC + MBP)",
    build=RAASObjectomalyScorer,
    score_range=(0.0, 1.0),
    default_threshold=0.5,
    default_min_area=0,
    defaults={"variant": "maskomaly", "device": "auto", "refine": True},
)
