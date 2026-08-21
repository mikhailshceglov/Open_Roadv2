"""OoDDINO wired into the shared CLI.

The awkward part of fitting this method to the skeleton is that the skeleton
asks for one float per pixel and OoDDINO's output is a *set of instances*. Three
ways of answering that are offered, and which one is chosen changes what the
metrics mean:

``instances`` (default)
    The instances rasterised back to a dense map, each pixel taking its best
    instance's confidence and background a hard zero. This is what the pipeline
    actually asserts, SAM included, so the metrics score the real output.
``probability``
    The ADT soft probability, before instance filtering and before SAM. Denser
    and better-ranked, but it is an intermediate, not the pipeline's answer.
``residual``
    The pixel branch alone -- ``energy − median(energy | class)`` -- with the
    detector, the prior, ARNS and SAM all bypassed. Useful precisely because it
    isolates the half of the architecture that does not depend on the detector
    firing.

Heavy imports stay inside the scorer: building the spec must not need torch,
because ``open-road methods`` builds it just to print one line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from open_road.method import MethodSpec

HERE = Path(__file__).resolve().parent
SCORE_SOURCES = ("instances", "probability", "residual")


class OoDDINOScorer:
    """Frame in (BGR), dense anomaly score out, internals left in ``last_debug``."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        from open_road.device import resolve_device

        from .models import GroundingDINODetector, SAM2Segmenter, SegFormerLogits, configure_cache

        settings = dict(config or {})
        self.settings = settings
        self.score_source = str(settings.get("score_source", "instances"))
        if self.score_source not in SCORE_SOURCES:
            raise ValueError(
                f"score_source must be one of {SCORE_SOURCES}, got {self.score_source!r}"
            )

        self.device = resolve_device(str(settings.get("device", "auto")))
        configure_cache(settings.get("weights_dir"))
        self.prompt = str(settings.get("prompt", "anomaly . unknown object . obstacle . debris ."))

        pixel = dict(settings.get("pixel", {}))
        self.pixel = SegFormerLogits(
            model_id=pixel.get("model_id", "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"),
            device=self.device,
        )
        self.score_kind = pixel.get("score", "class_residual")

        detector = dict(settings.get("detector", {}))
        # The residual source never consults the detector, so do not build one.
        self.detector = (
            None
            if self.score_source == "residual"
            else GroundingDINODetector(
                model_id=detector.get("model_id", "IDEA-Research/grounding-dino-base"),
                device=self.device,
                box_threshold=float(detector.get("box_threshold", 0.25)),
                text_threshold=float(detector.get("text_threshold", 0.25)),
            )
        )
        self.top_k = int(detector.get("top_k", 0))

        sam = dict(settings.get("sam", {}))
        self.sam_enabled = bool(sam.get("enabled", True)) and self.score_source != "residual"
        self.sam = (
            SAM2Segmenter(
                model_id=sam.get("model_id", "facebook/sam2.1-hiera-large"), device=self.device
            )
            if self.sam_enabled
            else None
        )
        self.sam_settings = {
            "box_pad": float(sam.get("box_pad", 0.05)),
            "min_area_frac": float(sam.get("min_area_frac", 0.05)),
            "max_area_frac": float(sam.get("max_area_frac", 0.98)),
        }

        self.ouafs = dict(settings.get("ouafs", {}))
        self.adt = dict(settings.get("adt", {}))
        self.min_confidence = float(settings.get("min_confidence", 0.6))
        self.save_maps = bool(settings.get("save_maps", True))
        self.last_debug: dict[str, Any] | None = None

    def score(self, image_bgr: np.ndarray) -> np.ndarray:
        import cv2

        from . import maths, pipeline
        from .boxes import rasterize

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        logits = self.pixel.logits(image_rgb)

        if self.score_source == "residual":
            score = maths.pixel_score(logits, kind=self.score_kind)
            self.last_debug = (
                {"score": score, "entropy": maths.entropy_map(logits)} if self.save_maps else None
            )
            return score.astype(np.float32)

        detections = self.detector.detect(image_rgb, self.prompt)
        result = pipeline.run(
            logits,
            detections,
            score_kind=self.score_kind,
            top_k=self.top_k,
            prior_threshold=float(self.ouafs.get("prior_threshold", 0.3)),
            alpha=float(self.adt.get("alpha", 0.5)),
            delta=float(self.adt.get("delta", 0.2)),
            fg_quantile=float(self.adt.get("fg_quantile", 0.2)),
            bg_quantile=float(self.adt.get("bg_quantile", 0.995)),
            min_gap=float(self.adt.get("min_gap", 0.2)),
            min_blob_area=int(self.adt.get("min_blob_area", 100)),
            max_blob_area_frac=float(self.adt.get("max_blob_area_frac", 0.4)),
            require_foreground=bool(self.ouafs.get("require_foreground", True)),
            min_foreground_overlap=float(self.ouafs.get("min_foreground_overlap", 0.5)),
        )
        result = pipeline.postprocess(
            result,
            image_rgb,
            containment=float(self.adt.get("containment", 0.8))
            if self.adt.get("suppress_nested", True)
            else 0.0,
            min_confidence=self.min_confidence,
            segmenter=self.sam,
            **self.sam_settings,
        )

        self.last_debug = self._debug(result) if self.save_maps else None

        if self.score_source == "probability":
            return result.maps.probability.astype(np.float32)
        return rasterize(result.instances, height, width)

    def _debug(self, result) -> dict[str, Any]:
        maps = result.maps
        return {
            "01_entropy": maps.entropy,
            "02_distance": maps.distance,
            "03_prior": maps.prior,
            "04_score": maps.score,
            "05_score_norm": maps.score_norm,
            "06_foreground": maps.foreground,
            "07_probability": maps.probability,
            "08_anomaly": maps.anomaly,
            "stages": {
                "detections": len(result.detections),
                "kept_by_prior": len(result.kept),
                "instances": len(result.instances),
                "t_fg": round(maps.t_fg, 4),
                "t_bg": round(maps.t_bg, 4),
                # Empty when the frame produced something; otherwise names the
                # stage that emptied it, which is the question you actually ask
                # when a frame scores zero.
                "diagnosis": result.diagnosis,
                "boxes": [
                    {"bbox": item.box.as_list(), "label": item.label,
                     "confidence": round(item.confidence, 4)}
                    for item in result.detections
                ],
                "instances_detail": [item.to_dict() for item in result.instances],
            },
        }


METHOD = MethodSpec(
    name="ooddino",
    description="GroundingDINO proposals + SegFormer logits, fused by OUAFS/ARNS/ADT, refined by SAM 2.1",
    build=OoDDINOScorer,
    # Instance confidences are ARNS means, which live in [alpha, alpha + 0.5] =
    # [0.5, 1.0]; background rasterises to 0. The `residual` source is unbounded
    # and its preview PNG will clip -- the .npy beside it does not.
    score_range=(0.0, 1.0),
    # Every reported pixel carries a confidence of at least min_confidence
    # (0.6), and background is exactly 0, so this only separates reported from
    # unreported. The decision itself was made by the dual threshold upstream.
    default_threshold=0.5,
    # Matches the pipeline's own min_blob_area, so render and eval agree with
    # what the method already discarded.
    default_min_area=100,
    defaults={"score_source": "instances", "device": "auto"},
)
