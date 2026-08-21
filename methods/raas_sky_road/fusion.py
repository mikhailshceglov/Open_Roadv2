"""Semantic fusion: attenuate by what a pixel is, then win the exception back.

The observation behind this stage is that a road-scene anomaly detector wastes
most of its false positives on classes that are never the answer. Sky is
featureless and uncertain, vegetation is high-frequency noise, building facades
are cluttered — all three light up a residual-energy map and none of them is an
obstacle in the road.

So each pixel's score is multiplied by a factor keyed on its predicted
Cityscapes class: sky 0.35, vegetation and building 0.55, road 1.0. Nothing is
hard-masked, and that matters — a hard mask cannot be undone, and the whole
point of the branch this came from is that an object *can* be in the sky. A
falling tyre, cargo off a truck, debris in the air: all of them sit on sky
pixels and would be erased by a mask.

The exception mechanism is the second half. SAM proposes compact regions; each
is measured against its own surroundings, and a region that looks like an object
gets its **original, un-attenuated score restored**:

    fused = clip( f[class] · score,                          unprotected
                  max(f[class] · score, score),              protected
                  max(…, 0.65),                              protected by CLIP
                  0, 1 )

Three independent things can protect a region — strong contrast against its
ring, a non-background dominant class, or CLIP preferring an OOD prompt — and
any one of them suffices.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

CITYSCAPES_CLASS_NAMES = (
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle",
)


def sigmoid(value: float) -> float:
    """Clamped before exponentiating, so a large margin cannot overflow."""
    return float(1.0 / (1.0 + np.exp(-float(np.clip(value, -60.0, 60.0)))))


def ring_median(score: np.ndarray, mask: np.ndarray, radius: int) -> float:
    """Median score in an elliptical ring just outside ``mask``.

    This is the region's local background. Comparing a region against its own
    surroundings rather than against the whole frame is what makes a small
    bright object on dark tarmac and a small bright object against bright sky
    score alike.
    """
    import cv2

    size = 2 * max(1, int(radius)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = np.logical_and(dilated, ~mask)
    if not ring.any():
        return float(np.median(score))
    return float(np.median(score[ring]))


def dominant_class(semantic: np.ndarray, mask: np.ndarray) -> int:
    """The most common Cityscapes class inside ``mask``; -1 when it is empty."""
    values = semantic[mask]
    if not values.size:
        return -1
    return int(np.bincount(values.astype(np.int64)).argmax())


def class_name(index: int) -> str:
    if 0 <= index < len(CITYSCAPES_CLASS_NAMES):
        return CITYSCAPES_CLASS_NAMES[index]
    return "unknown"


def attenuate(score: np.ndarray, semantic: np.ndarray,
              class_factors: Mapping[Any, Any]) -> np.ndarray:
    """Scale each pixel by its class's factor. Classes not listed keep 1.0."""
    factors = np.ones(score.shape, dtype=np.float32)
    for class_id, factor in class_factors.items():
        factors[semantic == int(class_id)] = np.float32(factor)
    return score * factors


def boost_road(fused: np.ndarray, score: np.ndarray, semantic: np.ndarray,
               road_ids: Sequence[int], boost: float) -> np.ndarray:
    """Lift road pixels, which are where an obstacle actually matters.

    A no-op unless ``boost > 1``. Note it lifts against the *original* score,
    not the attenuated one, so it cannot be cancelled by a small class factor.
    """
    if boost <= 1.0:
        return fused
    road = np.isin(semantic, [int(value) for value in road_ids])
    fused = fused.copy()
    fused[road] = np.maximum(fused[road], score[road] * float(boost))
    return fused


def describe_candidates(
    score: np.ndarray,
    semantic: np.ndarray,
    bundle,
    *,
    min_area_frac: float = 0.0,
    max_area_frac: float = 0.03,
    quantile: float = 0.9,
    ring_radius: int = 9,
    max_candidates: int = 64,
) -> list[dict[str, Any]]:
    """Measure every SAM region against its surroundings, best first.

    ``score`` is the pre-attenuation map on purpose: a region is judged on what
    the detector originally believed, not on what the class factor left of it.
    Otherwise a genuine object in the sky could never argue its way back.
    """
    image_area = float(score.size)
    records: list[dict[str, Any]] = []
    for index in range(bundle.n):
        mask = bundle.masks[index]
        fraction = float(bundle.area[index]) / image_area if image_area else 0.0
        if not min_area_frac <= fraction <= max_area_frac or not mask.any():
            continue
        region_score = float(np.quantile(score[mask], quantile))
        ring = ring_median(score, mask, ring_radius)
        best = dominant_class(semantic, mask)
        records.append(
            {
                "index": int(index),
                "bbox": [int(value) for value in bundle.bbox[index]],
                "area_fraction": fraction,
                "quality": float(bundle.quality[index]),
                "dominant_class": best,
                "dominant_class_name": class_name(best),
                "score": region_score,
                "ring_score": ring,
                "contrast": region_score - ring,
            }
        )
    # Rank by how anomalous *and* how much it stands out, weighted by how much
    # SAM trusts the mask. Negative contrast must not reward a region, hence max.
    records.sort(
        key=lambda item: (item["score"] + max(0.0, item["contrast"])) * item["quality"],
        reverse=True,
    )
    return records[:max_candidates]


def apply_global_fusion(
    calibrated: np.ndarray,
    semantic: np.ndarray,
    image_bgr: np.ndarray,
    bundle,
    config: Mapping[str, Any],
    clip_validator: Optional[Any] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Attenuate by class, then restore the regions that argue their way back.

    Returns the fused map, the protected-region mask, and diagnostics carrying
    every candidate's measurements and the reasons it was or was not protected.
    """
    score = np.asarray(calibrated, dtype=np.float32)
    semantic = np.asarray(semantic, dtype=np.int16)
    if score.ndim != 2 or semantic.shape != score.shape:
        raise ValueError(
            f"Global fusion shape mismatch: score={score.shape}, semantic={semantic.shape}"
        )

    fused = attenuate(score, semantic, config.get("class_factors", {}))
    fused = boost_road(
        fused, score, semantic,
        config.get("road_class_ids", [0]),
        float(config.get("road_boost", 1.0)),
    )

    candidates = dict(config.get("candidates", {}))
    background_ids = {int(v) for v in config.get("background_class_ids", [])}
    min_score = float(candidates.get("min_score", 0.45))
    min_contrast = float(candidates.get("min_contrast", 0.12))
    probe_min_score = float(candidates.get("clip_probe_min_score", 0.2))
    clip_margin_threshold = float(candidates.get("clip_margin_threshold", 0.0))
    clip_floor = float(candidates.get("clip_candidate_floor", 0.65))

    records = describe_candidates(
        score, semantic, bundle,
        min_area_frac=float(candidates.get("min_area_frac", 0.0)),
        max_area_frac=float(candidates.get("max_area_frac", 0.03)),
        quantile=float(candidates.get("score_quantile", 0.9)),
        ring_radius=int(candidates.get("ring_radius", 9)),
        max_candidates=int(candidates.get("max_candidates", 64)),
    )

    probe = [item["index"] for item in records if item["score"] >= probe_min_score]
    clip_scores = (
        clip_validator.score_regions(image_bgr, bundle, probe)
        if clip_validator is not None and probe
        else {}
    )

    protected_mask = np.zeros(score.shape, dtype=bool)
    protected_count = 0
    for item in records:
        item.update(clip_scores.get(item["index"], {}))
        geometric = item["score"] >= min_score and item["contrast"] >= min_contrast
        semantic_object = (
            item["score"] >= min_score and item["dominant_class"] not in background_ids
        )
        clip_ood = item.get("clip_margin", float("-inf")) >= clip_margin_threshold

        item["protected"] = bool(geometric or semantic_object or clip_ood)
        item["reasons"] = [
            name for name, on in (
                ("contrast", geometric),
                ("semantic_object", semantic_object),
                ("clip_ood", clip_ood),
            ) if on
        ]
        if not item["protected"]:
            continue

        mask = bundle.masks[item["index"]]
        protected_mask |= mask
        # Undo the class attenuation for this region only.
        fused[mask] = np.maximum(fused[mask], score[mask])
        if clip_ood:
            fused[mask] = np.maximum(fused[mask], clip_floor)
        protected_count += 1

    diagnostics = {
        "candidate_count": len(records),
        "clip_probed_count": len(probe) if clip_validator is not None else 0,
        "protected_count": protected_count,
        "candidates": records,
    }
    return (
        np.ascontiguousarray(np.clip(fused, 0.0, 1.0), dtype=np.float32),
        np.ascontiguousarray(protected_mask),
        diagnostics,
    )
