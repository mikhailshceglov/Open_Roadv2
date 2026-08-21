"""The cascade: pixel branch, detector branch, and the two couplings between them.

    image
      ├─► SegFormer-B2 ──► logits ──┬─► entropy  E ──┐
      │                             ├─► distance D ──┼─► OUAFS prior P
      │                             └─► score    I = energy − median(energy | class)
      │                                                │
      └─► GroundingDINO ──► proposals ─────────────────┘  filtered by mean P inside the box
                                 │
                                 ▼
                       foreground = union of surviving boxes
                                 │
              I ──► ARNS(I, fg) ──► I_norm ──► dual thresholds T_fg / T_bg
                                 │                     │
                                 └──► ADT ramp ──► anomaly = P ≥ 0.5
                                              │
                              connected components → suppress nested
                                              │
                                     SAM 2.1 (box prompt) → instances

The two branches are not symmetric. The pixel branch does all of the arithmetic
and produces every reported pixel; the detector branch produces only a binary
mask that (a) chooses which region gets the lenient threshold and (b) gates
which blobs may be reported at all. The pixel branch also feeds the detector
branch first, through the prior that filters its proposals — so this is a
cascade with a loop, not two independent paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from . import maths
from .boxes import (
    Detection,
    Instance,
    boxes_to_mask,
    connected_instances,
    filter_boxes_by_prior,
    refine_with_sam,
    select_top_k,
    suppress_nested,
)


@dataclass
class Maps:
    """Every dense intermediate, kept so a failure can be looked at."""

    labels: np.ndarray
    entropy: np.ndarray
    distance: np.ndarray
    prior: np.ndarray
    score: np.ndarray
    score_norm: np.ndarray
    foreground: np.ndarray
    probability: np.ndarray
    anomaly: np.ndarray
    t_fg: float
    t_bg: float


@dataclass
class Result:
    maps: Maps
    detections: list[Detection]
    kept: list[Detection]
    instances: list[Instance]

    @property
    def diagnosis(self) -> str:
        """Why this frame produced nothing, in one word. Empty when it did."""
        if self.instances:
            return ""
        if not self.detections:
            return "empty_detector"
        if not self.kept:
            return "prior_rejected_all_boxes"
        return "threshold_or_area_killed_all_blobs"


def run(
    logits: np.ndarray,
    detections: Sequence[Detection],
    *,
    score_kind: str = "class_residual",
    top_k: int = 0,
    prior_threshold: float = 0.3,
    alpha: float = 0.5,
    delta: float = 0.2,
    fg_quantile: float = 0.2,
    bg_quantile: float = 0.995,
    min_gap: float = 0.2,
    min_blob_area: int = 100,
    max_blob_area_frac: float = 0.4,
    require_foreground: bool = True,
    min_foreground_overlap: float = 0.5,
) -> Result:
    """One frame, from logits and proposals to instances."""
    height, width = logits.shape[1], logits.shape[2]

    labels = np.argmax(logits, axis=0).astype(np.int32)
    entropy = maths.entropy_map(logits)
    distance = maths.distance_map(logits)
    prior = maths.ouafs_prior(entropy, distance)

    proposals = select_top_k(detections, top_k)
    kept = filter_boxes_by_prior(proposals, prior, prior_threshold)
    foreground = boxes_to_mask([item.box for item in kept], height, width)

    score = maths.pixel_score(logits, kind=score_kind)
    score_norm = maths.arns_normalize(score, foreground, alpha=alpha)
    t_fg, t_bg = maths.dual_thresholds(
        score_norm,
        foreground,
        fg_quantile=fg_quantile,
        bg_quantile=bg_quantile,
        min_gap=min_gap,
        alpha=alpha,
    )
    probability = maths.adt_probability(score_norm, foreground, t_fg, t_bg, delta=delta)
    anomaly = probability >= 0.5

    instances = connected_instances(
        anomaly,
        score_norm,
        min_blob_area=min_blob_area,
        max_blob_area_frac=max_blob_area_frac,
        detections=kept,
        foreground=foreground if require_foreground else None,
        min_foreground_overlap=min_foreground_overlap,
    )

    maps = Maps(
        labels=labels,
        entropy=entropy,
        distance=distance,
        prior=prior,
        score=score,
        score_norm=score_norm,
        foreground=foreground,
        probability=probability,
        anomaly=anomaly,
        t_fg=float(t_fg),
        t_bg=float(t_bg),
    )
    return Result(maps=maps, detections=list(detections), kept=kept, instances=instances)


def postprocess(
    result: Result,
    image_rgb: np.ndarray,
    *,
    containment: float = 0.8,
    min_confidence: float = 0.6,
    segmenter: Optional[object] = None,
    box_pad: float = 0.05,
    min_area_frac: float = 0.05,
    max_area_frac: float = 0.98,
) -> Result:
    """Nested suppression, the confidence cut, then SAM — in that order.

    SAM runs last so it never sees a blob that was about to be discarded, and it
    never adds or removes an instance: it only sharpens boundaries.
    """
    instances = suppress_nested(result.instances, containment=containment)
    instances = [item for item in instances if item.confidence >= min_confidence]
    if segmenter is not None and instances:
        instances = refine_with_sam(
            image_rgb,
            instances,
            segmenter,
            box_pad=box_pad,
            min_area_frac=min_area_frac,
            max_area_frac=max_area_frac,
        )
    result.instances = instances
    return result
