"""Boxes, blobs, and the filters between them.

The detector branch contributes exactly one thing to the pixel branch: a binary
foreground mask, the union of the proposals that survived the prior. It never
contributes a score, a mask or a label to the output. That asymmetry is the
architecture's defining property and its main weakness — with
``require_foreground`` on, a pixel outside every proposal cannot be reported at
all, so a frame where the detector returns nothing is an unrecoverable miss no
matter how clean the pixel branch's score is. On RoadAnomaly that accounted for
10 of 19 failing frames in the prototype.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Box:
    """Absolute xyxy, right-exclusive."""

    x1: int
    y1: int
    x2: int
    y2: int

    def clip(self, width: int, height: int) -> "Box":
        return Box(
            x1=max(0, min(int(self.x1), width)),
            y1=max(0, min(int(self.y1), height)),
            x2=max(0, min(int(self.x2), width)),
            y2=max(0, min(int(self.y2), height)),
        )

    def pad(self, ratio: float) -> "Box":
        pad_x = int((self.x2 - self.x1) * ratio)
        pad_y = int((self.y2 - self.y1) * ratio)
        return Box(self.x1 - pad_x, self.y1 - pad_y, self.x2 + pad_x, self.y2 + pad_y)

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def intersection(self, other: "Box") -> int:
        x1, y1 = max(self.x1, other.x1), max(self.y1, other.y1)
        x2, y2 = min(self.x2, other.x2), min(self.y2, other.y2)
        return max(0, x2 - x1) * max(0, y2 - y1)

    def iou(self, other: "Box") -> float:
        overlap = self.intersection(other)
        union = self.area + other.area - overlap
        return overlap / union if union else 0.0

    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class Detection:
    """One proposal from the open-vocabulary detector."""

    box: Box
    label: str
    confidence: float


@dataclass
class Instance:
    """One reported anomaly: a mask, its box, and the score behind it."""

    box: Box
    mask: np.ndarray
    confidence: float
    dino_label: Optional[str] = None
    refined_by_sam: bool = False

    def to_dict(self) -> dict:
        return {
            "bbox": self.box.as_list(),
            "area": int(np.asarray(self.mask, dtype=bool).sum()),
            "confidence": round(float(self.confidence), 4),
            "dino_label": self.dino_label,
            "refined_by_sam": self.refined_by_sam,
        }


def boxes_to_mask(boxes: Sequence[Box], height: int, width: int) -> np.ndarray:
    """Union of filled rectangles. Rectangles, not masks -- the detector has none."""
    mask = np.zeros((height, width), dtype=bool)
    for box in boxes:
        clipped = box.clip(width, height)
        mask[clipped.y1 : clipped.y2, clipped.x1 : clipped.x2] = True
    return mask


def select_top_k(detections: Sequence[Detection], k: int) -> list[Detection]:
    """The k strongest proposals in this frame.

    An absolute box threshold cannot serve both a frame whose best object scores
    0.18 and one whose noise scores 0.32; rank within the frame transfers where
    the raw score does not. ``k <= 0`` disables it, which is the shipped setting.
    """
    if k <= 0 or len(detections) <= k:
        return list(detections)
    return sorted(detections, key=lambda item: item.confidence, reverse=True)[:k]


def filter_boxes_by_prior(
    detections: Sequence[Detection], prior: np.ndarray, threshold: float
) -> list[Detection]:
    """Keep proposals whose mean OUAFS prior clears ``threshold``.

    This is where the pixel branch feeds the detector branch, before the
    detector branch feeds back. Note the prototype's sweep found the parameter
    saturates below 0.3 -- 0.25, 0.2 and 0.1 all gave the same IoU -- so at the
    shipped setting this filter is close to inert.
    """
    if threshold <= 0:
        return list(detections)
    height, width = prior.shape
    kept: list[Detection] = []
    for item in detections:
        clipped = item.box.clip(width, height)
        region = prior[clipped.y1 : clipped.y2, clipped.x1 : clipped.x2]
        if region.size and float(region.mean()) >= float(threshold):
            kept.append(item)
    return kept


def overlapping_label(box: Box, detections: Sequence[Detection], min_iou: float = 0.1) -> Optional[str]:
    """The label of the best-overlapping proposal, for reading the output only."""
    best, best_iou = None, min_iou
    for item in detections:
        score = box.iou(item.box)
        if score >= best_iou:
            best, best_iou = item.label, score
    return best


def connected_instances(
    anomaly: np.ndarray,
    score_norm: np.ndarray,
    *,
    min_blob_area: int = 200,
    max_blob_area_frac: float = 0.4,
    detections: Sequence[Detection] = (),
    foreground: Optional[np.ndarray] = None,
    min_foreground_overlap: float = 0.0,
) -> list[Instance]:
    """Connected components of the decision mask, filtered by area and overlap."""
    import cv2

    height, width = anomaly.shape
    max_area = max(min_blob_area, int(height * width * max_blob_area_frac))
    count, components = cv2.connectedComponents(anomaly.astype(np.uint8))

    instances: list[Instance] = []
    for index in range(1, count):
        mask = components == index
        area = int(mask.sum())
        if area < min_blob_area or area > max_area:
            continue
        if foreground is not None and min_foreground_overlap > 0:
            inside = float(np.logical_and(mask, foreground).sum()) / max(area, 1)
            if inside < min_foreground_overlap:
                continue
        ys, xs = np.where(mask)
        box = Box(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        instances.append(
            Instance(
                box=box,
                mask=mask,
                confidence=float(score_norm[mask].mean()),
                dino_label=overlapping_label(box, detections),
            )
        )
    instances.sort(key=lambda item: item.confidence, reverse=True)
    return instances


def suppress_nested(instances: Sequence[Instance], containment: float = 0.8) -> list[Instance]:
    """Drop a blob sitting inside a larger one.

    Thresholded blobs split one object wherever its own texture dips, so a
    boulder arrives as a large blob plus specks inside it. The container wins. A
    genuinely separate small object resting on a large one is lost by design;
    that is the trade.
    """
    if len(instances) < 2 or containment <= 0:
        return list(instances)
    kept: list[Instance] = []
    for item in sorted(instances, key=lambda entry: entry.box.area, reverse=True):
        if item.box.area <= 0:
            continue
        if any(item.box.intersection(other.box) / item.box.area >= containment for other in kept):
            continue
        kept.append(item)
    kept.sort(key=lambda entry: entry.confidence, reverse=True)
    return kept


def refine_with_sam(
    image_rgb: np.ndarray,
    instances: Sequence[Instance],
    segmenter,
    *,
    box_pad: float = 0.05,
    min_area_frac: float = 0.05,
    max_area_frac: float = 0.98,
) -> list[Instance]:
    """Replace blob interiors with SAM masks, keeping the blob when SAM degenerates.

    The dual threshold gives a reliable box and a ragged interior; SAM gives the
    true edge. A mask that collapses to nothing or floods its whole box is
    rejected rather than trusted.

    Worth knowing before enabling it: measured on RoadAnomaly in the prototype
    this cost 11.7 F1 overall. It sharpens a lone obstacle and destroys a herd,
    because one connected component is one box is one prompt, and SAM returns
    one sheep out of eight.
    """
    if not instances:
        return list(instances)
    height, width = image_rgb.shape[:2]
    boxes = [item.box.pad(box_pad).clip(width, height) for item in instances]
    masks = segmenter.segment(image_rgb, boxes)

    refined: list[Instance] = []
    for item, box, mask in zip(instances, boxes, masks):
        mask = np.asarray(mask, dtype=bool)
        fraction = float(mask.sum()) / float(max(1, box.area))
        if fraction < min_area_frac or fraction > max_area_frac:
            refined.append(item)
            continue
        ys, xs = np.where(mask)
        refined.append(
            Instance(
                box=Box(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                mask=mask,
                confidence=item.confidence,
                dino_label=item.dino_label,
                refined_by_sam=True,
            )
        )
    return refined


def rasterize(instances: Sequence[Instance], height: int, width: int) -> np.ndarray:
    """Instances back to a dense map: each pixel takes its best instance's confidence.

    This is what makes an instance-producing method comparable with a dense one
    under the same pixel metrics. It is a faithful rendering of what the
    pipeline asserts, not a re-scoring: background is a hard zero, so every
    reported pixel outranks every unreported one and the ranking metrics
    measure the pipeline's actual decisions.
    """
    dense = np.zeros((height, width), dtype=np.float32)
    for item in instances:
        mask = np.asarray(item.mask, dtype=bool)
        np.maximum(dense, np.where(mask, np.float32(item.confidence), np.float32(0.0)), out=dense)
    return dense
