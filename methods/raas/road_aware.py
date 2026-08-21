"""Road-aware candidates: obstacles are the holes they punch in the road.

The road query segments drivable surface. Anything standing on the road is not
road, so it appears as a *hole* in that mask — while the road's outer boundary
stays intact. Filling the road's external contours into a solid polygon and
subtracting the road itself therefore leaves exactly the objects sitting on it,
with no appearance model and no detector involved.

That is the whole trick, and it is cheap: two morphology calls and a connected
components pass. Its limits follow directly from the construction — an obstacle
touching the road's edge does not enclose a hole, and a hole in the road surface
itself (a puddle, a shadow, a repair patch) is indistinguishable from an object.
"""

from __future__ import annotations

import numpy as np

ROAD_QUERY = 20
"""Which query segments road. A property of one checkpoint, not of the method."""

ROAD_THRESHOLD = 0.5
"""Mask probability above which a pixel counts as road."""


def road_mask(mask_pred: np.ndarray, height: int, width: int,
              query: int = ROAD_QUERY, threshold: float = ROAD_THRESHOLD) -> np.ndarray:
    """The road query, binarised and resized to the frame with nearest neighbour."""
    import cv2

    binary = (mask_pred[query] > threshold).astype(np.uint8) * 255
    return cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)


def road_polygon(road: np.ndarray) -> np.ndarray:
    """The road's external contours, filled solid — road plus whatever sits on it."""
    import cv2

    contours, _ = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(road)
    cv2.fillPoly(filled, contours, 255)
    return filled


def road_holes(mask_pred: np.ndarray, height: int, width: int,
               query: int = ROAD_QUERY, threshold: float = ROAD_THRESHOLD) -> np.ndarray:
    """Boolean map of candidate obstacles: inside the polygon, but not road."""
    road = road_mask(mask_pred, height, width, query, threshold)
    return np.logical_and(road_polygon(road) == 255, road == 0)


def candidate_components(holes: np.ndarray, min_side: int = 2) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Each hole as ``(mask, (x, y, w, h))``, skipping slivers.

    The original skips components with width or height ``<= 1`` and applies no
    minimum area at all, so single-pixel-wide streaks along the road edge do
    reach CLIP. Preserved rather than tightened: changing it changes the numbers.
    """
    import cv2

    count, labels = cv2.connectedComponents(holes.astype(np.uint8) * 255)
    components = []
    for index in range(1, count):
        mask = (labels == index).astype(np.uint8)
        x, y, width, height = cv2.boundingRect(mask)
        if width < min_side or height < min_side:
            continue
        components.append((mask, (int(x), int(y), int(width), int(height))))
    return components
