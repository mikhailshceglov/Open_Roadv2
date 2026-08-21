"""The Maskomaly soft mask, from Mask2Former query outputs.

The idea is inverted from most anomaly detectors: instead of asking "does this
pixel look strange", it asks "did anybody claim this pixel". A pixel that no
confident query claims is anomalous by elimination. That is why the map starts
at *one* everywhere and is pulled down, rather than starting at zero and being
pushed up.

Everything here is pure numpy over two arrays that the backbone produces:

    mask_cls   (N_queries, N_classes + 1)  softmax over classes; column 19 is void
    mask_pred  (N_queries, H, W)           per-pixel sigmoid

Output is ``[0, 1]`` float32, higher meaning more anomalous.

The magic query indices are properties of one checkpoint
(``model_final_17c1ee.pkl``), not of the method. They were ranked on SMIYC
validation and will be wrong for any other set of weights.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

ANOMALY_QUERIES: tuple[int, ...] = (49, 31, 83, 32)
"""Queries that empirically fire on anomalies. Ranked on SMIYC validation."""

GROUND_QUERIES: tuple[int, ...] = (19, 24)
"""Queries that empirically fire on ground. Ranked on Cityscapes validation."""

VOID_CLASS = 19
"""The no-object column of ``mask_cls``. 19 Cityscapes classes come before it."""

REJECTION_CONFIDENCE = 0.7
"""How sure a query must be about a known class before it may suppress a region."""

BORDER_THRESHOLD = 0.1
"""Mask probability above which a query counts as claiming a pixel."""

REJECTION_WEIGHT = 0.6
"""Blend of rejection and promotion. Anything above 0.5 works; 0.6 was used."""


def check_queries(mask_pred: np.ndarray, queries: Sequence[int], role: str) -> None:
    """Fail with the reason, not with a bare IndexError.

    The query indices are tuned to one checkpoint. Point this method at a model
    with a different number of queries and the failure is otherwise an
    ``IndexError`` deep inside a loop, which says nothing about the cause.
    """
    available = int(mask_pred.shape[0])
    out_of_range = [index for index in queries if not 0 <= index < available]
    if out_of_range:
        raise IndexError(
            f"{role} queries {out_of_range} do not exist: the model has {available} "
            f"queries. These indices were ranked for model_final_17c1ee.pkl and are "
            f"meaningless for other weights."
        )


def promotion_map(mask_cls: np.ndarray, mask_pred: np.ndarray,
                  queries: Sequence[int] = ANOMALY_QUERIES) -> np.ndarray:
    """Evidence *for* anomaly: the anomaly-ranked queries, weighted by confidence."""
    check_queries(mask_pred, queries, "anomaly")
    promotion = np.zeros_like(mask_pred[0], dtype=np.float32)
    for index in queries:
        promotion = np.maximum(promotion, mask_pred[index] * np.max(mask_cls[index]))
    return promotion


def rejection_map(mask_cls: np.ndarray, mask_pred: np.ndarray) -> np.ndarray:
    """Evidence *against* anomaly: every confident non-void query suppresses its region.

    Starts at one and is pulled towards zero. A query that is 90% sure it sees a
    car pulls its own mask down to 0.1 there, so a well-explained pixel ends up
    near zero and an unclaimed one stays near one.
    """
    rejection = np.ones_like(mask_pred[0], dtype=np.float32)
    for index in range(mask_cls.shape[0]):
        best = int(np.argmax(mask_cls[index]))
        if best == VOID_CLASS or mask_cls[index][best] <= REJECTION_CONFIDENCE:
            continue
        rejection = np.minimum(rejection, 1.0 - mask_pred[index] * mask_cls[index][best])
    return rejection


def contested_pixels(mask_cls: np.ndarray, mask_pred: np.ndarray,
                     threshold: float = BORDER_THRESHOLD) -> np.ndarray:
    """Pixels claimed by two or more non-void queries — object borders.

    The original walks all pairs of non-void masks and zeroes a pixel wherever
    both exceed the threshold. That is O(N²) over ~100 full-resolution masks and
    was the pipeline's dominant cost. The condition "some pair both exceed t" is
    exactly "at least two exceed t", so the whole double loop collapses into one
    sum over the query axis — bit-identical, and linear instead of quadratic.
    ``test_raas_maths.py`` checks the two against each other.
    """
    claimed = mask_pred[np.argmax(mask_cls, axis=1) != VOID_CLASS] > threshold
    return claimed.sum(axis=0) >= 2


def ground_correction(mask_cls: np.ndarray, mask_pred: np.ndarray,
                      queries: Sequence[int] = GROUND_QUERIES) -> np.ndarray:
    """Suppression from the ground queries, applied regardless of their confidence.

    Unlike the rejection loop these are not gated on 0.7: ground is mislabelled
    often enough that the queries are trusted unconditionally.
    """
    check_queries(mask_pred, queries, "ground")
    correction = np.ones_like(mask_pred[0], dtype=np.float32)
    for index in queries:
        best = int(np.argmax(mask_cls[index]))
        correction = np.minimum(correction, 1.0 - mask_pred[index] * mask_cls[index][best])
    return correction


def soft_mask(
    mask_cls: np.ndarray,
    mask_pred: np.ndarray,
    *,
    anomaly_queries: Sequence[int] = ANOMALY_QUERIES,
    ground_queries: Sequence[int] = GROUND_QUERIES,
    rejection_weight: float = REJECTION_WEIGHT,
) -> np.ndarray:
    """The full Maskomaly map, at the resolution of ``mask_pred``.

    Callers resize to the frame themselves; the original uses ``INTER_AREA``.
    """
    promotion = promotion_map(mask_cls, mask_pred, anomaly_queries)

    rejection = rejection_map(mask_cls, mask_pred)
    rejection = np.where(contested_pixels(mask_cls, mask_pred), 0.0, rejection)
    rejection = np.minimum(rejection, ground_correction(mask_cls, mask_pred, ground_queries))

    fused = rejection_weight * rejection + (1.0 - rejection_weight) * promotion
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def resolve_ranking(analysis_file: str | None, masks: int = 4) -> tuple[int, ...]:
    """Anomaly queries from an analysis file, or the SMIYC-ranked defaults.

    ``model_ori`` computes this and then ignores it — its loop hardcodes the
    defaults. ``model_id``/``model_ood`` do honour it. With no analysis file all
    three are identical, which is the usual case.
    """
    if not analysis_file:
        return ANOMALY_QUERIES
    payload = np.load(analysis_file)["cp"]
    ranking = np.argsort(payload)[::-1]
    take = int(masks) | int(np.argmax(payload[ranking] < 0.25))
    return tuple(int(value) for value in ranking[:take])
