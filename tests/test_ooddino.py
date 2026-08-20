"""OoDDINO's arithmetic and fusion, on hand-made tensors.

Added by the ooddino branch. None of it loads a model: the maths is pure numpy
precisely so that a reconstruction of unpublished equations can be pinned down
by tests rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from methods.ooddino import maths
from methods.ooddino.boxes import (
    Box,
    Detection,
    Instance,
    boxes_to_mask,
    connected_instances,
    filter_boxes_by_prior,
    rasterize,
    select_top_k,
    suppress_nested,
)


def _logits(confident: bool, classes: int = 19, size: int = 4) -> np.ndarray:
    array = np.zeros((classes, size, size), dtype=np.float32)
    if confident:
        # 30 rather than 10: with 18 competing zero logits, a margin of 10 still
        # leaves p_max at 0.999 and an entropy of 0.003, which is small but not
        # the zero the test is about.
        array[0] = 30.0
    return array


# -- pixel branch -----------------------------------------------------------


def test_entropy_is_zero_when_certain_and_one_when_uniform() -> None:
    assert maths.entropy_map(_logits(confident=True)).max() < 1e-3
    # All-equal logits are the uniform distribution, whose entropy is log C.
    assert maths.entropy_map(_logits(confident=False)) == pytest.approx(1.0, abs=1e-5)


def test_distance_is_one_minus_p_max() -> None:
    uniform = maths.distance_map(_logits(confident=False))
    assert uniform == pytest.approx(1.0 - 1.0 / 19, abs=1e-5)
    assert maths.distance_map(_logits(confident=True)).max() < 1e-3


def test_energy_is_lower_for_a_confident_pixel() -> None:
    # Liu et al.: higher energy means more OOD, and a confident pixel is not.
    assert maths.energy_map(_logits(confident=True)).mean() < maths.energy_map(
        _logits(confident=False)
    ).mean()


def test_class_residual_cancels_a_class_wide_offset() -> None:
    # The whole point: a class that is uncertain everywhere must not dominate.
    values = np.array([[10.0, 12.0], [0.0, 2.0]])
    labels = np.array([[0, 0], [1, 1]])

    residual = maths.class_conditional_residual(values, labels)

    # Each class is centred on its own median, so both rows get the same spread.
    assert residual[0].tolist() == residual[1].tolist()
    assert residual.sum() == pytest.approx(0.0)


def test_class_residual_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        maths.class_conditional_residual(np.zeros((2, 2)), np.zeros((3, 3)))


def test_pixel_score_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown pixel score"):
        maths.pixel_score(_logits(confident=True), kind="magic")


def test_minmax_of_a_flat_map_is_zeros_not_a_division_by_zero() -> None:
    assert maths.minmax01(np.full((3, 3), 7.0)).tolist() == np.zeros((3, 3)).tolist()


def test_orthogonalize_removes_the_shared_component() -> None:
    base = np.array([[1.0, 2.0], [3.0, 4.0]])

    residual = maths.orthogonalize(base, base * 2.0)

    # A map that is entirely parallel to the base has nothing left over.
    assert np.abs(residual).max() < 1e-6


# -- fusion -----------------------------------------------------------------


def test_arns_output_lands_in_alpha_to_alpha_plus_half() -> None:
    score = np.random.default_rng(0).normal(size=(16, 16))
    foreground = np.zeros((16, 16), dtype=bool)
    foreground[:8] = True

    normalised = maths.arns_normalize(score, foreground, alpha=0.5)

    assert normalised.min() >= 0.5
    assert normalised.max() <= 1.0


def test_arns_centres_each_region_on_its_own_mean() -> None:
    # Background is offset far above foreground; after ARNS both regions
    # straddle the same midpoint, which is exactly what makes one threshold
    # per region necessary afterwards.
    score = np.zeros((8, 8))
    score[4:] = 100.0
    foreground = np.zeros((8, 8), dtype=bool)
    foreground[:4] = True

    normalised = maths.arns_normalize(score, foreground, alpha=0.5)

    assert normalised[:4].mean() == pytest.approx(normalised[4:].mean(), abs=1e-4)


def test_dual_thresholds_are_forced_apart_when_quantiles_collapse() -> None:
    flat = np.full((8, 8), 0.75, dtype=np.float32)
    foreground = np.zeros((8, 8), dtype=bool)
    foreground[:4] = True

    t_fg, t_bg = maths.dual_thresholds(flat, foreground, min_gap=0.2, alpha=0.5)

    assert t_bg - t_fg >= 0.2 - 1e-6


def test_adt_decision_is_exactly_score_versus_the_regional_threshold() -> None:
    values = np.array([[0.4, 0.9]], dtype=np.float32)
    foreground = np.array([[True, False]])

    probability = maths.adt_probability(values, foreground, t_fg=0.3, t_bg=0.95, delta=0.2)

    # Inside a proposal 0.4 clears the lenient 0.3; outside, 0.9 misses 0.95.
    assert (probability >= 0.5).tolist() == [[True, False]]


def test_delta_softens_the_probability_without_moving_the_decision() -> None:
    values = np.array([[0.4, 0.9]], dtype=np.float32)
    foreground = np.array([[True, False]])

    narrow = maths.adt_probability(values, foreground, 0.3, 0.95, delta=0.01)
    wide = maths.adt_probability(values, foreground, 0.3, 0.95, delta=0.5)

    assert (narrow >= 0.5).tolist() == (wide >= 0.5).tolist()
    assert narrow[0, 0] > wide[0, 0]


# -- boxes and instances ----------------------------------------------------


def _detection(x1, y1, x2, y2, confidence=0.5) -> Detection:
    return Detection(box=Box(x1, y1, x2, y2), label="obstacle", confidence=confidence)


def test_boxes_become_a_union_of_filled_rectangles() -> None:
    mask = boxes_to_mask([Box(0, 0, 2, 2), Box(4, 4, 6, 6)], 8, 8)

    assert mask.sum() == 8
    assert mask[0, 0] and mask[5, 5] and not mask[3, 3]


def test_top_k_is_disabled_at_zero_and_keeps_the_strongest_otherwise() -> None:
    detections = [_detection(0, 0, 1, 1, c) for c in (0.1, 0.9, 0.5)]

    assert len(select_top_k(detections, 0)) == 3
    assert [d.confidence for d in select_top_k(detections, 2)] == [0.9, 0.5]


def test_the_prior_filters_proposals_it_does_not_believe_in() -> None:
    prior = np.zeros((8, 8), dtype=np.float32)
    prior[0:2, 0:2] = 1.0
    detections = [_detection(0, 0, 2, 2), _detection(4, 4, 6, 6)]

    kept = filter_boxes_by_prior(detections, prior, threshold=0.5)

    assert len(kept) == 1
    assert kept[0].box == Box(0, 0, 2, 2)
    # A non-positive threshold is a pass-through, not a reject-all.
    assert len(filter_boxes_by_prior(detections, prior, threshold=0.0)) == 2


def test_require_foreground_discards_a_blob_outside_every_proposal() -> None:
    anomaly = np.zeros((16, 16), dtype=bool)
    anomaly[0:4, 0:4] = True     # inside the proposal
    anomaly[12:16, 12:16] = True  # outside it
    foreground = boxes_to_mask([Box(0, 0, 4, 4)], 16, 16)
    score = np.full((16, 16), 0.8, dtype=np.float32)

    kept = connected_instances(
        anomaly, score, min_blob_area=4, foreground=foreground, min_foreground_overlap=0.5
    )

    assert len(kept) == 1
    assert kept[0].box == Box(0, 0, 4, 4)


def test_blobs_outside_the_area_window_are_dropped() -> None:
    anomaly = np.zeros((16, 16), dtype=bool)
    anomaly[0, 0] = True          # too small
    anomaly[2:6, 2:6] = True      # just right
    score = np.full((16, 16), 0.8, dtype=np.float32)

    kept = connected_instances(anomaly, score, min_blob_area=4, max_blob_area_frac=0.5)

    assert [item.box for item in kept] == [Box(2, 2, 6, 6)]


def test_nested_suppression_keeps_the_container() -> None:
    big = Instance(box=Box(0, 0, 10, 10), mask=np.ones((10, 10), bool), confidence=0.7)
    speck = Instance(box=Box(2, 2, 4, 4), mask=np.ones((2, 2), bool), confidence=0.9)

    kept = suppress_nested([speck, big], containment=0.8)

    assert [item.box for item in kept] == [Box(0, 0, 10, 10)]


def test_rasterize_gives_background_a_hard_zero() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True
    dense = rasterize([Instance(box=Box(1, 1, 3, 3), mask=mask, confidence=0.8)], 8, 8)

    assert dense[2, 2] == pytest.approx(0.8)
    assert dense[7, 7] == 0.0
    # Every reported pixel outranks every unreported one, which is what makes
    # the pixel metrics measure the pipeline's real decisions.
    assert dense[mask].min() > dense[~mask].max()


def test_overlapping_instances_take_the_higher_confidence() -> None:
    left = np.zeros((8, 8), dtype=bool)
    left[0:4, 0:4] = True
    right = np.zeros((8, 8), dtype=bool)
    right[2:6, 2:6] = True

    dense = rasterize(
        [
            Instance(box=Box(0, 0, 4, 4), mask=left, confidence=0.6),
            Instance(box=Box(2, 2, 6, 6), mask=right, confidence=0.9),
        ],
        8,
        8,
    )

    assert dense[3, 3] == pytest.approx(0.9)
