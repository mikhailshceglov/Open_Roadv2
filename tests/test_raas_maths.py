"""RAAS arithmetic on hand-made query tensors.

Added by the raas branch. Nothing here loads a model: the whole point of pulling
the formulas out of the original model classes is that a reconstruction can be
pinned down by tests instead of trusted, and this method cannot currently be run
at all — its Swin-L checkpoint has no published URL.
"""

from __future__ import annotations

import numpy as np
import pytest

from methods.raas import clip_filter, road_aware, soft_mask
from methods.raas.soft_mask import VOID_CLASS

N_CLASSES = 20  # 19 Cityscapes classes + void


def _queries(count: int = 100, size: int = 8):
    """Empty (mask_cls, mask_pred): every query is confidently void, claiming nothing.

    100 queries because that is what Mask2Former emits, and because the anomaly
    and ground indices reach as high as 83.
    """
    mask_cls = np.zeros((count, N_CLASSES), dtype=np.float32)
    mask_cls[:, VOID_CLASS] = 1.0
    mask_pred = np.zeros((count, size, size), dtype=np.float32)
    return mask_cls, mask_pred


def _claim(mask_cls, mask_pred, query, region, class_id, confidence):
    """Make one query confidently predict `class_id` over `region`."""
    mask_cls[query] = 0.0
    mask_cls[query, class_id] = confidence
    mask_cls[query, VOID_CLASS] = 1.0 - confidence
    mask_pred[query][region] = 1.0


# -- the elimination formula ------------------------------------------------


def test_an_unclaimed_frame_is_entirely_anomalous() -> None:
    # The map starts at one and is only ever pulled down, so a frame nobody
    # claims stays at the rejection weight. This is the method's whole premise.
    mask_cls, mask_pred = _queries()

    result = soft_mask.soft_mask(mask_cls, mask_pred)

    assert result == pytest.approx(soft_mask.REJECTION_WEIGHT)


def test_a_confident_known_class_suppresses_its_region() -> None:
    mask_cls, mask_pred = _queries()
    region = (slice(0, 4), slice(0, 4))
    _claim(mask_cls, mask_pred, query=0, region=region, class_id=13, confidence=0.9)

    result = soft_mask.soft_mask(mask_cls, mask_pred)

    # 0.6 * (1 - 1.0*0.9) = 0.06 inside, 0.6 outside.
    assert result[region] == pytest.approx(0.06, abs=1e-5)
    assert result[6, 6] == pytest.approx(0.6, abs=1e-5)
    assert result[region].max() < result[6, 6]


def test_a_query_below_the_confidence_gate_suppresses_nothing() -> None:
    mask_cls, mask_pred = _queries()
    region = (slice(0, 4), slice(0, 4))
    _claim(mask_cls, mask_pred, query=0, region=region, class_id=13, confidence=0.5)

    result = soft_mask.soft_mask(mask_cls, mask_pred)

    assert result == pytest.approx(soft_mask.REJECTION_WEIGHT)


def test_a_void_query_never_suppresses_however_confident() -> None:
    # Void is "no object here", which is evidence *for* anomaly, not against it.
    mask_cls, mask_pred = _queries()
    mask_pred[0][:] = 1.0  # claims the whole frame, but as void

    result = soft_mask.soft_mask(mask_cls, mask_pred)

    assert result == pytest.approx(soft_mask.REJECTION_WEIGHT)


def test_the_anomaly_queries_push_a_region_up() -> None:
    mask_cls, mask_pred = _queries()
    region = (slice(2, 6), slice(2, 6))
    # An anomaly query firing, labelled void as anomalies are.
    mask_pred[soft_mask.ANOMALY_QUERIES[0]][region] = 1.0

    result = soft_mask.soft_mask(mask_cls, mask_pred)

    # 0.6*1 + 0.4*(1.0 * 1.0) = 1.0 inside, 0.6 outside.
    assert result[region] == pytest.approx(1.0, abs=1e-5)
    assert result[0, 0] == pytest.approx(0.6, abs=1e-5)


def test_the_output_stays_within_zero_and_one() -> None:
    rng = np.random.default_rng(0)
    mask_cls = rng.random((90, N_CLASSES)).astype(np.float32)
    mask_cls /= mask_cls.sum(axis=1, keepdims=True)
    mask_pred = rng.random((90, 16, 16)).astype(np.float32)

    result = soft_mask.soft_mask(mask_cls, mask_pred)

    assert result.min() >= 0.0 and result.max() <= 1.0
    assert result.dtype == np.float32


# -- the O(N^2) collapse ----------------------------------------------------


def _contested_naive(mask_cls, mask_pred, threshold=soft_mask.BORDER_THRESHOLD):
    """The original all-pairs loop, kept only to check the fast form against."""
    positive = mask_pred[np.argmax(mask_cls, axis=1) != VOID_CLASS]
    zeroed = np.zeros(mask_pred.shape[1:], dtype=bool)
    for i in range(positive.shape[0]):
        for j in range(i + 1, positive.shape[0]):
            zeroed |= np.logical_and(positive[i] > threshold, positive[j] > threshold)
    return zeroed


def test_the_vectorised_border_rule_matches_the_all_pairs_loop() -> None:
    # "Some pair both exceed t" is exactly "at least two exceed t". The original
    # walked ~100 full-resolution masks pairwise for this and it dominated the
    # runtime; the collapse must be bit-identical, not merely close.
    rng = np.random.default_rng(1)
    mask_cls = rng.random((40, N_CLASSES)).astype(np.float32)
    mask_pred = rng.random((40, 12, 12)).astype(np.float32)

    fast = soft_mask.contested_pixels(mask_cls, mask_pred)
    slow = _contested_naive(mask_cls, mask_pred)

    assert np.array_equal(fast, slow)


def test_two_overlapping_objects_have_their_shared_pixels_zeroed() -> None:
    mask_cls, mask_pred = _queries()
    _claim(mask_cls, mask_pred, 0, (slice(0, 6), slice(0, 6)), class_id=13, confidence=0.9)
    _claim(mask_cls, mask_pred, 1, (slice(4, 8), slice(4, 8)), class_id=11, confidence=0.9)

    contested = soft_mask.contested_pixels(mask_cls, mask_pred)

    assert contested[5, 5]        # claimed by both
    assert not contested[1, 1]    # claimed by one
    assert not contested[7, 7]    # claimed by one


# -- road-aware geometry ----------------------------------------------------


def test_an_object_on_the_road_becomes_a_hole() -> None:
    # The road query covers a band; the obstacle is a gap in it. Filling the
    # road's outer contour and subtracting the road leaves exactly the gap.
    mask_pred = np.zeros((21, 20, 20), dtype=np.float32)
    mask_pred[road_aware.ROAD_QUERY][5:15, 2:18] = 1.0
    mask_pred[road_aware.ROAD_QUERY][8:12, 8:12] = 0.0   # the obstacle

    holes = road_aware.road_holes(mask_pred, 20, 20)

    assert holes[9:11, 9:11].all()
    assert not holes[6, 4]     # plain road
    assert not holes[1, 1]     # outside the road entirely


def test_a_road_with_nothing_on_it_yields_no_candidates() -> None:
    mask_pred = np.zeros((21, 20, 20), dtype=np.float32)
    mask_pred[road_aware.ROAD_QUERY][5:15, 2:18] = 1.0

    holes = road_aware.road_holes(mask_pred, 20, 20)

    assert not holes.any()
    assert road_aware.candidate_components(holes) == []


def test_slivers_are_skipped_but_real_components_survive() -> None:
    holes = np.zeros((20, 20), dtype=bool)
    holes[4:9, 4:9] = True   # a real component
    holes[15, 2:12] = True   # one pixel tall

    components = road_aware.candidate_components(holes)

    assert len(components) == 1
    assert components[0][1] == (4, 4, 5, 5)


# -- the CLIP decision rules ------------------------------------------------


def _probabilities(values: dict[int, float], size: int) -> np.ndarray:
    array = np.full(size, (1.0 - sum(values.values())) / (size - len(values)), dtype=np.float32)
    for index, value in values.items():
        array[index] = value
    return array


def test_id_rule_suppresses_a_confidently_named_region() -> None:
    decision = clip_filter.decide_id(_probabilities({13: 0.95}, len(clip_filter.ID_PROMPTS)))

    assert not decision.anomalous
    assert decision.score == clip_filter.KNOWN_SCORE
    assert decision.label == clip_filter.ID_PROMPTS[13]


def test_id_rule_reports_a_region_it_cannot_name() -> None:
    decision = clip_filter.decide_id(_probabilities({13: 0.4}, len(clip_filter.ID_PROMPTS)))

    assert decision.anomalous
    assert decision.score == clip_filter.ANOMALY_SCORE
    assert decision.label == "unknown"


def test_ood_rule_forces_known_when_the_ood_prompts_attract_nothing() -> None:
    # The asymmetry worth knowing about: a region CLIP finds unconvincing across
    # *all* prompts is suppressed rather than reported, whatever its ID score.
    size = len(clip_filter.ID_PROMPTS) + len(clip_filter.OOD_PROMPTS)
    probabilities = np.full(size, 0.05, dtype=np.float32)
    probabilities[len(clip_filter.ID_PROMPTS) :] = 1e-5

    decision = clip_filter.decide_ood(probabilities)

    assert not decision.anomalous
    assert decision.id_probability < clip_filter.ID_CONFIDENCE


def test_ood_rule_needs_the_known_name_to_win_by_a_margin() -> None:
    size = len(clip_filter.ID_PROMPTS) + len(clip_filter.OOD_PROMPTS)
    # Known name is confident, but the OOD prompt is right behind it.
    probabilities = _probabilities({13: 0.88, size - 1: 0.86}, size)

    decision = clip_filter.decide_ood(probabilities)

    assert decision.anomalous, "a narrow win over the OOD prompt must not count as known"


def test_apply_decision_writes_only_inside_the_component() -> None:
    score = np.full((10, 10), 0.6, dtype=np.float32)
    component = np.zeros((10, 10), dtype=np.uint8)
    component[2:5, 2:5] = 1
    decision = clip_filter.Decision(
        anomalous=False, score=clip_filter.KNOWN_SCORE, label="a photo of car on the road",
        id_probability=0.9,
    )

    clip_filter.apply_decision(score, component, (2, 2, 3, 3), decision)

    assert score[3, 3] == pytest.approx(clip_filter.KNOWN_SCORE)
    assert score[7, 7] == pytest.approx(0.6)


def test_the_variants_disagree_on_the_same_evidence() -> None:
    # A region with a middling known name and real OOD mass: the id rule reports
    # it (0.5 < 0.85), and so does the ood rule (no margin). Where they diverge
    # is the near-zero-OOD case above, which is the reason both exist.
    size = len(clip_filter.ID_PROMPTS) + len(clip_filter.OOD_PROMPTS)
    probabilities = _probabilities({13: 0.5, size - 1: 0.3}, size)

    assert clip_filter.decide_id(probabilities[: len(clip_filter.ID_PROMPTS)]).anomalous
    assert clip_filter.decide_ood(probabilities).anomalous


def test_query_indices_beyond_the_model_fail_with_the_reason() -> None:
    # The indices belong to one checkpoint. Against a model with fewer queries
    # the bare IndexError says nothing about why.
    mask_cls, mask_pred = _queries(count=10)

    with pytest.raises(IndexError, match="model_final_17c1ee"):
        soft_mask.soft_mask(mask_cls, mask_pred)
