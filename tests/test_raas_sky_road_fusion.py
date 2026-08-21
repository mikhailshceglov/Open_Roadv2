"""Semantic fusion on hand-made maps and a stand-in SAM bundle.

Added by the raas-sky-road branch. No model is loaded: fusion is pure numpy over
a score map, a semantic map and a set of region masks, which is what lets a port
of unreleased code be pinned down rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from methods.raas_sky_road import fusion

ROAD, VEGETATION, SKY = 0, 8, 10


class Bundle:
    """The parts of Objectomaly's MaskBundle that fusion actually reads."""

    def __init__(self, masks, quality=None):
        self.masks = np.asarray(masks, dtype=bool)
        self.n = len(self.masks)
        self.area = np.array([int(m.sum()) for m in self.masks])
        self.quality = np.array(quality if quality is not None else [1.0] * self.n)
        self.bbox = []
        for m in self.masks:
            ys, xs = np.where(m)
            self.bbox.append(
                [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                if len(xs) else [0, 0, 0, 0]
            )


def _semantic(shape, sky_rows=slice(0, 8)):
    semantic = np.full(shape, ROAD, dtype=np.int16)
    semantic[sky_rows] = SKY
    return semantic


def _config(**overrides):
    config = {
        "class_factors": {SKY: 0.35, VEGETATION: 0.55, ROAD: 1.0},
        "road_class_ids": [ROAD],
        "road_boost": 1.0,
        "background_class_ids": list(range(19)),
        "candidates": {
            "min_area_frac": 0.0, "max_area_frac": 0.5, "score_quantile": 0.9,
            "min_score": 0.45, "min_contrast": 0.12, "ring_radius": 2,
            "max_candidates": 16, "clip_probe_min_score": 0.05,
            "clip_margin_threshold": 0.0, "clip_candidate_floor": 0.65,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config


# -- attenuation ------------------------------------------------------------


def test_sky_is_attenuated_and_road_is_not() -> None:
    score = np.full((16, 16), 0.8, dtype=np.float32)

    fused = fusion.attenuate(score, _semantic(score.shape), {SKY: 0.35, ROAD: 1.0})

    assert fused[0, 0] == pytest.approx(0.28)    # 0.8 * 0.35
    assert fused[12, 0] == pytest.approx(0.8)


def test_a_class_with_no_factor_keeps_its_score() -> None:
    score = np.full((4, 4), 0.6, dtype=np.float32)
    semantic = np.full((4, 4), 13, dtype=np.int16)   # car, not in the table

    assert fusion.attenuate(score, semantic, {SKY: 0.35}) == pytest.approx(0.6)


def test_nothing_is_ever_hard_masked() -> None:
    # A masked sky could not be argued back, and airborne debris is the whole
    # reason this stage exists. Attenuation must stay strictly positive.
    score = np.full((4, 4), 0.9, dtype=np.float32)
    semantic = np.full((4, 4), SKY, dtype=np.int16)

    fused = fusion.attenuate(score, semantic, {SKY: 0.35})

    assert (fused > 0).all()


def test_road_boost_lifts_against_the_original_score() -> None:
    # The boost must not be cancellable by a small class factor, so it is taken
    # against the pre-attenuation score.
    score = np.full((8, 8), 0.5, dtype=np.float32)
    semantic = np.full((8, 8), ROAD, dtype=np.int16)
    attenuated = score * 0.5

    boosted = fusion.boost_road(attenuated, score, semantic, [ROAD], 1.05)

    assert boosted == pytest.approx(0.525)   # 0.5 * 1.05, not 0.25 * 1.05


def test_a_boost_of_one_changes_nothing() -> None:
    score = np.full((4, 4), 0.5, dtype=np.float32)
    semantic = np.full((4, 4), ROAD, dtype=np.int16)

    assert fusion.boost_road(score, score, semantic, [ROAD], 1.0) is score


# -- region measurement -----------------------------------------------------


def test_ring_median_reads_the_surroundings_not_the_region() -> None:
    score = np.zeros((20, 20), dtype=np.float32)
    mask = np.zeros((20, 20), dtype=bool)
    mask[8:12, 8:12] = True
    score[mask] = 1.0          # bright object
    score[~mask] = 0.2         # dim background

    assert fusion.ring_median(score, mask, radius=2) == pytest.approx(0.2)


def test_contrast_separates_an_object_from_a_bright_background() -> None:
    # A bright region on a bright ground has no contrast; the same region on a
    # dark ground does. This is why the ring exists rather than a global mean.
    shape = (24, 24)
    mask = np.zeros(shape, dtype=bool)
    mask[10:14, 10:14] = True

    on_dark = np.full(shape, 0.1, dtype=np.float32); on_dark[mask] = 0.9
    on_bright = np.full(shape, 0.85, dtype=np.float32); on_bright[mask] = 0.9

    dark = fusion.describe_candidates(on_dark, _semantic(shape), Bundle([mask]),
                                      ring_radius=2)[0]
    bright = fusion.describe_candidates(on_bright, _semantic(shape), Bundle([mask]),
                                        ring_radius=2)[0]

    assert dark["contrast"] > 0.7
    assert bright["contrast"] < 0.1


def test_dominant_class_of_an_empty_mask_is_minus_one() -> None:
    assert fusion.dominant_class(np.zeros((4, 4), np.int16), np.zeros((4, 4), bool)) == -1


def test_candidates_outside_the_area_window_are_dropped() -> None:
    shape = (20, 20)
    small = np.zeros(shape, dtype=bool); small[0:2, 0:2] = True      # 1% of frame
    huge = np.zeros(shape, dtype=bool); huge[:, :] = True            # 100%
    score = np.full(shape, 0.5, dtype=np.float32)

    kept = fusion.describe_candidates(
        score, _semantic(shape), Bundle([small, huge]),
        min_area_frac=0.005, max_area_frac=0.05,
    )

    assert [item["index"] for item in kept] == [0]


def test_candidates_are_ranked_by_score_contrast_and_quality() -> None:
    shape = (32, 32)
    weak = np.zeros(shape, dtype=bool); weak[4:8, 4:8] = True
    strong = np.zeros(shape, dtype=bool); strong[20:24, 20:24] = True
    score = np.full(shape, 0.1, dtype=np.float32)
    score[weak] = 0.4
    score[strong] = 0.95

    ranked = fusion.describe_candidates(score, _semantic(shape),
                                        Bundle([weak, strong]), ring_radius=2)

    assert [item["index"] for item in ranked] == [1, 0]


def test_negative_contrast_does_not_reward_a_region() -> None:
    # max(0, contrast) in the sort key: a region dimmer than its ring must not
    # be pushed up the ranking by the magnitude of how much dimmer it is.
    shape = (24, 24)
    mask = np.zeros(shape, dtype=bool); mask[10:14, 10:14] = True
    score = np.full(shape, 0.9, dtype=np.float32); score[mask] = 0.1

    item = fusion.describe_candidates(score, _semantic(shape), Bundle([mask]),
                                      ring_radius=2)[0]

    assert item["contrast"] < 0


# -- protection -------------------------------------------------------------


def test_a_high_contrast_region_gets_its_original_score_back() -> None:
    shape = (24, 24)
    mask = np.zeros(shape, dtype=bool); mask[4:8, 4:8] = True   # inside the sky
    score = np.full(shape, 0.1, dtype=np.float32); score[mask] = 0.9

    fused, protected, diagnostics = fusion.apply_global_fusion(
        score, _semantic(shape), np.zeros((*shape, 3), np.uint8),
        Bundle([mask]), _config(),
    )

    assert protected[6, 6]
    assert fused[6, 6] == pytest.approx(0.9)      # restored, not 0.9 * 0.35
    assert fused[2, 20] == pytest.approx(0.035)   # unprotected sky stays attenuated
    assert diagnostics["protected_count"] == 1
    assert "contrast" in diagnostics["candidates"][0]["reasons"]


def test_a_low_contrast_region_stays_attenuated() -> None:
    shape = (24, 24)
    mask = np.zeros(shape, dtype=bool); mask[4:8, 4:8] = True
    score = np.full(shape, 0.5, dtype=np.float32)   # no contrast at all

    fused, protected, diagnostics = fusion.apply_global_fusion(
        score, _semantic(shape), np.zeros((*shape, 3), np.uint8),
        Bundle([mask]), _config(),
    )

    assert not protected.any()
    assert fused[6, 6] == pytest.approx(0.175)     # 0.5 * 0.35
    assert diagnostics["candidates"][0]["reasons"] == []


def test_the_shipped_background_list_makes_semantic_protection_unreachable() -> None:
    # Every Cityscapes class is listed as background, so `dominant_class not in
    # background_ids` can never hold. Pinned as a test because it is a live
    # defect carried over deliberately, not an accident of this port.
    shape = (24, 24)
    mask = np.zeros(shape, dtype=bool); mask[4:8, 4:8] = True
    score = np.full(shape, 0.6, dtype=np.float32)   # above min_score, no contrast

    _fused, _protected, diagnostics = fusion.apply_global_fusion(
        score, _semantic(shape), np.zeros((*shape, 3), np.uint8),
        Bundle([mask]), _config(),
    )

    assert "semantic_object" not in diagnostics["candidates"][0]["reasons"]


def test_shortening_the_background_list_revives_semantic_protection() -> None:
    shape = (24, 24)
    mask = np.zeros(shape, dtype=bool); mask[4:8, 4:8] = True
    score = np.full(shape, 0.6, dtype=np.float32)

    _fused, protected, diagnostics = fusion.apply_global_fusion(
        score, _semantic(shape), np.zeros((*shape, 3), np.uint8),
        Bundle([mask]), _config(background_class_ids=[ROAD, SKY, VEGETATION]),
    )
    # The region's dominant class is sky, still background -> not protected.
    assert not protected.any()

    _fused, protected, _d = fusion.apply_global_fusion(
        score, _semantic(shape), np.zeros((*shape, 3), np.uint8),
        Bundle([mask]), _config(background_class_ids=[ROAD]),
    )
    assert protected.any()


class _Validator:
    """Stand-in for CLIPRegionValidator, returning a fixed margin."""

    def __init__(self, margin):
        self.margin = margin
        self.probed = None

    def score_regions(self, image_bgr, bundle, indices):
        self.probed = list(indices)
        return {i: {"clip_margin": self.margin} for i in self.probed}


def test_a_clip_ood_region_is_floored_even_without_contrast() -> None:
    shape = (24, 24)
    mask = np.zeros(shape, dtype=bool); mask[4:8, 4:8] = True
    score = np.full(shape, 0.1, dtype=np.float32)   # far below the floor

    fused, protected, diagnostics = fusion.apply_global_fusion(
        score, _semantic(shape), np.zeros((*shape, 3), np.uint8),
        Bundle([mask]), _config(), _Validator(margin=0.2),
    )

    assert protected[6, 6]
    assert fused[6, 6] == pytest.approx(0.65)
    assert diagnostics["candidates"][0]["reasons"] == ["clip_ood"]


def test_a_clip_margin_below_the_threshold_protects_nothing() -> None:
    shape = (24, 24)
    mask = np.zeros(shape, dtype=bool); mask[4:8, 4:8] = True
    score = np.full(shape, 0.1, dtype=np.float32)

    _fused, protected, _d = fusion.apply_global_fusion(
        score, _semantic(shape), np.zeros((*shape, 3), np.uint8),
        Bundle([mask]), _config(candidates={"clip_margin_threshold": 0.5}),
        _Validator(margin=0.2),
    )

    assert not protected.any()


def test_only_regions_above_the_probe_threshold_reach_clip() -> None:
    # CLIP is the expensive part; the probe threshold is what keeps it off
    # regions nothing believes in.
    shape = (32, 32)
    dim = np.zeros(shape, dtype=bool); dim[2:5, 2:5] = True
    bright = np.zeros(shape, dtype=bool); bright[20:24, 20:24] = True
    score = np.full(shape, 0.01, dtype=np.float32)
    score[bright] = 0.8

    validator = _Validator(margin=-1.0)
    fusion.apply_global_fusion(
        score, _semantic(shape), np.zeros((*shape, 3), np.uint8),
        Bundle([dim, bright]), _config(candidates={"clip_probe_min_score": 0.5}),
        validator,
    )

    assert validator.probed == [1]


def test_the_output_is_clipped_and_shaped_like_the_input() -> None:
    shape = (16, 16)
    mask = np.zeros(shape, dtype=bool); mask[2:6, 2:6] = True
    score = np.full(shape, 0.9, dtype=np.float32)

    fused, protected, _d = fusion.apply_global_fusion(
        score, _semantic(shape), np.zeros((*shape, 3), np.uint8),
        Bundle([mask]), _config(road_boost=2.0),
    )

    assert fused.shape == shape and fused.dtype == np.float32
    assert fused.min() >= 0.0 and fused.max() <= 1.0
    assert protected.dtype == np.bool_


def test_a_semantic_map_of_the_wrong_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        fusion.apply_global_fusion(
            np.zeros((8, 8), np.float32), np.zeros((4, 4), np.int16),
            np.zeros((8, 8, 3), np.uint8), Bundle([np.ones((8, 8), bool)]), _config(),
        )
