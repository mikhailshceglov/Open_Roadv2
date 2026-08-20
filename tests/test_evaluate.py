"""Metrics, on tiny hand-checkable frames.

The interesting cases are not the arithmetic -- sklearn does that -- but the
two ways an evaluation goes silently wrong: a label encoding that resolves to
an empty ground truth, and void pixels counted as negatives.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from open_road.dataset import DatasetSpec
from open_road.io import RunLayout, read_json, save_score
from open_road.stages.evaluate import component_metrics, pixel_metrics, run_evaluate


def _dataset(tmp_path: Path, **overrides) -> DatasetSpec:
    return DatasetSpec.from_mapping({"name": "toy", "root": str(tmp_path), **overrides})


def _frame(tmp_path: Path, stem: str, label: np.ndarray) -> None:
    images = tmp_path / "original"
    labels = tmp_path / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    height, width = label.shape
    cv2.imwrite(str(images / f"{stem}.jpg"), np.zeros((height, width, 3), dtype=np.uint8))
    cv2.imwrite(str(labels / f"{stem}.png"), label.astype(np.uint8))


def _scores(tmp_path: Path, stem: str, score: np.ndarray) -> RunLayout:
    layout = RunLayout(tmp_path / "run")
    save_score(layout.score_path(stem), score)
    return layout


def test_a_perfect_ranking_scores_perfectly() -> None:
    gt = np.array([0, 0, 1, 1], dtype=bool)
    score = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)

    metrics = pixel_metrics(gt, score)

    assert metrics["AP"] == pytest.approx(1.0)
    assert metrics["AUROC"] == pytest.approx(1.0)
    assert metrics["FPR95"] == pytest.approx(0.0)
    assert metrics["F1"] == pytest.approx(1.0)


def test_component_metrics_on_one_hit_and_one_miss() -> None:
    gt = np.zeros((6, 6), dtype=bool)
    gt[0:2, 0:2] = True   # found
    gt[4:6, 4:6] = True   # missed
    pred = np.zeros((6, 6), dtype=bool)
    pred[0:2, 0:2] = True

    sious, ppvs = component_metrics(gt, pred)

    assert sorted(sious) == pytest.approx([0.0, 1.0])
    assert ppvs == pytest.approx([1.0])


def test_one_prediction_spanning_two_objects_is_not_punished_twice() -> None:
    # The "adjusted" in sIoU: pixels owned by the *other* GT component leave the
    # union, so a single prediction covering both scores 1.0 against each.
    gt = np.zeros((4, 8), dtype=bool)
    gt[1:3, 1:3] = True
    gt[1:3, 5:7] = True
    pred = np.zeros((4, 8), dtype=bool)
    pred[1:3, 1:7] = True

    sious, _ppvs = component_metrics(gt, pred)

    assert len(sious) == 2
    assert all(value > 0.4 for value in sious)


def test_void_pixels_leave_every_metric(tmp_path: Path) -> None:
    # A confident false positive sitting entirely on void must not cost
    # anything: nobody labelled those pixels, so nobody can be wrong there.
    label = np.array([[1, 1, 255, 255], [0, 0, 255, 255]], dtype=np.uint8)
    _frame(tmp_path, "frame", label)
    score = np.array([[9.0, 9.0, 9.0, 9.0], [-9.0, -9.0, 9.0, 9.0]], dtype=np.float32)
    layout = _scores(tmp_path, "frame", score)

    metrics = run_evaluate(
        _dataset(tmp_path, anomaly_value=1, void_value=255), layout, report=lambda _: None
    )

    assert metrics["pixel"]["void_pixels"] == 4
    assert metrics["pixel"]["AP"] == pytest.approx(1.0)
    assert metrics["pixel"]["FP"] == 0


def test_the_same_frame_without_void_handling_is_scored_worse(tmp_path: Path) -> None:
    # The counterpart to the test above, and the reason void handling was added:
    # treating 255 as "not anomaly" turns two unlabelled pixels into two false
    # positives and drags precision down.
    label = np.array([[1, 1, 255, 255], [0, 0, 255, 255]], dtype=np.uint8)
    _frame(tmp_path, "frame", label)
    score = np.array([[9.0, 9.0, 9.0, 9.0], [-9.0, -9.0, 9.0, 9.0]], dtype=np.float32)
    layout = _scores(tmp_path, "frame", score)

    metrics = run_evaluate(
        _dataset(tmp_path, anomaly_value=1, void_value=None), layout, report=lambda _: None
    )

    assert metrics["pixel"]["void_pixels"] == 0
    assert metrics["pixel"]["AP"] < 1.0


def test_an_all_zero_ground_truth_stops_instead_of_reporting_nothing(tmp_path: Path) -> None:
    # RoadAnomaly encodes anomaly as 2. Reading it as 1 yields an empty ground
    # truth, and the old scripts happily reported metrics against nothing.
    _frame(tmp_path, "frame", np.array([[0, 2], [0, 2]], dtype=np.uint8))
    layout = _scores(tmp_path, "frame", np.zeros((2, 2), dtype=np.float32))

    with pytest.raises(SystemExit, match="anomaly_value"):
        run_evaluate(_dataset(tmp_path, anomaly_value=1), layout, report=lambda _: None)


def test_metrics_json_is_written_and_names_the_run(tmp_path: Path) -> None:
    _frame(tmp_path, "frame", np.array([[0, 1], [0, 1]], dtype=np.uint8))
    layout = _scores(tmp_path, "frame", np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32))

    run_evaluate(
        _dataset(tmp_path, anomaly_value=1), layout, method="toy_method", report=lambda _: None
    )

    written = read_json(layout.metrics)
    assert written["method"] == "toy_method"
    assert written["dataset"] == "toy"
    assert written["frames"] == 1
    assert read_json(layout.manifest)["stages"]["evaluate"]["frames"] == 1


def test_evaluating_without_score_maps_says_to_infer_first(tmp_path: Path) -> None:
    _frame(tmp_path, "frame", np.array([[0, 1]], dtype=np.uint8))

    with pytest.raises(SystemExit, match="open-road infer"):
        run_evaluate(
            _dataset(tmp_path, anomaly_value=1), RunLayout(tmp_path / "run"), report=lambda _: None
        )
