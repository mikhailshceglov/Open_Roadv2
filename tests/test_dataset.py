"""The dataset layer exists to stop silently-zero runs. These are those cases."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from open_road.dataset import DatasetSpec


def _spec(tmp_path: Path, **overrides) -> DatasetSpec:
    payload = {"name": "toy", "root": str(tmp_path), **overrides}
    return DatasetSpec.from_mapping(payload)


def _write_label(tmp_path: Path, stem: str, array: np.ndarray, suffix: str = ".png") -> None:
    labels = tmp_path / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(labels / f"{stem}{suffix}"), array.astype(np.uint8))


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    # The prototype ignored unknown keys, so a whole `rescan:` block vanished
    # silently and the failure surfaced later as an unrelated AttributeError.
    with pytest.raises(ValueError, match="unknown dataset key"):
        DatasetSpec.from_mapping({"name": "toy", "root": str(tmp_path), "anomly_value": 2})


def test_missing_root_is_rejected() -> None:
    with pytest.raises(ValueError, match="no 'root'"):
        DatasetSpec.from_mapping({"name": "toy"})


def test_anomaly_value_selects_exactly_that_value(tmp_path: Path) -> None:
    _write_label(tmp_path, "frame", np.array([[0, 1], [2, 2]]))
    spec = _spec(tmp_path, anomaly_value=2)

    anomaly, valid = spec.load_label("frame", (2, 2))

    assert anomaly.tolist() == [[False, False], [True, True]]
    assert valid.all()


def test_anomaly_value_none_accepts_any_nonzero(tmp_path: Path) -> None:
    _write_label(tmp_path, "frame", np.array([[0, 1], [2, 0]]))
    spec = _spec(tmp_path, anomaly_value=None)

    anomaly, _valid = spec.load_label("frame", (2, 2))

    assert anomaly.tolist() == [[False, True], [True, False]]


def test_void_pixels_are_invalid_and_never_anomalous(tmp_path: Path) -> None:
    _write_label(tmp_path, "frame", np.array([[0, 1], [255, 1]]))
    spec = _spec(tmp_path, anomaly_value=1, void_value=255)

    anomaly, valid = spec.load_label("frame", (2, 2))

    assert valid.tolist() == [[True, True], [False, True]]
    assert anomaly.tolist() == [[False, True], [False, True]]


def test_label_is_resized_with_nearest_so_values_survive(tmp_path: Path) -> None:
    # Bilinear here would invent a 1 between a 0 and a 2 and quietly corrupt
    # the ground truth; a few RoadAnomaly frames really are annotated at a
    # different resolution, so this path is exercised in practice.
    _write_label(tmp_path, "frame", np.array([[0, 2], [0, 2]]))
    spec = _spec(tmp_path, anomaly_value=2)

    anomaly, _valid = spec.load_label("frame", (4, 4))

    assert anomaly.shape == (4, 4)
    assert set(np.unique(anomaly).tolist()) <= {True, False}
    assert anomaly.any()


def test_frames_skips_dotfiles_and_honours_pattern(tmp_path: Path) -> None:
    images = tmp_path / "original"
    images.mkdir()
    for name in ("validation_a.jpg", "validation_b.jpg", "test_c.jpg", "._validation_d.jpg"):
        cv2.imwrite(str(images / name), np.zeros((2, 2, 3), dtype=np.uint8))

    all_frames = [p.name for p in _spec(tmp_path).frames()]
    assert "._validation_d.jpg" not in all_frames, "AppleDouble twins are unreadable by imread"
    assert len(all_frames) == 3

    filtered = [p.stem for p in _spec(tmp_path, pattern="validation").frames()]
    assert filtered == ["validation_a", "validation_b"]


def test_frames_on_a_missing_folder_says_which_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no images at"):
        _spec(tmp_path).frames()
