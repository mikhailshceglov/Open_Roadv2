"""Render and infer, driven by a stand-in method rather than a real model."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from open_road.dataset import DatasetSpec
from open_road.io import RunLayout, read_json
from open_road.method import MethodSpec
from open_road.stages.infer import run_infer
from open_road.stages.render import keep_components, run_render


class ConstantScorer:
    """Scores a fixed rectangle high and everything else low."""

    def __init__(self, config):
        self.high = float(config.get("high", 1.0))

    def score(self, image_bgr: np.ndarray) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        score = np.full((height, width), -1.0, dtype=np.float32)
        score[: height // 2, : width // 2] = self.high
        return score


def _spec(**overrides) -> MethodSpec:
    return MethodSpec(
        **{
            "name": "toy",
            "description": "a stand-in",
            "build": ConstantScorer,
            "score_range": (-1.0, 1.0),
            "default_threshold": 0.0,
            "default_min_area": 0,
            **overrides,
        }
    )


def _dataset(tmp_path: Path, frames: int = 2, size: int = 8) -> DatasetSpec:
    images = tmp_path / "original"
    images.mkdir(parents=True, exist_ok=True)
    for index in range(frames):
        cv2.imwrite(str(images / f"f{index}.jpg"), np.zeros((size, size, 3), dtype=np.uint8))
    return DatasetSpec.from_mapping({"name": "toy", "root": str(tmp_path), "labels_dir": None})


def test_keep_components_drops_specks_and_keeps_the_rest() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:4, 0:4] = True   # area 16
    mask[9, 9] = True       # area 1

    kept, regions = keep_components(mask, min_area=4)

    assert len(regions) == 1
    assert kept[9, 9] == False  # noqa: E712 -- explicit about the speck being gone
    assert kept[0:4, 0:4].all()
    _component, (x, y, width, height, area) = regions[0]
    assert (x, y, width, height, area) == (0, 0, 4, 4, 16)


def test_infer_writes_a_score_and_a_preview_per_frame(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    layout = RunLayout(tmp_path / "run")

    summary = run_infer(_spec(), dataset, layout, {}, report=lambda _: None)

    assert summary["scored"] == 2
    assert layout.scored_stems() == ["f0", "f1"]
    assert layout.soft_mask_path("f0").is_file()
    assert read_json(layout.manifest)["stages"]["infer"]["method"] == "toy"


def test_infer_is_resumable(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    layout = RunLayout(tmp_path / "run")
    run_infer(_spec(), dataset, layout, {}, report=lambda _: None)

    again = run_infer(_spec(), dataset, layout, {}, report=lambda _: None)

    assert again == {"frames": 2, "scored": 0, "skipped": 2}


def test_infer_rejects_a_map_that_is_not_the_frame_size(tmp_path: Path) -> None:
    class Wrong:
        def __init__(self, config):
            pass

        def score(self, image_bgr):
            return np.zeros((3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="must resize to the input frame"):
        run_infer(
            _spec(build=Wrong), _dataset(tmp_path), RunLayout(tmp_path / "run"), {},
            report=lambda _: None,
        )


def test_the_method_config_reaches_the_scorer(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    layout = RunLayout(tmp_path / "run")

    run_infer(_spec(defaults={"high": 0.25}), dataset, layout, {}, report=lambda _: None)
    default_max = float(np.load(layout.score_path("f0")).max())

    override = RunLayout(tmp_path / "run2")
    run_infer(_spec(defaults={"high": 0.25}), dataset, override, {"high": 0.75},
              report=lambda _: None)

    assert default_max == pytest.approx(0.25)
    assert float(np.load(override.score_path("f0")).max()) == pytest.approx(0.75)


def test_render_records_the_knobs_that_produced_it(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    layout = RunLayout(tmp_path / "run")
    run_infer(_spec(), dataset, layout, {}, report=lambda _: None)

    run_render(_spec(), dataset, layout, threshold=0.5, min_area=2, report=lambda _: None)

    regions = read_json(layout.regions)
    assert regions["threshold"] == 0.5
    assert regions["min_area"] == 2
    assert regions["method"] == "toy"
    # 8x8 frame, high quadrant is 4x4 = one 16-pixel component.
    assert regions["frames"]["f0"]["regions"][0]["area"] == 16
    assert layout.mask.joinpath("f0.png").is_file()
    assert layout.overlay.joinpath("f0.jpg").is_file()


def test_render_falls_back_to_the_methods_own_defaults(tmp_path: Path) -> None:
    # A threshold does not transfer between methods, so it is the method that
    # supplies it when a run does not.
    dataset = _dataset(tmp_path)
    layout = RunLayout(tmp_path / "run")
    run_infer(_spec(), dataset, layout, {}, report=lambda _: None)

    run_render(_spec(default_threshold=-0.5, default_min_area=99), dataset, layout,
               report=lambda _: None)

    regions = read_json(layout.regions)
    assert regions["threshold"] == -0.5
    assert regions["min_area"] == 99
    # min_area 99 exceeds the whole 8x8 frame, so nothing survives.
    assert regions["frames"]["f0"]["regions"] == []


def test_render_without_score_maps_says_to_infer_first(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="open-road infer"):
        run_render(_spec(), _dataset(tmp_path), RunLayout(tmp_path / "run"), report=lambda _: None)
