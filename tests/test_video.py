"""Frame ordering and video encoding.

Ordering is the whole risk here: a clip encoded in lexicographic order is not
obviously broken in any single frame, it just plays as nonsense.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from open_road.dataset import DatasetSpec, natural_key
from open_road.io import RunLayout
from open_road.stages.video import has_ffmpeg, run_video


def _clip(tmp_path: Path, count: int = 12) -> DatasetSpec:
    for index in range(count):
        cv2.imwrite(str(tmp_path / f"{index}.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))
    return DatasetSpec.from_mapping(
        {"name": "clip", "root": str(tmp_path), "images_dir": ".", "labels_dir": None}
    )


def test_natural_key_orders_digits_as_numbers() -> None:
    names = [Path(f"{n}.jpg") for n in ("10", "2", "1", "100", "20")]

    assert [p.stem for p in sorted(names, key=natural_key)] == ["1", "2", "10", "20", "100"]


def test_frames_of_an_unpadded_clip_are_in_playback_order(tmp_path: Path) -> None:
    # Lexicographically this is 0, 1, 10, 11, 2, ... which plays as nonsense.
    frames = _clip(tmp_path).frames()

    assert [p.stem for p in frames] == [str(n) for n in range(12)]


def test_mixed_text_and_digits_still_sorts_sensibly() -> None:
    names = [Path(n) for n in ("frame_10.png", "frame_2.png", "frame_1.png")]

    assert [p.stem for p in sorted(names, key=natural_key)] == [
        "frame_1", "frame_2", "frame_10",
    ]


def test_encoding_needs_a_render_first(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="render"):
        run_video(_clip(tmp_path), RunLayout(tmp_path / "run"), report=lambda _: None)


def test_encoding_writes_a_playable_file_with_every_frame(tmp_path: Path) -> None:
    dataset = _clip(tmp_path)
    layout = RunLayout(tmp_path / "run")
    layout.overlay.mkdir(parents=True)
    for index in range(12):
        # A distinct brightness per frame, so a scrambled order would be
        # detectable by reading the decoded clip back.
        frame = np.full((8, 8, 3), index * 20, dtype=np.uint8)
        cv2.imwrite(str(layout.overlay / f"{index}.jpg"), frame)

    summary = run_video(dataset, layout, fps=5.0, report=lambda _: None)

    assert summary["frames"] == 12
    assert summary["encoder"] == ("libx264" if has_ffmpeg() else "mp4v")
    destination = Path(summary["path"])
    assert destination.is_file() and destination.stat().st_size > 0

    capture = cv2.VideoCapture(str(destination))
    decoded = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    assert decoded == 12


def test_an_unknown_source_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="overlay.*mask"):
        run_video(_clip(tmp_path), RunLayout(tmp_path / "run"), source="heatmap")
