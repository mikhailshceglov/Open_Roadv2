"""Encode a rendered clip back into a video.

A video dataset in this repository is a folder of frames, because that is how
the benchmarks ship them and because every other stage wants frames anyway. The
only thing missing afterwards is putting the overlays back together, which is
what this does.

Frame order comes from ``DatasetSpec.frames()`` rather than from globbing the
overlay directory: clips name frames by index without zero padding, so a
lexicographic listing runs 0, 1, 10, 100, 11 and silently scrambles the clip.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from open_road.dataset import DatasetSpec
from open_road.io import RunLayout, update_manifest

Reporter = Callable[[str], None]


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _encode_with_ffmpeg(frames: list[Path], destination: Path, fps: float) -> None:
    """H.264 from a zero-padded sequence symlinked in the order we want.

    The obvious alternatives both misbehave: a glob pattern hands ffmpeg the
    lexicographic order and undoes the point of sorting, and the concat demuxer
    ignores the final entry's duration unless the last file is repeated, which
    then encodes one frame too many. Numbering a temporary sequence makes both
    the order and the frame count exact.

    yuv420p because anything else fails to play in browsers and QuickTime, and
    the even-dimension filter because H.264 requires it.
    """
    import tempfile

    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        sequence = Path(staging)
        suffix = frames[0].suffix
        for index, path in enumerate(frames):
            (sequence / f"{index:06d}{suffix}").symlink_to(path.resolve())
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(fps),
                "-i", str(sequence / f"%06d{suffix}"),
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                str(destination),
            ],
            check=True,
        )


def _encode_with_opencv(frames: list[Path], destination: Path, fps: float) -> None:
    """Fallback when ffmpeg is absent. mp4v, which is more awkward to play."""
    import cv2

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise ValueError(f"unreadable overlay: {frames[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    try:
        for path in frames:
            image = cv2.imread(str(path))
            if image is None:
                continue
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(image)
    finally:
        writer.release()


def run_video(
    dataset: DatasetSpec,
    layout: RunLayout,
    *,
    fps: float = 10.0,
    source: str = "overlay",
    destination: Path | None = None,
    report: Reporter = print,
) -> dict[str, Any]:
    """Encode ``render/<source>/`` into an mp4, in the dataset's frame order."""
    directory = {"overlay": layout.overlay, "mask": layout.mask}.get(source)
    if directory is None:
        raise ValueError("source must be 'overlay' or 'mask'")
    if not directory.is_dir():
        raise SystemExit(f"nothing at {directory}; run `open-road render` first")

    suffix = ".jpg" if source == "overlay" else ".png"
    frames = [directory / f"{path.stem}{suffix}" for path in dataset.frames()]
    frames = [path for path in frames if path.is_file()]
    if not frames:
        raise SystemExit(f"no rendered frames in {directory}")

    destination = destination or layout.root / f"{dataset.name}_{source}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)

    encoder = "libx264" if has_ffmpeg() else "mp4v"
    if has_ffmpeg():
        _encode_with_ffmpeg(frames, destination, fps)
    else:
        report("ffmpeg not found; falling back to OpenCV's mp4v, which plays in fewer players")
        _encode_with_opencv(frames, destination, fps)

    size_mb = destination.stat().st_size / 1e6
    report(
        f"{len(frames)} frames at {fps:g} fps -> {destination} "
        f"({size_mb:.1f} MB, {len(frames) / fps:.1f}s, {encoder})"
    )

    summary = {
        "frames": len(frames),
        "fps": fps,
        "source": source,
        "encoder": encoder,
        "path": str(destination),
    }
    update_manifest(layout, "video", summary)
    return summary
