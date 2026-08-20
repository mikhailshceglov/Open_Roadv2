"""Reading frames, writing artefacts, and the layout of a run directory.

A run writes to one place and always the same shape, so that render, evaluate
and report can be method-agnostic:

    runs/<method>/<dataset>/
      score_raw/<stem>.npy            float32 HxW, the method's own output
      soft_mask/<stem>_soft_mask.png  8-bit preview, normalised by score_range
      render/mask/<stem>.png          thresholded binary mask
      render/overlay/<stem>.jpg
      render/regions.json
      intermediate/<stem>/*.png       a method's internals, when it exposes any
      metrics.json
      run.json                        what was run, with what, and when

``score_raw`` is the only artefact that is not derived: everything below it is
one threshold and one component filter away, and both are recorded in
``regions.json`` so a rendering can always be traced back to its knobs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCORE_SUFFIX = ".npy"
SOFT_MASK_SUFFIX = "_soft_mask.png"


def load_image(path: str | Path) -> np.ndarray:
    """A frame as uint8 BGR, the way every method here expects it."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unreadable image: {path}")
    return image


def save_score(path: Path, score: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, score.astype(np.float32))
    return path


def load_score(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """A stored score map, resized to ``shape`` if it does not already match."""
    import cv2

    score = np.load(path).astype(np.float32)
    if shape is not None and score.shape != shape:
        score = cv2.resize(score, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return score


def save_soft_mask(path: Path, unit_score: np.ndarray) -> Path:
    """Write the ``[0, 1]`` preview as an 8-bit PNG."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.clip(unit_score, 0.0, 1.0) * 255).astype(np.uint8))
    return path


def save_heatmap(path: Path, values: np.ndarray) -> Path:
    """Write a float map as a min-max normalised colour image.

    For looking at, not for measuring: the normalisation is per-image, so two
    frames' heatmaps are not comparable. The comparable artefact is the .npy.
    """
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    finite = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = float(finite.min()), float(finite.max())
    unit = np.zeros_like(finite) if high - low < 1e-8 else (finite - low) / (high - low)
    cv2.imwrite(str(path), cv2.applyColorMap((unit * 255).astype(np.uint8), cv2.COLORMAP_INFERNO))
    return path


def save_binary(path: Path, mask: np.ndarray) -> Path:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.asarray(mask).astype(bool) * 255).astype(np.uint8))
    return path


def save_debug(out_dir: Path, debug: dict[str, Any]) -> Path:
    """Write whatever a Scorer left behind, dispatching on type.

    Methods expose their internals by setting ``last_debug`` after ``score()``;
    this turns that dict into files without the stage knowing what any of the
    entries mean. Boolean arrays become binary masks, float arrays heatmaps,
    3-channel uint8 arrays images, and everything else JSON.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, value in debug.items():
        if isinstance(value, np.ndarray):
            if value.dtype == bool:
                save_binary(out_dir / f"{name}.png", value)
            elif value.ndim == 3 and value.shape[2] == 3 and value.dtype == np.uint8:
                import cv2

                cv2.imwrite(str(out_dir / f"{name}.png"), value)
            else:
                save_heatmap(out_dir / f"{name}.png", value)
        else:
            write_json(out_dir / f"{name}.json", value)
    return out_dir


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RunLayout:
    """The paths a run owns. Constructing one creates nothing."""

    root: Path

    @property
    def score_raw(self) -> Path:
        return self.root / "score_raw"

    @property
    def soft_mask(self) -> Path:
        return self.root / "soft_mask"

    @property
    def render(self) -> Path:
        return self.root / "render"

    @property
    def mask(self) -> Path:
        return self.render / "mask"

    @property
    def overlay(self) -> Path:
        return self.render / "overlay"

    @property
    def regions(self) -> Path:
        return self.render / "regions.json"

    @property
    def intermediate(self) -> Path:
        """Per-frame internals, when a method chooses to expose them."""
        return self.root / "intermediate"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.json"

    @property
    def manifest(self) -> Path:
        return self.root / "run.json"

    def score_path(self, stem: str) -> Path:
        return self.score_raw / f"{stem}{SCORE_SUFFIX}"

    def soft_mask_path(self, stem: str) -> Path:
        return self.soft_mask / f"{stem}{SOFT_MASK_SUFFIX}"

    def scored_stems(self) -> list[str]:
        if not self.score_raw.is_dir():
            return []
        return sorted(path.stem for path in self.score_raw.glob(f"*{SCORE_SUFFIX}"))


def update_manifest(layout: RunLayout, stage: str, payload: dict[str, Any]) -> Path:
    """Record one stage in ``run.json`` without discarding the others."""
    from datetime import datetime, timezone

    manifest: dict[str, Any] = {}
    if layout.manifest.is_file():
        try:
            manifest = read_json(layout.manifest)
        except (json.JSONDecodeError, OSError):
            manifest = {}
    stages = manifest.setdefault("stages", {})
    stages[stage] = {"finished_at": datetime.now(timezone.utc).isoformat(), **payload}
    return write_json(layout.manifest, manifest)
