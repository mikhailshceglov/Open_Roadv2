"""Where a dataset's frames and labels live, and how its labels are encoded.

Every benchmark in this area lays itself out differently and encodes anomaly
differently, and those differences are the single most common source of silent
zero-metric runs. RoadAnomaly ships anomaly as **2**, so code assuming ``== 1``
reads an all-zero ground truth and cheerfully scores against nothing. SMIYC uses
1 for anomaly and 255 for void, and ignoring void inflates the false-positive
count with pixels nobody labelled.

So the layout is described, not renamed: a dataset is a small YAML file and the
shared stages read the description rather than assuming a convention.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from open_road.paths import resolve_path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class DatasetSpec:
    """One labelled (or unlabelled) folder of frames."""

    name: str
    root: Path
    images_dir: str = "original"
    labels_dir: str | None = "labels"
    label_suffix: str = ".png"
    anomaly_value: int | None = None
    """Pixel value meaning "anomaly". ``None`` accepts any non-zero value."""
    void_value: int | None = None
    """Pixel value meaning "unlabelled"; excluded from every metric."""
    pattern: str = ""
    """Keep only frames whose stem contains this. SMIYC uses ``validation``."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetSpec":
        config_path = resolve_path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"no dataset config at {config_path}")
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return cls.from_mapping(payload, source=config_path)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], source: Path | None = None) -> "DatasetSpec":
        known = {field.name for field in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            # The prototype's configs silently ignored unknown keys, so a whole
            # `rescan:` block was dropped without a word and the pipeline then
            # died on AttributeError far from the cause. Fail here instead.
            where = f" in {source}" if source else ""
            raise ValueError(
                f"unknown dataset key(s){where}: {', '.join(sorted(unknown))}. "
                f"Known keys: {', '.join(sorted(known))}"
            )
        values = dict(payload)
        if "root" not in values:
            raise ValueError(f"dataset config{f' {source}' if source else ''} has no 'root'")
        # Expanded so $ROAD_ANOMALY_ROOT can point a dataset elsewhere without
        # editing a tracked file -- which is what keeps machine-specific paths
        # out of the repository.
        values["root"] = resolve_path(os.path.expandvars(str(values["root"])))
        if "name" not in values and source is not None:
            values["name"] = source.stem
        return cls(**values)

    # -- layout ----------------------------------------------------------

    @property
    def images_path(self) -> Path:
        return self.root / self.images_dir

    @property
    def labels_path(self) -> Path | None:
        return None if self.labels_dir is None else self.root / self.labels_dir

    @property
    def has_labels(self) -> bool:
        path = self.labels_path
        return path is not None and path.is_dir()

    def frames(self) -> list[Path]:
        """Every frame, sorted, honouring ``pattern``."""
        directory = self.images_path
        if not directory.is_dir():
            raise FileNotFoundError(
                f"dataset {self.name!r}: no images at {directory}. "
                f"Run the dataset's prepare script, or fix 'root' in its config."
            )
        return sorted(
            path
            for path in directory.iterdir()
            # Skip dotfiles: macOS AppleDouble twins (._name) carry image
            # extensions, so a suffix check alone lets them through and every
            # imread on them returns None.
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in IMAGE_SUFFIXES
            and self.pattern in path.stem
        )

    def label_path(self, stem: str) -> Path:
        base = self.labels_path
        if base is None:
            raise ValueError(f"dataset {self.name!r} declares no labels_dir")
        return base / f"{stem}{self.label_suffix}"

    # -- labels ----------------------------------------------------------

    def load_label(self, stem: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(anomaly, valid)`` boolean masks at ``shape``.

        ``valid`` is False exactly where the label says void; metrics drop those
        pixels entirely rather than counting them as negatives.
        """
        import cv2

        path = self.label_path(stem)
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(f"unreadable label: {path}")
        if raw.shape[:2] != shape:
            # A handful of RoadAnomaly frames were annotated at another
            # resolution; nearest keeps the label values exact.
            raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

        if self.anomaly_value is None:
            anomaly = raw > 0
        else:
            anomaly = raw == self.anomaly_value

        if self.void_value is None:
            valid = np.ones_like(anomaly, dtype=bool)
        else:
            valid = raw != self.void_value
            anomaly &= valid

        return anomaly, valid
