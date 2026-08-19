"""Shared, dependency-light cache contract for the RAAS/Objectomaly bridge."""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np


SCHEMA_VERSION = 1


def validate_frame_id(frame_id: str) -> str:
    """Reject identifiers that could escape the cache directory."""
    value = str(frame_id)
    if not value or Path(value).name != value or value in (".", ".."):
        raise ValueError("Unsafe frame id: {!r}".format(value))
    return value


def validate_anomaly_map(value: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape != tuple(target_hw):
        raise ValueError(
            "Expected anomaly map {}, got {}".format(tuple(target_hw), result.shape)
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("Anomaly map contains NaN or infinity")
    if result.size and (float(result.min()) < 0.0 or float(result.max()) > 1.0):
        raise ValueError("Anomaly map must be in [0, 1]")
    return np.ascontiguousarray(result, dtype=np.float32)


def validate_bgr_image(value: np.ndarray, target_hw=None) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected BGR HxWx3 image, got {}".format(image.shape))
    if target_hw is not None and image.shape[:2] != tuple(target_hw):
        raise ValueError(
            "Expected image {}, got {}".format(tuple(target_hw), image.shape[:2])
        )
    if image.dtype != np.uint8:
        raise ValueError("Expected uint8 image, got {}".format(image.dtype))
    return np.ascontiguousarray(image)


def write_manifest(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = dict(payload)
    document["schema_version"] = SCHEMA_VERSION
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def read_manifest(path: Path) -> Dict[str, object]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported manifest schema {} in {}".format(
                payload.get("schema_version"), path
            )
        )
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Manifest must contain an entries list: {}".format(path))
    return payload


def index_entries(entries: Iterable[Mapping[str, object]]) -> Dict[Tuple[str, str], Mapping[str, object]]:
    result = {}
    for entry in entries:
        key = (
            validate_frame_id(str(entry["dataset"])),
            validate_frame_id(str(entry["fid"])),
        )
        if key in result:
            raise ValueError("Duplicate manifest entry: {} / {}".format(*key))
        result[key] = entry
    return result


def resolve_cached_path(manifest_path: Path, relative_path: str) -> Path:
    root = Path(manifest_path).resolve().parent
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Cache path escapes manifest directory: {}".format(relative_path)) from exc
    return candidate
