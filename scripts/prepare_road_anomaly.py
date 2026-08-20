"""Lay the RoadAnomaly release out flat, with binary labels.

The published archive stores each frame's label under
``frames/<name>.labels/labels_semantic.png`` and encodes anomaly as **2**, not
1. Code assuming ``label == 1`` therefore reads an all-zero ground truth from
the raw download and happily reports metrics against nothing. This remaps 2 -> 1
and writes the two flat folders ``configs/datasets/road_anomaly.yaml`` expects.

    python scripts/prepare_road_anomaly.py --src <RoadAnomaly_jpg> --out data/road_anomaly

Download: https://datasets-cvlab.epfl.ch/2019-road-anomaly/RoadAnomaly_jpg.zip
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ANOMALY_VALUE = 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src", type=Path, required=True, help="Extracted RoadAnomaly_jpg folder")
    parser.add_argument("--out", type=Path, default=Path("data/road_anomaly"))
    args = parser.parse_args()

    frames_dir = args.src / "frames"
    images_out = args.out / "original"
    labels_out = args.out / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    names = json.loads((args.src / "frame_list.json").read_text(encoding="utf-8"))

    written = positive = total = 0
    for name in names:
        stem = Path(name).stem
        image_path = frames_dir / name
        label_path = frames_dir / f"{stem}.labels" / "labels_semantic.png"
        if not (image_path.is_file() and label_path.is_file()):
            print(f"[skip] {name}")
            continue

        shutil.copy(image_path, images_out / f"{stem}.jpg")

        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(str(image_path))
        if label.shape[:2] != image.shape[:2]:
            # A handful of frames were annotated at a different resolution.
            label = cv2.resize(
                label, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        binary = np.zeros_like(label)
        binary[label == ANOMALY_VALUE] = 1
        cv2.imwrite(str(labels_out / f"{stem}.png"), binary)

        positive += int((binary == 1).sum())
        total += binary.size
        written += 1

    print(f"frames:              {written}")
    print(f"anomalous pixels:    {100 * positive / max(total, 1):.2f}%")
    print(f"-> {images_out}\n-> {labels_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
