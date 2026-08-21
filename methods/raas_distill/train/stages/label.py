"""Stage 1 — run the RAAS teacher over the corpus and store its predictions.

Two arrays per frame:

* ``anomaly``  — the teacher's ``soft_mask``, float16, full resolution.
* ``semantic`` — the 19-class Cityscapes output, float16, quarter resolution.

Quarter resolution for the semantics is deliberate: the student's decoder runs
at 1/4 anyway, and storing them full size would cost ~288 GB for a 3.6k-frame
corpus instead of ~18 GB.

The stage is resumable — a frame whose ``.npz`` already exists is skipped, so a
job killed halfway does not pay for the same GPU seconds twice.
"""

import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ...student.preprocess import SEMANTIC_STRIDE
from ..common import (
    data_root,
    list_corpus,
    load_teacher,
    out_root,
    soft_mask_of,
    target_path,
)


def semantic_targets(segmentation) -> np.ndarray:
    """[19, H, W] teacher semantics, average-pooled to 1/4 and cast to float16."""
    if segmentation is None or "sem_seg" not in segmentation:
        raise RuntimeError(
            "Teacher did not expose 'sem_seg'; the get_probs_and_seg wrapper in "
            "common.load_teacher is not in effect."
        )
    sem = segmentation["sem_seg"]
    if sem.ndim != 3:
        raise ValueError("Expected sem_seg of shape [C, H, W], got {}".format(tuple(sem.shape)))
    pooled = F.avg_pool2d(sem.unsqueeze(0).float(), SEMANTIC_STRIDE, ceil_mode=True)
    return pooled.squeeze(0).cpu().numpy().astype(np.float16)


def anomaly_target(soft_mask, height: int, width: int) -> np.ndarray:
    array = np.asarray(soft_mask, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError("Expected a 2-D anomaly map, got {}".format(array.shape))
    if not np.all(np.isfinite(array)):
        raise ValueError("Anomaly map contains NaN or infinity")
    if array.shape != (height, width):
        array = cv2.resize(array, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(array, 0.0, 1.0).astype(np.float16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default=os.environ.get("TEACHER", "maskomaly_id"))
    parser.add_argument("--corpus", type=Path, default=data_root() / "frames")
    parser.add_argument("--targets", type=Path, default=out_root() / "targets")
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.environ.get("LABEL_LIMIT", "0")),
        help="Label at most N frames (0 = all). Used for smoke tests.",
    )
    args = parser.parse_args()

    frames, held_out = list_corpus(args.corpus)
    if args.limit:
        frames = frames[: args.limit]
    print(
        "corpus: {} training frames, {} validation frames held out".format(
            len(frames), len(held_out)
        ),
        flush=True,
    )

    pending = [f for f in frames if not target_path(f, args.corpus, args.targets).exists()]
    print("{} already labelled, {} to go".format(len(frames) - len(pending), len(pending)), flush=True)
    if not pending:
        print("nothing to do", flush=True)
        return 0

    model = load_teacher(args.teacher)
    durations = []

    for index, frame in enumerate(pending, start=1):
        image = cv2.imread(str(frame), cv2.IMREAD_COLOR)  # BGR, as the teacher expects
        if image is None:
            raise RuntimeError("Could not read image: {}".format(frame))
        height, width = image.shape[:2]

        started = time.time()
        soft_mask = soft_mask_of(model, image)
        anomaly = anomaly_target(soft_mask, height, width)
        semantic = semantic_targets(model.last_segmentation)
        elapsed = time.time() - started
        durations.append(elapsed)

        destination = target_path(frame, args.corpus, args.targets)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary name first: a job killed mid-write must not leave
        # a truncated .npz that the resume logic would then treat as complete.
        staging = destination.with_suffix(".npz.partial")
        # Write through an open handle: np.savez_compressed appends '.npz' to a
        # path that does not already end in it, which would leave the archive at
        # <name>.npz.partial.npz and break the rename below.
        with staging.open("wb") as handle:
            np.savez_compressed(handle, anomaly=anomaly, semantic=semantic)
        staging.replace(destination)

        if index % 10 == 0 or index == len(pending):
            mean = sum(durations) / len(durations)
            remaining = (len(pending) - index) * mean
            print(
                "[{}/{}] {}  {:.2f}s  (mean {:.2f}s, ~{:.0f}s left)".format(
                    index, len(pending), frame.name, elapsed, mean, remaining
                ),
                flush=True,
            )

    stats = {
        "frames_labelled": len(pending),
        "seconds_per_frame_mean": sum(durations) / len(durations),
        "seconds_total": sum(durations),
        "teacher": args.teacher,
    }
    args.targets.mkdir(parents=True, exist_ok=True)
    (args.targets / "label_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print("label stage done: {}".format(json.dumps(stats)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
