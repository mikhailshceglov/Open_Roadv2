"""Crops of (image, teacher anomaly map, teacher semantics) for training.

Sampling is biased towards anomalous regions.  With anomaly pixels well under
1% of the corpus, uniform crops would leave most batches with no positive
signal at all and the Dice term with nothing to work on.
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import list_corpus, target_path  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SEMANTIC_STRIDE = 4
ANOMALY_SEED_THRESHOLD = 0.5
POSITIVE_CROP_FRACTION = 0.5


class DistillationDataset(Dataset):
    def __init__(self, corpus_root: Path, targets_root: Path, crop: int = 768,
                 scale_range=(0.75, 1.5), seed: int = 0):
        self.corpus_root = Path(corpus_root)
        self.targets_root = Path(targets_root)
        self.crop = int(crop)
        self.scale_range = scale_range

        frames, _ = list_corpus(self.corpus_root)
        self.frames = [
            frame for frame in frames
            if target_path(frame, self.corpus_root, self.targets_root).exists()
        ]
        if not self.frames:
            raise RuntimeError(
                "No labelled frames under {}. Run the label stage first.".format(
                    self.targets_root
                )
            )
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.frames)

    def _load(self, index):
        frame = self.frames[index]
        image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Could not read image: {}".format(frame))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with np.load(target_path(frame, self.corpus_root, self.targets_root)) as payload:
            anomaly = payload["anomaly"].astype(np.float32)
            semantic = payload["semantic"].astype(np.float32)

        if anomaly.shape != image.shape[:2]:
            anomaly = cv2.resize(
                anomaly, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        return image, anomaly, semantic

    def _crop_origin(self, anomaly, height, width):
        """Top-left corner of the crop, biased towards anomalous pixels."""
        limit_y, limit_x = height - self.crop, width - self.crop
        if self.rng.random() < POSITIVE_CROP_FRACTION:
            ys, xs = np.nonzero(anomaly > ANOMALY_SEED_THRESHOLD)
            if len(ys):
                pick = self.rng.integers(len(ys))
                y = int(np.clip(ys[pick] - self.crop // 2, 0, limit_y))
                x = int(np.clip(xs[pick] - self.crop // 2, 0, limit_x))
                return y, x
        return int(self.rng.integers(limit_y + 1)), int(self.rng.integers(limit_x + 1))

    def __getitem__(self, index):
        image, anomaly, semantic = self._load(index)

        scale = self.rng.uniform(*self.scale_range)
        height = max(self.crop, int(round(image.shape[0] * scale)))
        width = max(self.crop, int(round(image.shape[1] * scale)))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        anomaly = cv2.resize(anomaly, (width, height), interpolation=cv2.INTER_LINEAR)
        # Semantics live at 1/4 and are resized to the crop's 1/4 grid directly,
        # so they never pass through a full-resolution intermediate.
        semantic = np.stack([
            cv2.resize(channel, (width // SEMANTIC_STRIDE, height // SEMANTIC_STRIDE),
                       interpolation=cv2.INTER_LINEAR)
            for channel in semantic
        ])

        y, x = self._crop_origin(anomaly, height, width)
        image = image[y : y + self.crop, x : x + self.crop]
        anomaly = anomaly[y : y + self.crop, x : x + self.crop]
        sy, sx, scrop = y // SEMANTIC_STRIDE, x // SEMANTIC_STRIDE, self.crop // SEMANTIC_STRIDE
        semantic = semantic[:, sy : sy + scrop, sx : sx + scrop]

        if self.rng.random() < 0.5:
            image = image[:, ::-1]
            anomaly = anomaly[:, ::-1]
            semantic = semantic[:, :, ::-1]

        if semantic.shape[-2:] != (scrop, scrop):
            pad_h = scrop - semantic.shape[-2]
            pad_w = scrop - semantic.shape[-1]
            semantic = np.pad(semantic, ((0, 0), (0, max(0, pad_h)), (0, max(0, pad_w))), mode="edge")
            semantic = semantic[:, :scrop, :scrop]

        image = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "pixel_values": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "anomaly": torch.from_numpy(np.ascontiguousarray(anomaly))[None],
            "semantic": torch.from_numpy(np.ascontiguousarray(semantic)),
        }
