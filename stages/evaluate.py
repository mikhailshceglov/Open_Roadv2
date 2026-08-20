"""Stage 3 — official SMIYC metrics for the student, plus teacher fidelity.

Reuses ``run_smiyc_eval.execute``: it already accepts a ``model_loader``, so the
student plugs into the official evaluation loop by exposing the one method the
loop calls, ``get_soft_mask(bgr)``.  No changes to the benchmark wrapper.

Two numbers come out of here:

* the official AUPR / FPR@95 / sIoU / PPV / F1 on the 40 labelled frames;
* fidelity -- how closely the student reproduces the teacher on frames nobody
  labelled.  With only 40 labelled frames, fidelity is what you watch while
  iterating, so the benchmark stays honest.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import _scripts_module, data_root, list_corpus, out_root, target_path  # noqa: E402
from student.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from student.model import Student  # noqa: E402

SIZE_MULTIPLE = 32


class StudentPredictor:
    """Adapts the student to the benchmark's ``get_soft_mask(bgr)`` contract."""

    def __init__(self, checkpoint: Path, device, short_side: int = 1024):
        self.device = device
        self.short_side = short_side
        self.model = Student().to(device).eval()
        state = torch.load(checkpoint, map_location=device)
        self.model.load_state_dict(state["model"])

    @torch.no_grad()
    def get_soft_mask(self, image_bgr):
        height, width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Only ever upscale: ObstacleTrack anomalies are tens of pixels across
        # and downsampling is how a student loses them.
        scale = max(1.0, self.short_side / min(height, width))
        target_h = int(np.ceil(height * scale / SIZE_MULTIPLE) * SIZE_MULTIPLE)
        target_w = int(np.ceil(width * scale / SIZE_MULTIPLE) * SIZE_MULTIPLE)
        resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1))[None].to(self.device)

        _, anomaly_logits = self.model(tensor)
        anomaly = F.interpolate(
            anomaly_logits.float(), size=(height, width), mode="bilinear", align_corners=False
        )
        return torch.sigmoid(anomaly)[0, 0].cpu().numpy()


def official_metrics(checkpoint: Path, datasets_root: Path, output: Path, device, short_side: int):
    smiyc = _scripts_module("run_smiyc_eval")
    predictor = StudentPredictor(checkpoint, device, short_side)

    args = SimpleNamespace(
        datasets_root=datasets_root,
        output=output,
        models=["student"],
        phase="all",
        # preflight only checks these exist; the student checkpoint is the
        # honest thing to point at, since no Mask2Former is loaded here.
        config_file=checkpoint,
        weights=checkpoint,
        visualize=False,
        masks=4,
        analysis_file=None,
    )
    return smiyc.execute(args, model_loader=lambda name, _args: predictor)


@torch.no_grad()
def fidelity(checkpoint: Path, corpus: Path, targets: Path, device, short_side: int, limit: int):
    """Rank agreement between student and teacher on unlabelled frames."""
    from scipy.stats import spearmanr

    predictor = StudentPredictor(checkpoint, device, short_side)
    frames, _ = list_corpus(corpus)
    frames = [f for f in frames if target_path(f, corpus, targets).exists()][:limit]
    if not frames:
        return None

    correlations = []
    rng = np.random.default_rng(0)
    for frame in frames:
        image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
        if image is None:
            continue
        student_map = predictor.get_soft_mask(image)
        with np.load(target_path(frame, corpus, targets)) as payload:
            teacher_map = payload["anomaly"].astype(np.float32)
        if teacher_map.shape != student_map.shape:
            teacher_map = cv2.resize(
                teacher_map, student_map.shape[::-1], interpolation=cv2.INTER_LINEAR
            )
        # Spearman over every pixel of every frame would be tens of billions of
        # pairs; a fixed random subsample is enough to track a training run.
        index = rng.choice(student_map.size, size=min(20000, student_map.size), replace=False)
        rho = spearmanr(student_map.ravel()[index], teacher_map.ravel()[index]).statistic
        if np.isfinite(rho):
            correlations.append(float(rho))

    if not correlations:
        return None
    return {"frames": len(correlations), "spearman_mean": float(np.mean(correlations))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=out_root() / "checkpoints" / "student_final.pt")
    parser.add_argument("--datasets-root", type=Path, default=data_root() / "smiyc")
    parser.add_argument("--corpus", type=Path, default=data_root() / "frames")
    parser.add_argument("--targets", type=Path, default=out_root() / "targets")
    parser.add_argument("--output", type=Path, default=out_root() / "eval")
    parser.add_argument("--short-side", type=int, default=int(os.environ.get("EVAL_SHORT_SIDE", "1024")))
    parser.add_argument("--fidelity-frames", type=int, default=int(os.environ.get("FIDELITY_FRAMES", "25")))
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError("Student checkpoint not found: {}".format(args.checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = official_metrics(args.checkpoint, args.datasets_root, args.output, device, args.short_side)
    scores = fidelity(args.checkpoint, args.corpus, args.targets, device, args.short_side, args.fidelity_frames)
    if scores:
        print("\nteacher fidelity on {} unlabelled frames: Spearman {:.4f}".format(
            scores["frames"], scores["spearman_mean"]), flush=True)

    payload = {"official": rows, "fidelity": scores}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "student_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
