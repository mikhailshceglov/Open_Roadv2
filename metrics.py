"""Score the student against labelled frames.

Self-contained: unlike ``stages/evaluate.py``, which plugs into the RAAS
repository's evaluation loop, this needs nothing but the weights and a folder of
labelled images.  It is the script to reach for when checking a checkpoint, a
resolution, or a change to post-processing.

    python3 metrics.py --dataset /data/dataset_ObstacleTrack
    python3 metrics.py --dataset /data/dataset_AnomalyTrack --short-side 1024 736 544

Defaults follow the SegmentMeIfYouCan layout::

    <dataset>/images/validation0000.jpg
    <dataset>/labels_masks/validation0000_labels_semantic.png

with 1 marking anomaly and 255 marking void.  Every part of that is a flag, so
other datasets work by describing them rather than by renaming files.

The pixel metrics match the definitions used by the RAAS project, so numbers are
comparable with the teacher's:

* **AUPR** — area under the precision/recall curve.  The headline number; it is
  the one that survives extreme class imbalance.
* **FPR@95** — false positive rate at the threshold where 95% of anomalous
  pixels are caught.  What decides whether a system is deployable: an AUPR that
  looks fine can hide a model that floods the frame at usable recall.
* **AUROC**, **AP** — reported for completeness.

Both are threshold-free: they depend only on how the model *ranks* pixels, so a
model whose scores are not calibrated is not penalised.  ``--best-f1`` adds a
threshold sweep for when a single operating point is what you need.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from infer import AnomalySegmenter, IMAGE_SUFFIXES  # noqa: E402


def auroc_and_fpr95(gt, score):
    """AUROC, plus the false positive rate at 95% true positive rate."""
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(gt, score)
    at95 = 0.0
    for true_positive, false_positive in zip(tpr, fpr):
        if true_positive > 0.95:
            at95 = false_positive
            break
    return float(auc(fpr, tpr)), float(at95)


def aupr(gt, score):
    from sklearn.metrics import auc, precision_recall_curve

    precision, recall, _ = precision_recall_curve(gt, score)
    order = recall.argsort()
    return float(auc(recall[order], precision[order]))


def best_f1(gt, score, steps=100):
    """Highest F1 over a threshold sweep, and the threshold that reached it."""
    from sklearn.metrics import f1_score

    best, best_threshold = 0.0, 0.0
    for index in range(1, steps):
        threshold = index / steps
        value = f1_score(gt, score >= threshold, zero_division=0)
        if value > best:
            best, best_threshold = float(value), threshold
    return best, best_threshold


def collect(dataset: Path, pattern: str, label_suffix: str, images_dir: str, labels_dir: str):
    images = sorted(
        path for path in (dataset / images_dir).iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES and path.stem.startswith(pattern)
    )
    pairs = []
    for path in images:
        label = dataset / labels_dir / "{}{}".format(path.stem, label_suffix)
        if label.is_file():
            pairs.append((path, label))
    return pairs


def score_dataset(segmenter, pairs, anomaly_value, void_value):
    """Pool every valid pixel across frames, and keep per-frame AP alongside."""
    from sklearn.metrics import average_precision_score

    truths, scores, per_frame, elapsed = [], [], [], []
    for index, (image_path, label_path) in enumerate(pairs, 1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if image is None or label is None:
            print("  skipped (unreadable): {}".format(image_path.name))
            continue
        if label.shape != image.shape[:2]:
            label = cv2.resize(label, image.shape[1::-1], interpolation=cv2.INTER_NEAREST)

        started = time.perf_counter()
        score = segmenter(image)
        elapsed.append(time.perf_counter() - started)

        valid = label != void_value
        truth = (label[valid] == anomaly_value).astype(np.uint8)
        flat = score[valid].astype(np.float32)
        truths.append(truth)
        scores.append(flat)
        per_frame.append({
            "frame": image_path.stem,
            "AP": 100 * float(average_precision_score(truth, flat)) if truth.any() else None,
            "anomaly_share": 100 * float(truth.mean()),
        })
        print("  [{}/{}] {}".format(index, len(pairs), image_path.stem), flush=True)

    if not truths:
        raise SystemExit("No usable image/label pairs found")
    return np.concatenate(truths), np.concatenate(scores), per_frame, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).resolve().parent / "weights" / "student_final.pt")
    parser.add_argument("--short-side", type=int, nargs="+", default=[1024],
                        help="one value, or several to sweep")
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--labels-dir", default="labels_masks")
    parser.add_argument("--label-suffix", default="_labels_semantic.png")
    parser.add_argument("--pattern", default="validation",
                        help="only score frames whose name starts with this; '' for all")
    parser.add_argument("--anomaly-value", type=int, default=1)
    parser.add_argument("--void-value", type=int, default=255)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--best-f1", action="store_true", help="also sweep for the best F1 threshold")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    pairs = collect(args.dataset, args.pattern, args.label_suffix, args.images_dir, args.labels_dir)
    if not pairs:
        raise SystemExit(
            "No labelled frames under {}. Expected {}/<{}*> and {}/<stem>{}".format(
                args.dataset, args.images_dir, args.pattern, args.labels_dir, args.label_suffix))
    print("{}: {} labelled frames".format(args.dataset.name, len(pairs)))

    report = {"dataset": str(args.dataset), "frames": len(pairs), "runs": {}}
    for short_side in args.short_side:
        print("\nshort side {}".format(short_side))
        segmenter = AnomalySegmenter(args.checkpoint, args.device, short_side, args.half)
        truth, score, per_frame, elapsed = score_dataset(
            segmenter, pairs, args.anomaly_value, args.void_value)

        from sklearn.metrics import average_precision_score
        roc_auc, fpr95 = auroc_and_fpr95(truth, score)
        entry = {
            "AUPR": 100 * aupr(truth, score),
            "FPR@95": 100 * fpr95,
            "AUROC": 100 * roc_auc,
            "AP": 100 * float(average_precision_score(truth, score)),
            "anomaly_pixel_share": 100 * float(truth.mean()),
            "ms_per_frame": 1000 * float(np.mean(elapsed[1:] or elapsed)),
            "per_frame": per_frame,
        }
        if args.best_f1:
            entry["best_F1"], entry["best_F1_threshold"] = best_f1(truth, score)
        report["runs"][str(short_side)] = entry

        print("  AUPR {:.2f}  FPR@95 {:.2f}  AUROC {:.2f}  AP {:.2f}  ({:.0f} ms/frame)".format(
            entry["AUPR"], entry["FPR@95"], entry["AUROC"], entry["AP"], entry["ms_per_frame"]))
        if args.best_f1:
            print("  best F1 {:.2f} at threshold {:.2f}".format(
                100 * entry["best_F1"], entry["best_F1_threshold"]))

    if len(args.short_side) > 1:
        print("\n{:>11}{:>9}{:>10}{:>9}{:>12}".format("short side", "AUPR", "FPR@95", "AUROC", "ms/frame"))
        for short_side, entry in report["runs"].items():
            print("{:>11}{:>9.2f}{:>10.2f}{:>9.2f}{:>12.0f}".format(
                short_side, entry["AUPR"], entry["FPR@95"], entry["AUROC"], entry["ms_per_frame"]))

    worst = sorted((f for f in report["runs"][str(args.short_side[0])]["per_frame"]
                    if f["AP"] is not None), key=lambda f: f["AP"])[:5]
    if worst:
        print("\nweakest frames (AP, and the share of the frame that is anomalous):")
        for frame in worst:
            print("  {:<26}{:>7.2f}{:>9.2f}%".format(frame["frame"], frame["AP"], frame["anomaly_share"]))
        print("  Note: per-frame AP is bounded below by the anomaly share, so a frame with"
              "\n  few anomalous pixels scores lower than one full of them at equal quality.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("\nwritten: {}".format(args.json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
