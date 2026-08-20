"""Anomaly-segmentation metrics, pixel-level and component-level.

Pixel metrics are the usual three -- AP (area under precision-recall), AUROC,
and FPR95 (false-positive rate at 95% true-positive rate) -- pooled over every
valid pixel in the dataset rather than averaged per frame. Pooling is what the
benchmarks report; per-frame averaging gives a systematically different number
because frames differ enormously in how much anomaly they contain.

The component metrics follow SegmentMeIfYouCan (Chan et al., 2021). They exist
because pixel AP rewards covering one large object and says nothing about
whether each object was found:

  sIoU_gt  per ground-truth component, its IoU against the union of predicted
           components that touch it, with pixels belonging to *other* ground
           truth components excluded from the union -- so one prediction
           spanning two objects is not punished twice.
  PPV      per predicted component, the fraction of it landing on any
           ground-truth component. Precision, per object.
  F1*      averaged over thresholds tau in [0.25, 0.75]: a ground-truth
           component counts as found when sIoU > tau, a prediction counts as
           false when PPV <= tau. Averaging over tau avoids tuning one
           arbitrary overlap cutoff.

Void pixels are dropped from every metric. Datasets that mark them (SMIYC uses
255) label pixels nobody could annotate; counting them as negatives inflates
the false-positive rate with pixels that were never anybody's to get right.

These are this repository's own metrics, not the official SMIYC evaluator.
They are internally consistent and fine for comparing methods here; they are
not a benchmark submission, and the component numbers in particular apply no
minimum component size, so they are not paper-comparable.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from open_road.dataset import DatasetSpec
from open_road.io import RunLayout, load_image, load_score, update_manifest, write_json

TAUS = np.arange(0.25, 0.76, 0.05)
Reporter = Callable[[str], None]


def pixel_metrics(gt: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Threshold-free ranking metrics, plus the best-F1 operating point."""
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    ap = float(average_precision_score(gt, score))
    auroc = float(roc_auc_score(gt, score))

    fpr, tpr, _ = roc_curve(gt, score)
    fpr95 = float(np.interp(0.95, tpr, fpr))

    precision, recall, thresholds = precision_recall_curve(gt, score)
    # precision_recall_curve returns one more point than it has thresholds.
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.zeros_like(denominator)
    np.divide(2 * precision * recall, denominator, out=f1, where=denominator > 0)
    best = int(f1.argmax())

    return {
        "AP": ap,
        "AUROC": auroc,
        "FPR95": fpr95,
        "F1": float(f1[best]),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "threshold": float(thresholds[best]),
    }


def component_metrics(gt: np.ndarray, pred: np.ndarray) -> tuple[list[float], list[float]]:
    """sIoU per ground-truth component and PPV per predicted component, one frame."""
    import cv2

    n_gt, gt_labels = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
    n_pred, pred_labels = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)

    sious: list[float] = []
    for index in range(1, n_gt):
        component = gt_labels == index
        touching = np.unique(pred_labels[component])
        touching = touching[touching > 0]
        if touching.size == 0:
            sious.append(0.0)
            continue
        union_pred = np.isin(pred_labels, touching)
        intersection = np.logical_and(component, union_pred)
        union = np.logical_or(component, union_pred)
        # Exclude pixels owned by other GT components -- the "adjusted" in sIoU.
        others = np.logical_and(gt, ~component)
        union = np.logical_and(union, ~others)
        sious.append(float(intersection.sum()) / max(int(union.sum()), 1))

    ppvs: list[float] = []
    for index in range(1, n_pred):
        component = pred_labels == index
        ppvs.append(float(np.logical_and(component, gt).sum()) / max(int(component.sum()), 1))

    return sious, ppvs


def summarise_components(sious: np.ndarray, ppvs: np.ndarray) -> dict[str, Any]:
    f1_star = []
    for tau in TAUS:
        true_positive = int((sious > tau).sum())
        false_negative = int((sious <= tau).sum())
        false_positive = int((ppvs <= tau).sum())
        denominator = 2 * true_positive + false_negative + false_positive
        f1_star.append(2 * true_positive / denominator if denominator else 0.0)

    # The conventional single operating point.
    tp = int((sious > 0.5).sum())
    fn = int((sious <= 0.5).sum())
    fp = int((ppvs <= 0.5).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return {
        "sIoU_gt": float(sious.mean()) if sious.size else 0.0,
        "PPV": float(ppvs.mean()) if ppvs.size else 0.0,
        "F1_star": float(np.mean(f1_star)),
        "gt_components": int(sious.size),
        "predicted_components": int(ppvs.size),
        "tau_0.5": {
            "precision": precision,
            "recall": recall,
            "F1": 2 * precision * recall / max(precision + recall, 1e-9),
            "TP": tp,
            "FP": fp,
            "FN": fn,
        },
    }


def run_evaluate(
    dataset: DatasetSpec,
    layout: RunLayout,
    *,
    threshold: float | None = None,
    min_area: int = 0,
    method: str | None = None,
    report: Reporter = print,
) -> dict[str, Any]:
    """Score ``layout``'s maps against ``dataset``'s labels into ``metrics.json``.

    ``min_area`` must be the same filter ``render`` applies. Scoring the raw
    thresholded mask while shipping a filtered one measures a mask nobody would
    deploy: on RoadAnomaly the unfiltered mask has 774 predicted components
    against 298 ground-truth ones, and PPV is dominated by specks that never
    reach the output.

    No method is loaded: evaluation reads only ``score_raw/``, which is why it
    runs in a plain environment even when the method's own pins do not.
    """
    if not dataset.has_labels:
        raise SystemExit(f"dataset {dataset.name!r} has no labels; nothing to evaluate against")

    stems = set(layout.scored_stems())
    if not stems:
        raise SystemExit(f"no score maps in {layout.score_raw}; run `open-road infer` first")

    frames: list[tuple[str, np.ndarray, np.ndarray]] = []
    skipped: list[str] = []
    for path in dataset.frames():
        if path.stem not in stems:
            skipped.append(path.stem)
            continue
        if not dataset.label_path(path.stem).is_file():
            skipped.append(path.stem)
            continue
        height, width = load_image(path).shape[:2]
        anomaly, valid = dataset.load_label(path.stem, (height, width))
        score = load_score(layout.score_path(path.stem), shape=(height, width))
        frames.append((path.stem, anomaly, np.where(valid, score, np.nan)))

    if not frames:
        raise SystemExit("no frame had both a score map and a label")
    if skipped:
        report(f"skipped {len(skipped)} frame(s) with no score map or no label")

    # Pool valid pixels only. NaN marks void, put there above.
    gt_all = np.concatenate([anomaly.ravel() for _stem, anomaly, _score in frames])
    score_all = np.concatenate([score.ravel() for _stem, _anomaly, score in frames])
    keep = ~np.isnan(score_all)
    gt_all, score_all = gt_all[keep], score_all[keep]

    if not gt_all.any():
        raise SystemExit(
            "ground truth is empty across the whole dataset. Check 'anomaly_value' in "
            "the dataset config -- RoadAnomaly encodes anomaly as 2, not 1."
        )

    pixel = pixel_metrics(gt_all, score_all)
    cut = pixel["threshold"] if threshold is None else threshold

    from sklearn.metrics import average_precision_score

    from open_road.stages.render import keep_components

    all_sious: list[float] = []
    all_ppvs: list[float] = []
    per_frame: list[dict[str, Any]] = []
    for stem, anomaly, score in frames:
        valid = ~np.isnan(score)
        prediction = np.where(valid, score >= cut, False)
        if min_area > 0:
            # The same filter render applies, so the mask that is scored is the
            # mask that is shipped.
            prediction, _regions = keep_components(prediction, min_area)
        sious, ppvs = component_metrics(anomaly, prediction)
        all_sious.extend(sious)
        all_ppvs.extend(ppvs)

        intersection = float(np.logical_and(anomaly, prediction).sum())
        f1_denominator = float(anomaly.sum() + prediction.sum())
        # Per-frame AP is undefined without both classes present, which is why
        # it is None rather than 0 on a frame that is entirely background.
        frame_ap = None
        if anomaly[valid].any() and not anomaly[valid].all():
            frame_ap = round(float(average_precision_score(anomaly[valid], score[valid])), 4)
        per_frame.append(
            {
                "frame": stem,
                "AP": frame_ap,
                "F1": round(2 * intersection / f1_denominator, 4) if f1_denominator else 1.0,
                "anomaly_share": round(float(anomaly.sum()) / max(int(valid.sum()), 1), 6),
            }
        )

    component = summarise_components(np.asarray(all_sious), np.asarray(all_ppvs))

    prediction_all = score_all >= cut
    metrics = {
        "method": method,
        "dataset": dataset.name,
        "frames": len(frames),
        "threshold": float(cut),
        "min_area": int(min_area),
        "pixel": {
            **{key: round(value, 6) for key, value in pixel.items()},
            "TP": int((prediction_all & gt_all).sum()),
            "FP": int((prediction_all & ~gt_all).sum()),
            "FN": int((~prediction_all & gt_all).sum()),
            "void_pixels": int((~keep).sum()),
        },
        "component": component,
        "per_frame": sorted(per_frame, key=lambda row: row["F1"]),
    }

    write_json(layout.metrics, metrics)
    _report(metrics, report)
    update_manifest(
        layout,
        "evaluate",
        {"threshold": float(cut), "min_area": int(min_area), "frames": len(frames)},
    )
    return metrics


def _report(metrics: dict[str, Any], report: Reporter) -> None:
    pixel = metrics["pixel"]
    component = metrics["component"]
    name = metrics.get("method") or "run"

    report(f"\n{name} on {metrics['dataset']} — {metrics['frames']} frames, "
           f"threshold {metrics['threshold']:.4f}, min_area {metrics['min_area']}")
    report("  pixel")
    for key in ("AP", "AUROC", "FPR95", "precision", "recall", "F1"):
        report(f"    {key:<12} {100 * pixel[key]:6.2f}%")
    if pixel["void_pixels"]:
        report(f"    {'void':<12} {pixel['void_pixels']} pixels excluded")
    report("  component (SegmentMeIfYouCan)")
    report(f"    {'sIoU_gt':<12} {100 * component['sIoU_gt']:6.2f}%   "
           f"({component['gt_components']} ground-truth components)")
    report(f"    {'PPV':<12} {100 * component['PPV']:6.2f}%   "
           f"({component['predicted_components']} predicted)")
    report(f"    {'F1*':<12} {100 * component['F1_star']:6.2f}%   (mean over tau 0.25..0.75)")

    weakest = metrics["per_frame"][:5]
    if weakest:
        report("  weakest frames")
        for row in weakest:
            report(f"    {row['frame']:<40} F1 {100 * row['F1']:6.2f}%")
