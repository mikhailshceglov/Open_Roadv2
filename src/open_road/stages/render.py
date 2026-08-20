"""Turn score maps into masks, regions and pictures.

A method emits one float per pixel; everything a downstream consumer actually
wants -- a binary mask, a list of objects, something to look at -- is produced
here, and none of it is what the network returned. The mask is one threshold
away from the score map, and the regions are one component filter after that.
Both knobs are recorded in ``regions.json`` so a picture can always be traced
back to the numbers that produced it.

Defaults come from the method, because they do not transfer between methods:
RbA's -0.0161 was swept on RoadAnomaly against an unbounded ``-tanh().sum()``
and is meaningless to a method emitting probabilities in [0, 1].
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from open_road.dataset import DatasetSpec
from open_road.io import RunLayout, load_image, load_score, update_manifest
from open_road.method import MethodSpec

FILL = (60, 60, 235)      # BGR, the anomaly fill
BOX = (0, 200, 255)       # box outline
EDGE = (255, 255, 255)    # mask contour
TRUTH = (80, 220, 80)     # ground-truth outline

DrawMode = str  # "seg" | "boxes" | "both"
Reporter = Callable[[str], None]


def keep_components(mask: np.ndarray, min_area: int) -> tuple[np.ndarray, list]:
    """The mask minus its specks, plus each survivor's bounding box and area."""
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    kept = np.zeros_like(mask)
    regions = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if area < min_area:
            continue
        component = labels == index
        kept |= component
        regions.append((component, (int(x), int(y), int(width), int(height), int(area))))
    return kept, regions


def label_box(canvas: np.ndarray, text: str, x: int, y: int, color) -> None:
    import cv2

    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    top = max(text_height + 8, y)
    cv2.rectangle(canvas, (x, top - text_height - 8), (x + text_width + 10, top + 2), color, -1)
    cv2.putText(canvas, text, (x + 5, top - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)


def draw(
    image: np.ndarray,
    mask: np.ndarray,
    regions: list,
    scores: np.ndarray,
    mode: DrawMode = "seg",
    alpha: float = 0.5,
    truth: np.ndarray | None = None,
) -> np.ndarray:
    """Overlay the mask (``seg``), the components (``boxes``) or both."""
    import cv2

    canvas = image.copy()

    if mode in ("seg", "both"):
        layer = np.zeros_like(canvas)
        layer[mask] = FILL
        canvas = cv2.addWeighted(canvas, 1.0, layer, alpha, 0)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, EDGE, 2)

    if mode in ("boxes", "both"):
        for component, (x, y, width, height, _area) in regions:
            cv2.rectangle(canvas, (x, y), (x + width, y + height), BOX, 2)
            label_box(canvas, f"{float(scores[component].mean()):.3f}", x, y, BOX)

    if truth is not None:
        outline, _ = cv2.findContours(
            truth.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, outline, -1, TRUTH, 2)

    return canvas


def run_render(
    spec: MethodSpec,
    dataset: DatasetSpec,
    layout: RunLayout,
    *,
    threshold: float | None = None,
    min_area: int | None = None,
    mode: DrawMode = "seg",
    alpha: float = 0.5,
    draw_labels: bool = True,
    report: Reporter = print,
) -> dict[str, Any]:
    """Render every score map in ``layout``, writing masks, overlays, regions."""
    import cv2

    cut = spec.default_threshold if threshold is None else threshold
    area = spec.default_min_area if min_area is None else min_area

    stems = set(layout.scored_stems())
    if not stems:
        raise SystemExit(f"no score maps in {layout.score_raw}; run `open-road infer` first")

    layout.mask.mkdir(parents=True, exist_ok=True)
    layout.overlay.mkdir(parents=True, exist_ok=True)
    show_truth = draw_labels and dataset.has_labels

    records: dict[str, Any] = {}
    total_regions = 0
    for path in dataset.frames():
        if path.stem not in stems:
            continue
        image = load_image(path)
        height, width = image.shape[:2]
        scores = load_score(layout.score_path(path.stem), shape=(height, width))

        mask, regions = keep_components(scores >= cut, area)

        truth = None
        if show_truth and dataset.label_path(path.stem).is_file():
            truth, _valid = dataset.load_label(path.stem, (height, width))

        cv2.imwrite(str(layout.mask / f"{path.stem}.png"), (mask * 255).astype(np.uint8))
        cv2.imwrite(
            str(layout.overlay / f"{path.stem}.jpg"),
            draw(image, mask, regions, scores, mode, alpha, truth),
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )

        records[path.stem] = {
            "width": width,
            "height": height,
            "regions": [
                {
                    "bbox": [x, y, x + w, y + h],
                    "area": region_area,
                    "score_mean": round(float(scores[component].mean()), 4),
                    "score_max": round(float(scores[component].max()), 4),
                }
                for component, (x, y, w, h, region_area) in regions
            ],
        }
        total_regions += len(regions)
        report(f"{path.name:<48}{len(regions):>3} region(s)")

    from open_road.io import write_json

    write_json(
        layout.regions,
        {"method": spec.name, "threshold": cut, "min_area": area, "frames": records},
    )
    report(f"\n{len(records)} frames, {total_regions} regions -> {layout.render}")

    summary = {
        "frames": len(records),
        "regions": total_regions,
        "threshold": cut,
        "min_area": area,
        "draw": mode,
    }
    update_manifest(layout, "render", summary)
    return summary
