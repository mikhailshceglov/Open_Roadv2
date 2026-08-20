"""Run a method over a dataset and store one score map per frame.

This is the only stage that touches a model. Everything downstream reads
``score_raw/*.npy``, which means render and evaluate never need the method's
environment -- useful, since the methods' dependency pins do not co-exist.

The stage is resumable: a frame whose ``.npy`` already exists is skipped unless
``overwrite`` is set, so a run interrupted at frame 50 of 500 does not re-pay
for the first 50.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

from open_road.dataset import DatasetSpec
from open_road.io import (
    RunLayout,
    load_image,
    save_debug,
    save_score,
    save_soft_mask,
    update_manifest,
)
from open_road.method import MethodSpec

Reporter = Callable[[str], None]


def run_infer(
    spec: MethodSpec,
    dataset: DatasetSpec,
    layout: RunLayout,
    config: Mapping[str, Any] | None = None,
    *,
    limit: int = 0,
    overwrite: bool = False,
    save_intermediate: bool = True,
    report: Reporter = print,
) -> dict[str, Any]:
    """Score every frame in ``dataset``, writing into ``layout``."""
    frames = dataset.frames()
    if limit > 0:
        frames = frames[:limit]
    if not frames:
        raise SystemExit(f"dataset {dataset.name!r} has no frames under {dataset.images_path}")

    settings = {**spec.defaults, **(config or {})}

    pending = [
        path for path in frames if overwrite or not layout.score_path(path.stem).is_file()
    ]
    if not pending:
        report(f"{len(frames)} frames already scored; nothing to do (use --overwrite to redo)")
        return {"frames": len(frames), "scored": 0, "skipped": len(frames)}

    # Built only once there is work: constructing a Scorer loads weights.
    scorer = spec.build(settings)
    report(f"{spec.name}: {len(pending)} frame(s) to score, {len(frames) - len(pending)} cached")

    elapsed: list[float] = []
    for index, path in enumerate(pending, start=1):
        image = load_image(path)
        started = time.perf_counter()
        score = scorer.score(image)
        elapsed.append(time.perf_counter() - started)

        height, width = image.shape[:2]
        if score.shape != (height, width):
            raise ValueError(
                f"method {spec.name!r} returned a {score.shape} map for a "
                f"{(height, width)} frame; a Scorer must resize to the input frame"
            )

        save_score(layout.score_path(path.stem), score)
        save_soft_mask(layout.soft_mask_path(path.stem), spec.to_unit(score))

        # Opt-in, and the stage stays ignorant of what any of it means: a
        # method that wants its internals on disk leaves them in `last_debug`.
        debug = getattr(scorer, "last_debug", None)
        if save_intermediate and debug:
            save_debug(layout.intermediate / path.stem, debug)

        report(
            f"  [{index}/{len(pending)}] {path.name}  "
            f"max {float(score.max()):.3f}  {elapsed[-1] * 1000:.0f} ms"
        )

    # The first frame carries lazy CUDA/MPS init and is not representative.
    steady = elapsed[1:] or elapsed
    seconds_per_frame = sum(steady) / len(steady)
    report(f"{seconds_per_frame * 1000:.0f} ms/frame ({1 / seconds_per_frame:.2f} FPS)")

    summary = {
        "frames": len(frames),
        "scored": len(pending),
        "skipped": len(frames) - len(pending),
        "seconds_per_frame": round(seconds_per_frame, 4),
        "intermediate": save_intermediate and bool(getattr(scorer, "last_debug", None)),
        "settings": dict(settings),
    }
    update_manifest(layout, "infer", {"method": spec.name, "dataset": dataset.name, **summary})
    return summary
