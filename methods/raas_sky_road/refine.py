"""SAM-guided refinement of a coarse anomaly map, via Objectomaly.

RAAS produces a map whose *values* are reasonable and whose *boundaries* are not:
query masks are smooth blobs, so an obstacle's score bleeds across its edge.
Objectomaly fixes edges by borrowing them from SAM, which segments objects
without knowing or caring what they are:

    SAM ──► masks ──► postprocess ──► OASC ──► MBP ──► refined map
                                      │        │
                    per-region recalibration   boundary-band residual

Neither OASC nor MBP is implemented here. Their maths lives in the Objectomaly
repository, which this method calls into rather than copies — and that is not
laziness. The pinned Objectomaly commit ships **no LICENSE**, and the source
branch's own `docs/OBJECTOMALY_LICENSE_STATUS.md` is explicit: keep it an
unmodified external checkout, do not vendor it, do not redistribute a combined
archive. So it stays behind ``$OBJECTOMALY_ROOT``.

What this module owns is the wiring: config, the SAM generator, mask caching,
and the call order.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT_ENV = "OBJECTOMALY_ROOT"
SAM_CHECKPOINT_ENV = "SAM_CHECKPOINT"

PINNED_COMMIT = "66d2ad2a1b02d79389f4265d9d1d99ab6412324f"
"""The commit the source branch pinned and verified at runtime."""


def objectomaly_root() -> Path:
    """The Objectomaly checkout, or a message naming what is missing and why."""
    value = os.environ.get(ROOT_ENV)
    if not value:
        raise RuntimeError(
            f"${ROOT_ENV} is not set. The refinement stage needs Objectomaly "
            f"(https://github.com/hon121215/Objectomaly, pinned at {PINNED_COMMIT[:12]}). "
            f"It is not vendored here: the pinned commit has no LICENSE, so it must "
            f"stay an unmodified external checkout. Clone it and point ${ROOT_ENV} at it."
        )
    root = Path(value).expanduser().resolve()
    if not (root / "objectomaly").is_dir():
        raise RuntimeError(
            f"${ROOT_ENV} is {root}, but objectomaly/ is not there. Point it at the "
            f"repository root."
        )
    return root


def load_objectomaly() -> dict[str, Any]:
    """Import the four pieces this method needs, on first use."""
    root = str(objectomaly_root())
    if root not in sys.path:
        sys.path.insert(0, root)

    from objectomaly.masks.postprocess import postprocess
    from objectomaly.masks.sam import SAMMaskGenerator
    from objectomaly.refinement.mbp import apply_mbp
    from objectomaly.refinement.oasc import apply_oasc

    return {
        "SAMMaskGenerator": SAMMaskGenerator,
        "postprocess": postprocess,
        "apply_oasc": apply_oasc,
        "apply_mbp": apply_mbp,
    }


def checkpoint_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_sam_checkpoint(config: Mapping[str, Any]) -> Path:
    """The SAM weights, verified against the expected digest if one is configured.

    The digest check is worth keeping: SAM ships several checkpoints with
    similar names, and loading ViT-B where ViT-H was measured degrades the
    result silently rather than failing.
    """
    value = config.get("checkpoint") or os.environ.get(SAM_CHECKPOINT_ENV)
    if not value:
        raise RuntimeError(
            f"No SAM checkpoint. Set ${SAM_CHECKPOINT_ENV} or sam.checkpoint in the "
            f"method config. ViT-H is what the published numbers used: "
            f"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {path}")

    expected = str(config.get("checkpoint_md5", "")).lower()
    if expected:
        actual = checkpoint_md5(path)
        if actual != expected:
            raise RuntimeError(
                f"SAM checkpoint MD5 mismatch for {path}: expected {expected}, got {actual}. "
                f"A different SAM variant will degrade the result silently."
            )
    return path


class Refiner:
    """SAM masks in, boundary-corrected anomaly map out.

    The mask bundle is cached per frame because it depends only on the image:
    three RAAS variants over the same dataset share one SAM pass, which is the
    difference between running SAM once and three times at ~41 s a frame.
    """

    def __init__(self, config: Mapping[str, Any], device: str = "cpu") -> None:
        self.config = dict(config)
        self.device = device
        self._modules: dict[str, Any] | None = None
        self._generator = None
        self._bundles: dict[str, Any] = {}

    @property
    def modules(self) -> dict[str, Any]:
        if self._modules is None:
            self._modules = load_objectomaly()
        return self._modules

    @property
    def generator(self):
        if self._generator is None:
            sam = dict(self.config.get("sam", {}))
            self._generator = self.modules["SAMMaskGenerator"](
                checkpoint=str(resolve_sam_checkpoint(sam)),
                variant=sam.get("variant", "vit_h"),
                device=self.device,
                points_per_side=int(sam.get("points_per_side", 32)),
                pred_iou_thresh=float(sam.get("pred_iou_thresh", 0.88)),
                stability_score_thresh=float(sam.get("stability_score_thresh", 0.95)),
            )
        return self._generator

    def masks(self, image_bgr: np.ndarray, cache_key: str | None = None):
        """The postprocessed SAM bundle for one frame."""
        if cache_key is not None and cache_key in self._bundles:
            return self._bundles[cache_key]
        raw = self.generator.generate(image_bgr)
        bundle = self.modules["postprocess"](raw, **dict(self.config.get("postprocess", {})))
        if cache_key is not None:
            self._bundles[cache_key] = bundle
        return bundle

    def calibrate(self, coarse: np.ndarray, bundle) -> np.ndarray:
        """OASC: recalibrate the map region by region against SAM's segmentation."""
        oasc = dict(self.config.get("oasc", {}))
        return self.modules["apply_oasc"](
            coarse, bundle, oasc.get("variant", "quality_aware_residual_blending"),
            **dict(oasc.get("params", {})),
        )

    def sharpen(self, calibrated: np.ndarray, image_bgr: np.ndarray, bundle) -> np.ndarray:
        """MBP: rewrite the residual in a band around each SAM boundary."""
        mbp = dict(self.config.get("mbp", {}))
        params = dict(mbp.get("params", {}))
        if "gaussian_ksize" in params:
            # JSON and YAML both give a list; OpenCV insists on a tuple.
            params["gaussian_ksize"] = tuple(params["gaussian_ksize"])
        return self.modules["apply_mbp"](
            calibrated, mbp.get("variant", "boundary_band_residual"),
            image_bgr=image_bgr, bundle=bundle, **params,
        )

    def __call__(self, coarse: np.ndarray, image_bgr: np.ndarray,
                 cache_key: str | None = None) -> tuple[np.ndarray, Any]:
        """The full stage. Returns the refined map and the bundle behind it."""
        bundle = self.masks(image_bgr, cache_key)
        calibrated = self.calibrate(coarse, bundle)
        refined = self.sharpen(calibrated, image_bgr, bundle)
        return np.clip(np.asarray(refined, dtype=np.float32), 0.0, 1.0), bundle
