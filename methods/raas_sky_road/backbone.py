"""Mask2Former behind an adapter, because it is not part of this repository.

RAAS wraps Mask2Former with a *patched* detectron2 predictor: the stock
``DefaultPredictor`` returns one value, the patched one returns three, and the
extra two — the raw query classifications and query masks — are the entire input
to the soft-mask formula. Nothing works without that patch.

The patch, Mask2Former, detectron2 and the Swin-L checkpoint all live in the
RAAS monorepo, which is a separate checkout. Point ``$RAAS_ROOT`` at it. The
original derived this path as ``Path(__file__).parent.parent`` and so could only
ever run from inside that tree; naming it explicitly is what makes the method
portable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT_ENV = "RAAS_ROOT"
CHECKPOINT_ENV = "RAAS_WEIGHTS"

DEFAULT_CHECKPOINT = Path("Maskomaly/maskomaly/ckpt/model_final_17c1ee.pkl")
DEFAULT_CONFIG = Path(
    "Mask2Former/configs/cityscapes/semantic-segmentation/swin/"
    "maskformer2_swin_large_IN21k_384_bs16_90k.yaml"
)


def raas_root() -> Path:
    """The RAAS checkout, or a message naming exactly what is missing."""
    value = os.environ.get(ROOT_ENV)
    if not value:
        raise RuntimeError(
            f"${ROOT_ENV} is not set. This method needs the RAAS monorepo "
            f"(detectron2 + Mask2Former + Maskomaly's patched predictor + the Swin-L "
            f"checkpoint); it is not vendored here. Clone it and point ${ROOT_ENV} at it."
        )
    root = Path(value).expanduser().resolve()
    for required in ("Mask2Former", "detectron2", "Maskomaly"):
        if not (root / required).is_dir():
            raise RuntimeError(
                f"${ROOT_ENV} is {root}, but {required}/ is not there. Point it at the "
                f"root of the checkout, not at a subdirectory."
            )
    return root


def configure_import_paths(root: Path | None = None) -> Path:
    """Put the RAAS trees on ``sys.path``, replacements first.

    The ordering is load-bearing, not cosmetic: ``detectron2_replacements`` must
    shadow ``detectron2`` or you get the stock predictor back and the model
    returns one value where three are expected — surfacing far away as
    ``ValueError: not enough values to unpack (expected 3, got 1)``.
    """
    root = root or raas_root()
    ordered = [
        root / "Maskomaly" / "detectron2_replacements",
        root / "Maskomaly" / "maskomaly",
        root / "detectron2",
        root / "Mask2Former",
    ]
    for path in reversed(ordered):
        entry = str(path)
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    return root


def checkpoint_path(root: Path, override: str | None = None) -> Path:
    value = override or os.environ.get(CHECKPOINT_ENV)
    path = Path(value).expanduser() if value else root / DEFAULT_CHECKPOINT
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(
            f"Swin-L checkpoint not found at {path}. It is the upstream Mask2Former "
            f"Cityscapes model (maskformer2_swin_large_IN21k_384_bs16_90k); no branch "
            f"of this project records a download URL for it."
        )
    return path


class QueryPredictor:
    """The patched predictor, exposing what the soft-mask formula needs.

    Returns ``(mask_cls, mask_pred, semantic)`` where ``mask_cls`` is a softmax
    over classes per query and ``mask_pred`` a per-pixel sigmoid per query.
    ``semantic`` is the Cityscapes argmax from the same forward pass, kept
    because the fusion variants need it and re-running the backbone to get it
    would double the cost.
    """

    def __init__(self, config_file: str | None = None, checkpoint: str | None = None,
                 device: str = "cpu") -> None:
        root = configure_import_paths()
        self.root = root
        self.device = device

        from detectron2.config import get_cfg
        from detectron2.engine.defaults import DefaultPredictor
        from detectron2.projects.deeplab import add_deeplab_config
        from mask2former import add_maskformer2_config

        config_path = Path(config_file) if config_file else root / DEFAULT_CONFIG
        if not config_path.is_absolute():
            config_path = root / config_path

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(str(config_path))
        # detectron2 defaults to cuda and becomes unimportable without a GPU.
        cfg.merge_from_list(
            ["MODEL.WEIGHTS", str(checkpoint_path(root, checkpoint)), "MODEL.DEVICE", device]
        )
        cfg.freeze()
        self._predictor = DefaultPredictor(cfg)

    def __call__(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch
        from torch.nn import functional as F

        result: Any = self._predictor(image_bgr)
        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError(
                "The predictor returned one value where three were expected. "
                "detectron2_replacements/ is not shadowing detectron2/ on sys.path."
            )
        segmentation, mask_cls, mask_pred = result

        with torch.no_grad():
            classes = F.softmax(mask_cls, dim=1).cpu().numpy()
            masks = mask_pred.sigmoid().cpu().numpy()
            semantic = segmentation["sem_seg"].argmax(0).cpu().numpy().astype(np.int16)
        return classes, masks, semantic


def build_args(config_file: str, checkpoint: str, masks: int = 4,
               analysis_file: str | None = None) -> SimpleNamespace:
    """The argparse-shaped object the original model classes expect.

    Kept for anyone wanting to drive the upstream classes directly rather than
    this package's reimplementation.
    """
    return SimpleNamespace(
        config_file=config_file,
        opts=["MODEL.WEIGHTS", checkpoint],
        masks=masks,
        analysis_file=analysis_file,
    )
