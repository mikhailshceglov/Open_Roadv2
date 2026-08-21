"""The Objectomaly bridge: what it does before Objectomaly is even present.

Objectomaly is an external checkout with no LICENSE, so it is never vendored and
is absent here. These tests cover the wiring that runs on this side of the
boundary — the parts that decide whether the call is even attempted, and that
must fail with a usable message rather than an ImportError when it is not.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from methods.raas_sky_road import refine


def test_a_missing_root_explains_why_it_is_not_vendored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(refine.ROOT_ENV, raising=False)

    with pytest.raises(RuntimeError, match="no LICENSE"):
        refine.objectomaly_root()


def test_a_root_pointing_at_the_wrong_directory_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(refine.ROOT_ENV, str(tmp_path))

    with pytest.raises(RuntimeError, match="objectomaly/ is not there"):
        refine.objectomaly_root()


def test_a_valid_root_resolves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "objectomaly").mkdir()
    monkeypatch.setenv(refine.ROOT_ENV, str(tmp_path))

    assert refine.objectomaly_root() == tmp_path.resolve()


def test_the_sam_digest_is_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # SAM ships several checkpoints with similar names. Loading ViT-B where
    # ViT-H was measured degrades the result silently, so a wrong digest must
    # stop the run rather than be a warning.
    monkeypatch.delenv(refine.SAM_CHECKPOINT_ENV, raising=False)
    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"not really a checkpoint")
    actual = hashlib.md5(checkpoint.read_bytes()).hexdigest()

    resolved = refine.resolve_sam_checkpoint(
        {"checkpoint": str(checkpoint), "checkpoint_md5": actual}
    )
    assert resolved == checkpoint.resolve()

    with pytest.raises(RuntimeError, match="MD5 mismatch"):
        refine.resolve_sam_checkpoint(
            {"checkpoint": str(checkpoint), "checkpoint_md5": "0" * 32}
        )


def test_an_empty_digest_skips_the_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(refine.SAM_CHECKPOINT_ENV, raising=False)
    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"x")

    assert refine.resolve_sam_checkpoint({"checkpoint": str(checkpoint)}) == checkpoint.resolve()


def test_a_missing_checkpoint_names_the_expected_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(refine.SAM_CHECKPOINT_ENV, raising=False)

    with pytest.raises(RuntimeError, match="ViT-H"):
        refine.resolve_sam_checkpoint({})


def test_the_environment_variable_is_a_fallback_for_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"x")
    monkeypatch.setenv(refine.SAM_CHECKPOINT_ENV, str(checkpoint))

    assert refine.resolve_sam_checkpoint({}) == checkpoint.resolve()


class _Bundle:
    n = 3


class _Modules(dict):
    """Stand-in for the Objectomaly imports, recording the call order."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        super().__init__(
            {
                "postprocess": self._postprocess,
                "apply_oasc": self._oasc,
                "apply_mbp": self._mbp,
            }
        )

    def _postprocess(self, raw, **kwargs):
        self.calls.append("postprocess")
        return _Bundle()

    def _oasc(self, coarse, bundle, variant, **params):
        self.calls.append(f"oasc:{variant}")
        self.params = params
        return coarse * 0.5

    def _mbp(self, calibrated, variant, *, image_bgr, bundle, **params):
        self.calls.append(f"mbp:{variant}")
        self.ksize = params.get("gaussian_ksize")
        # Deliberately out of range, to check the caller clips.
        return calibrated + 0.9


def _refiner(config=None) -> refine.Refiner:
    refiner = refine.Refiner(config or {})
    refiner._modules = _Modules()
    refiner._generator = type("G", (), {"generate": staticmethod(lambda image: None)})()
    return refiner


def test_the_stage_order_is_sam_then_oasc_then_mbp() -> None:
    refiner = _refiner()

    refined, bundle = refiner(np.full((4, 4), 0.4, dtype=np.float32), np.zeros((4, 4, 3), np.uint8))

    assert refiner._modules.calls == [
        "postprocess",
        "oasc:quality_aware_residual_blending",
        "mbp:boundary_band_residual",
    ]
    assert bundle.n == 3
    # 0.4 * 0.5 + 0.9 = 1.1, clipped.
    assert refined.max() <= 1.0
    assert refined.dtype == np.float32


def test_the_gaussian_kernel_is_handed_over_as_a_tuple() -> None:
    # YAML and JSON both give a list; OpenCV insists on a tuple.
    refiner = _refiner({"mbp": {"params": {"gaussian_ksize": [7, 7]}}})

    refiner(np.zeros((4, 4), dtype=np.float32), np.zeros((4, 4, 3), np.uint8))

    assert refiner._modules.ksize == (7, 7)


def test_one_sam_pass_is_reused_across_frames_with_the_same_key() -> None:
    # SAM was ~41s a frame, two thirds of the pipeline, and depends only on the
    # image — so three RAAS variants over one dataset must share the pass.
    refiner = _refiner()
    image = np.zeros((4, 4, 3), np.uint8)

    refiner.masks(image, cache_key="frame0")
    refiner.masks(image, cache_key="frame0")
    refiner.masks(image, cache_key="frame1")

    assert refiner._modules.calls.count("postprocess") == 2
