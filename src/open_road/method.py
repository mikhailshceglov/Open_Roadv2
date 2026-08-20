"""What a method must provide, and what the shared stages promise in return.

The methods in this repository disagree about nearly everything. One emits a
sigmoid in ``[0, 1]``; another an unbounded ``-sem_seg.tanh().sum(0)`` living
around ``[-19, 19]``. One is inference-only; another carries a four-stage
training pipeline. Their torch pins are not mutually satisfiable, which is why
each method declares its own requirements and none of them share an environment.

They agree on exactly one thing:

    one float per pixel, higher meaning more anomalous.

That primitive is the whole contract. Everything they disagree about is
*declared* on ``MethodSpec`` rather than hardcoded into the shared stages --
which is precisely what lets ``render`` and ``evaluate`` be written once instead
of once per method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Scorer(Protocol):
    """A loaded model, ready to score frames."""

    def score(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a float32 ``(H, W)`` map, higher meaning more anomalous.

        ``image_bgr`` is a uint8 ``(H, W, 3)`` array exactly as ``cv2.imread``
        returns it. The map must come back at the frame's own height and width:
        methods that run at a different internal resolution resize before
        returning, because only the method knows the right interpolation for
        its own output.
        """


@dataclass(frozen=True)
class MethodSpec:
    """Everything the shared stages need to know about one method.

    Declared once, at ``methods/<name>/method.py`` module level, as ``METHOD``.
    Constructing a spec must not load weights or import torch -- ``open-road
    methods`` builds every spec just to list them, and that has to stay cheap.
    """

    name: str
    """Directory name under ``methods/``. Also the first path segment of a run."""

    description: str
    """One line, shown by ``open-road methods``."""

    build: Callable[[Mapping[str, Any]], Scorer]
    """Config mapping in, loaded ``Scorer`` out. Called once per run, lazily."""

    score_range: tuple[float, float]
    """Expected output range, used only to render the 8-bit preview PNG.

    Metrics never see it: AP, AUROC and FPR95 depend on the ordering of pixels,
    not their scale. Getting it wrong makes ``soft_mask/`` look washed out and
    changes nothing else.
    """

    default_threshold: float
    """Where to cut the score when a run does not say.

    Load-bearing and method-specific: RbA's -0.0161 was swept on RoadAnomaly and
    means nothing to a method emitting probabilities.
    """

    default_min_area: int = 0
    """Connected components smaller than this are dropped during render."""

    defaults: Mapping[str, Any] = field(default_factory=dict)
    """Method config defaults, overridden by ``configs/methods/<name>.yaml``."""

    def __post_init__(self) -> None:
        low, high = self.score_range
        if not high > low:
            raise ValueError(
                f"method {self.name!r}: score_range must be increasing, got {self.score_range}"
            )

    def to_unit(self, score: np.ndarray) -> np.ndarray:
        """Map a score map onto ``[0, 1]`` for the 8-bit preview."""
        low, high = self.score_range
        return np.clip((score - low) / (high - low), 0.0, 1.0)
