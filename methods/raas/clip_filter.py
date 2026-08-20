"""CLIP as the arbiter over road-hole candidates.

The road-hole geometry finds *something* on the road but has no idea what. CLIP
is asked to name each candidate against the Cityscapes vocabulary; a confident
name means it is a known road user and the region is suppressed, anything else
is left as anomaly.

Two decision rules ship, and they are the only difference between the
``maskomaly_id`` and ``maskomaly_ood`` variants:

``id``   softmax over the 19 known prompts alone. Confident name → known.
``ood``  softmax over 19 known + 3 open-ended "something unusual" prompts, with
         a three-way rule that also demands the known name beat the OOD one by
         a margin.

Both write a *hard constant* into the region — 0.05 for known, 1.0 for anomaly —
rather than modulating the underlying score. That is a real property of the
method, not a simplification here: whatever the soft mask believed about those
pixels is discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

ID_PROMPTS: tuple[str, ...] = (
    "a photo of road",
    "a photo of sidewalk",
    "a photo of building",
    "a photo of wall",
    "a photo of fence",
    "a photo of pole",
    "a photo of traffic light",
    "a photo of traffic sign",
    "a photo of vegetation",
    "a photo of terrain",
    "a photo of sky",
    "a photo of person on the road",
    "a photo of rider on the road",
    "a photo of car on the road",
    "a photo of truck on the road",
    "a photo of bus on the road",
    "a photo of train on the track",
    "a photo of motorcycle on the road",
    "a photo of bicycle on the road",
)

OOD_PROMPTS: tuple[str, ...] = (
    "something unusual in a driving scene",
    "an unexpected object in the road environment",
    "a strange or unknown item on the street",
)

KNOWN_SCORE = 0.05
"""Written into a region CLIP recognised."""

ANOMALY_SCORE = 1.0
"""Written into a region CLIP did not recognise."""

ID_CONFIDENCE = 0.85
"""How sure CLIP must be about a known name to suppress the region."""

OOD_FLOOR = 0.001
"""Below this OOD mass, the region is forced to known regardless of anything else."""

OOD_MARGIN = 0.1
"""How far the known name must beat the OOD one, in the ``ood`` rule."""


@dataclass(frozen=True)
class Decision:
    """One candidate's verdict, kept for the intermediate dump."""

    anomalous: bool
    score: float
    label: str
    id_probability: float
    ood_probability: float | None = None


def decide_id(probabilities: np.ndarray, prompts: Sequence[str] = ID_PROMPTS) -> Decision:
    """Known if the best of the 19 class prompts clears ``ID_CONFIDENCE``."""
    best = int(np.argmax(probabilities))
    confidence = float(probabilities[best])
    known = confidence > ID_CONFIDENCE
    return Decision(
        anomalous=not known,
        score=KNOWN_SCORE if known else ANOMALY_SCORE,
        label=prompts[best] if known else "unknown",
        id_probability=confidence,
    )


def decide_ood(
    probabilities: np.ndarray,
    id_prompts: Sequence[str] = ID_PROMPTS,
    ood_prompts: Sequence[str] = OOD_PROMPTS,
) -> Decision:
    """Three-way rule over a softmax spanning both prompt sets.

    Note the first branch: when the OOD prompts attract almost no mass the
    region is called known *whatever* the ID confidence is — so a region CLIP
    finds equally unconvincing across all 22 prompts is suppressed rather than
    reported. That asymmetry is in the original and is preserved.
    """
    id_probabilities = probabilities[: len(id_prompts)]
    ood_probabilities = probabilities[len(id_prompts) :]
    best_id = int(np.argmax(id_probabilities))
    max_id = float(id_probabilities[best_id])
    max_ood = float(np.max(ood_probabilities))

    if max_ood < OOD_FLOOR:
        known = True
    elif max_id > max_ood and max_id > ID_CONFIDENCE and (max_id - max_ood) > OOD_MARGIN:
        known = True
    else:
        known = False

    return Decision(
        anomalous=not known,
        score=KNOWN_SCORE if known else ANOMALY_SCORE,
        label=id_prompts[best_id] if known else "unknown",
        id_probability=max_id,
        ood_probability=max_ood,
    )


def apply_decision(score: np.ndarray, component: np.ndarray, box: tuple[int, int, int, int],
                   decision: Decision) -> None:
    """Write the verdict into the region, in place.

    Restricted to the component's bounding box exactly as the original does,
    which matters when two candidates' boxes overlap: the later one wins inside
    its own component only.
    """
    x, y, width, height = box
    window = score[y : y + height, x : x + width]
    window[component[y : y + height, x : x + width] > 0] = decision.score
