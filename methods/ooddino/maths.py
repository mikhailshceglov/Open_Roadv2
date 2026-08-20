"""The pixel branch's arithmetic, and the fusion that follows it.

Pure numpy, no models: everything here is testable on a hand-made logit tensor,
which matters because these formulas are a reconstruction of a paper whose
weights were never released. The official OoDDINO repository is empty, so the
OUAFS module and the ADT-Net heads are reimplemented from the equations rather
than loaded — this is a zero-shot reconstruction on off-the-shelf checkpoints,
not the trained method, and it should not be read as reproducing its numbers.

Every function follows the same sign convention as the rest of the repository:
**higher means more anomalous**.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def softmax(logits: np.ndarray, eps: float = EPS) -> np.ndarray:
    """``(C, H, W)`` logits to probabilities, stabilised by the per-pixel max."""
    scores = np.asarray(logits, dtype=np.float64)
    peak = scores.max(axis=0, keepdims=True)
    exp = np.exp(scores - peak)
    return exp / np.clip(exp.sum(axis=0, keepdims=True), eps, None)


def entropy_map(logits: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Softmax entropy, divided by ``log C`` so it lands in ``[0, 1]``."""
    probs = np.clip(softmax(logits, eps=eps), eps, 1.0)
    n_classes = max(2, int(probs.shape[0]))
    raw = -np.sum(probs * np.log(probs), axis=0)
    return (raw / np.log(n_classes)).astype(np.float32)


def distance_map(logits: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Distance from the predicted one-hot vertex: ``1 - p_max``."""
    return (1.0 - softmax(logits, eps=eps).max(axis=0)).astype(np.float32)


def energy_map(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Liu et al. free energy, ``E = -T logsumexp(logits / T)``. Higher is more OOD."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scores = np.asarray(logits, dtype=np.float64) / temperature
    peak = scores.max(axis=0, keepdims=True)
    logsumexp = peak[0] + np.log(np.exp(scores - peak).sum(axis=0))
    return (-temperature * logsumexp).astype(np.float32)


def class_conditional_residual(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """``I = E - median(E | class)``, one offset per predicted class.

    Absolute energy is not comparable across classes: sky is uncertain
    everywhere and would dominate a global ranking. Subtracting each class's own
    median asks a better question -- is this pixel unusual *for what the network
    thinks it is* -- so typical sky lands near zero while a dusty obstacle the
    network calls road keeps a large residual.
    """
    data = np.asarray(values, dtype=np.float64)
    predicted = np.asarray(labels)
    if data.shape != predicted.shape:
        raise ValueError("values and labels must have the same shape")
    residual = np.empty_like(data)
    for class_id in np.unique(predicted):
        mask = predicted == class_id
        residual[mask] = data[mask] - float(np.median(data[mask]))
    return residual.astype(np.float32)


def pixel_score(logits: np.ndarray, kind: str = "class_residual") -> np.ndarray:
    """The pixel branch's anomaly score. ``class_residual`` is what the config uses."""
    scores = np.asarray(logits, dtype=np.float64)
    if kind == "entropy":
        return entropy_map(scores)
    if kind == "energy":
        return energy_map(scores)
    if kind == "class_residual":
        return class_conditional_residual(energy_map(scores), np.argmax(scores, axis=0))
    raise ValueError(f"unknown pixel score: {kind!r}")


def minmax01(values: np.ndarray) -> np.ndarray:
    """Min-max to ``[0, 1]``; all zeros when the map is flat."""
    data = np.asarray(values, dtype=np.float64)
    low = float(data.min()) if data.size else 0.0
    high = float(data.max()) if data.size else 1.0
    if high - low < EPS:
        return np.zeros(data.shape, dtype=np.float32)
    return ((data - low) / (high - low)).astype(np.float32)


def orthogonalize(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Remove the part of ``secondary`` lying along ``primary``.

    A single scalar projection over the whole flattened image, which is what
    makes this cheap and also what limits it: entropy and 1 - p_max are close to
    monotone functions of each other, so the residual is mostly noise. Measured
    on RoadAnomaly the resulting prior separates anomaly from background worse
    than a coin (AUROC 0.42), which is the known weak point of this branch.
    """
    base = np.asarray(primary, dtype=np.float64).reshape(-1)
    other = np.asarray(secondary, dtype=np.float64)
    flat = other.reshape(-1)
    denominator = float(np.dot(base, base)) + EPS
    projected = (float(np.dot(flat, base)) / denominator) * base
    return (flat - projected).reshape(other.shape).astype(np.float32)


def ouafs_prior(entropy: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Sequential uncertainty fusion (Algorithm 1), without the trained module.

    The paper concatenates encoder features before fusing; there are none to
    concatenate here, so this keeps the two steps that do not need them: gate by
    a sigmoid centred on the image's median entropy, then modulate by the
    orthogonalised distance map in place of the cross-attention.
    """
    entropy = np.asarray(entropy, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    centre = float(np.median(entropy)) if entropy.size else 0.0
    gated = 1.0 / (1.0 + np.exp(-(entropy - centre)))
    return minmax01(gated * minmax01(orthogonalize(entropy, distance)))


def region_mean(values: np.ndarray, mask: np.ndarray) -> float:
    """Mean inside ``mask``, falling back to the global mean when it is empty."""
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if selected.size == 0:
        return float(np.mean(values)) if np.asarray(values).size else 0.0
    return float(selected.mean())


def arns_normalize(score: np.ndarray, foreground: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Adaptive Region Normalization (Eq. 2). Output spans ``[alpha, alpha + 0.5]``.

    Each region is centred on its own mean before the sigmoid, which is the
    point -- a proposal's interior is judged against other proposal pixels, not
    against the sky. The cost is a discontinuity at every box edge, which is why
    the normalised map is a decision surface and not a calibrated score.
    """
    values = np.asarray(score, dtype=np.float64)
    fg = np.asarray(foreground, dtype=bool)
    mu = np.where(fg, region_mean(values, fg), region_mean(values, ~fg))
    return (0.5 / (1.0 + np.exp(-(values - mu))) + float(alpha)).astype(np.float32)


def dual_thresholds(
    score_norm: np.ndarray,
    foreground: np.ndarray,
    fg_quantile: float = 0.4,
    bg_quantile: float = 0.9,
    min_gap: float = 0.15,
    alpha: float = 0.3,
) -> tuple[float, float]:
    """Parameter-free stand-in for the trained ADT-Net heads.

    ARNS maps each region's mean onto ``alpha + 0.25``, so two quantiles taken
    either side of it can collapse onto each other and erase the distinction the
    dual threshold exists to make. The gap is forced open around that midpoint.
    """
    values = np.asarray(score_norm, dtype=np.float64)
    fg = np.asarray(foreground, dtype=bool)
    inside, outside = values[fg], values[~fg]

    t_fg = float(np.quantile(inside if inside.size else values, fg_quantile))
    t_bg = float(np.quantile(outside if outside.size else values, bg_quantile))

    midpoint = float(alpha) + 0.25
    gap = max(float(min_gap), 0.0)
    if t_bg - t_fg < gap:
        t_fg = min(t_fg, midpoint - gap / 2.0)
        t_bg = max(t_bg, midpoint + gap / 2.0)
    return t_fg, t_bg


def adt_probability(
    score_norm: np.ndarray,
    foreground: np.ndarray,
    t_fg: float,
    t_bg: float,
    delta: float = 0.1,
) -> np.ndarray:
    """Soft dual threshold (Eq. 5) as a ramp of width ``delta`` centred on T.

    ``delta`` softens the reported probability only; the binary decision
    ``P >= 0.5`` is exactly ``score_norm >= T`` whatever it is set to.
    """
    values = np.asarray(score_norm, dtype=np.float64)
    threshold = np.where(np.asarray(foreground, dtype=bool), float(t_fg), float(t_bg))
    return np.clip((values - threshold) / max(float(delta), 1e-6) + 0.5, 0.0, 1.0).astype(np.float32)
