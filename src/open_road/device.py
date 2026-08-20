"""Device selection, kept in one place so no method hardcodes ``cuda``.

detectron2-based methods default to ``cuda`` internally and become unimportable
on a laptop; every method in this repository takes its device from here instead.
"""

from __future__ import annotations


def resolve_device(name: str = "auto") -> str:
    """``auto`` picks the best available backend; anything else passes through."""
    if name != "auto":
        return name

    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
