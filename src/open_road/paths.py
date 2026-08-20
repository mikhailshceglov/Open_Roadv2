"""Where the repository is, so nothing else has to guess.

Every path in a config is either absolute or relative to the repository root.
The root is found by walking up from this file until a ``pyproject.toml``
appears, which keeps ``open-road`` working from any working directory --
including from inside ``runs/`` while looking at yesterday's results.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_ENV = "OPEN_ROAD_ROOT"


def repo_root() -> Path:
    """The checkout this package was installed from."""
    override = os.environ.get(ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    # Installed as a wheel with no repo around it. Nothing that needs the root
    # (methods, configs, runs) will work, but importing the package still does.
    return Path.cwd()


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Absolute paths pass through; relative ones hang off the repo root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (root or repo_root()) / candidate


def methods_dir() -> Path:
    return repo_root() / "methods"


def runs_dir() -> Path:
    return repo_root() / "runs"


def run_dir(method: str, dataset: str) -> Path:
    """``runs/<method>/<dataset>`` -- the one place a run's artefacts live."""
    return runs_dir() / method / dataset
