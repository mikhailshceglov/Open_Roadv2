"""Find the methods on this branch by looking at the filesystem.

One branch holds one method, and the branches have to merge into each other
without conflicts. A central ``REGISTRY = {"rba": ..., "raas_distill": ...}``
would be edited by every branch and would therefore conflict on every single
merge, so there isn't one.

Instead a method registers by *existing*: ``methods/<name>/method.py`` with a
module-level ``METHOD``. Adding a method touches no file another branch owns,
which is the property that keeps the merges clean.

Methods are imported one at a time and failures are captured rather than
raised, because a method's dependencies (a pinned transformers, a detectron2
build) will routinely be absent from the environment you are currently in.
``open-road methods`` should still tell you what exists and why it will not
load, rather than dying on the first bad import.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

from open_road.method import MethodSpec
from open_road.paths import methods_dir, repo_root

MODULE_NAME = "method"
SPEC_ATTRIBUTE = "METHOD"


@dataclass(frozen=True)
class Broken:
    """A method directory that exists but could not be loaded."""

    name: str
    reason: str


@dataclass(frozen=True)
class Discovery:
    working: dict[str, MethodSpec]
    broken: tuple[Broken, ...]

    @property
    def names(self) -> list[str]:
        return sorted(self.working)


def _candidates(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        entry
        for entry in root.iterdir()
        if entry.is_dir()
        and not entry.name.startswith((".", "_"))
        and (entry / f"{MODULE_NAME}.py").is_file()
    )


def _ensure_importable() -> None:
    """Put the repo root on ``sys.path`` so ``methods.<name>`` imports.

    Methods are real packages under a ``methods`` namespace and use relative
    imports internally (``from .student.model import Student``). That, rather
    than injecting each method directory onto ``sys.path``, is what stops two
    methods that both ship a ``student/`` package from shadowing each other.
    """
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_spec(name: str) -> MethodSpec:
    _ensure_importable()
    module = importlib.import_module(f"methods.{name}.{MODULE_NAME}")
    spec = getattr(module, SPEC_ATTRIBUTE, None)
    if spec is None:
        raise AttributeError(
            f"methods/{name}/{MODULE_NAME}.py defines no {SPEC_ATTRIBUTE}; "
            f"a method registers by exposing {SPEC_ATTRIBUTE} = MethodSpec(...)"
        )
    if not isinstance(spec, MethodSpec):
        raise TypeError(f"methods/{name}: {SPEC_ATTRIBUTE} is {type(spec).__name__}, not MethodSpec")
    if spec.name != name:
        raise ValueError(
            f"methods/{name}: {SPEC_ATTRIBUTE}.name is {spec.name!r}; it must match "
            f"the directory name, since that is what selects the method and names the run"
        )
    return spec


def discover(root: Path | None = None) -> Discovery:
    """Every method directory on this branch, loaded where possible."""
    working: dict[str, MethodSpec] = {}
    broken: list[Broken] = []
    for entry in _candidates(root or methods_dir()):
        try:
            working[entry.name] = _load_spec(entry.name)
        except Exception as error:  # noqa: BLE001 -- reported, not swallowed
            broken.append(Broken(entry.name, f"{type(error).__name__}: {error}"))
    return Discovery(working=working, broken=tuple(broken))


def load(name: str, root: Path | None = None) -> MethodSpec:
    """One method by name, raising with the real cause if it will not import."""
    directory = (root or methods_dir()) / name
    if not (directory / f"{MODULE_NAME}.py").is_file():
        found = discover(root)
        available = ", ".join(found.names) or "none on this branch"
        raise LookupError(f"unknown method {name!r}; available: {available}")
    return _load_spec(name)
