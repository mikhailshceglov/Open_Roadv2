"""Discovery is what keeps the branches merge-free; these are its guarantees."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from open_road import registry
from open_road.method import MethodSpec

SPEC_SOURCE = textwrap.dedent(
    '''
    from open_road.method import MethodSpec


    class _Scorer:
        def score(self, image_bgr):
            raise NotImplementedError


    METHOD = MethodSpec(
        name="{name}",
        description="a method that exists",
        build=lambda config: _Scorer(),
        score_range=(0.0, 1.0),
        default_threshold=0.5,
    )
    '''
)


@pytest.fixture
def methods_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway repo root with an importable ``methods`` package."""
    root = tmp_path / "repo"
    (root / "methods").mkdir(parents=True)
    (root / "methods" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text("", encoding="utf-8")
    monkeypatch.setenv("OPEN_ROAD_ROOT", str(root))
    monkeypatch.syspath_prepend(str(root))
    for name in [key for key in sys.modules if key == "methods" or key.startswith("methods.")]:
        del sys.modules[name]
    return root / "methods"


def _write_method(methods_root: Path, name: str, source: str | None = None) -> None:
    directory = methods_root / name
    directory.mkdir()
    (directory / "__init__.py").write_text("", encoding="utf-8")
    (directory / "method.py").write_text(
        source if source is not None else SPEC_SOURCE.format(name=name), encoding="utf-8"
    )


def test_a_method_registers_by_existing(methods_root: Path) -> None:
    _write_method(methods_root, "toy")

    found = registry.discover(methods_root)

    assert found.names == ["toy"]
    assert isinstance(found.working["toy"], MethodSpec)
    assert found.broken == ()


def test_main_has_no_methods_and_that_is_not_an_error(methods_root: Path) -> None:
    found = registry.discover(methods_root)

    assert found.names == []
    assert found.broken == ()


def test_an_unimportable_method_is_reported_not_raised(methods_root: Path) -> None:
    # A method's own pins are routinely absent from the environment you happen
    # to be in. Listing methods must still work and say why.
    _write_method(methods_root, "good")
    _write_method(methods_root, "bad", "import a_package_that_is_not_installed\n")

    found = registry.discover(methods_root)

    assert found.names == ["good"]
    assert [entry.name for entry in found.broken] == ["bad"]
    assert "a_package_that_is_not_installed" in found.broken[0].reason


def test_a_method_without_METHOD_is_broken(methods_root: Path) -> None:
    _write_method(methods_root, "empty", "X = 1\n")

    found = registry.discover(methods_root)

    assert [entry.name for entry in found.broken] == ["empty"]
    assert "METHOD" in found.broken[0].reason


def test_spec_name_must_match_its_directory(methods_root: Path) -> None:
    # The directory name selects the method and names the run directory; a spec
    # disagreeing with it would write results under a name nothing looks up.
    _write_method(methods_root, "here", SPEC_SOURCE.format(name="elsewhere"))

    found = registry.discover(methods_root)

    assert [entry.name for entry in found.broken] == ["here"]
    assert "must match" in found.broken[0].reason


def test_load_of_an_unknown_method_lists_what_exists(methods_root: Path) -> None:
    _write_method(methods_root, "toy")

    with pytest.raises(LookupError, match="available: toy"):
        registry.load("nope", methods_root)


def test_score_range_must_increase() -> None:
    with pytest.raises(ValueError, match="score_range"):
        MethodSpec(
            name="x",
            description="",
            build=lambda config: None,
            score_range=(1.0, 0.0),
            default_threshold=0.0,
        )
