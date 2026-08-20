"""The raas_distill spec, checked without loading the model.

Added by the raas-distill branch. It deliberately never constructs a Scorer:
the point is that discovery, and therefore ``open-road methods``, works in an
environment that has none of this method's dependencies installed.
"""

from __future__ import annotations

import pytest

from open_road import registry
from open_road.method import MethodSpec

NAME = "raas_distill"


@pytest.fixture
def spec() -> MethodSpec:
    found = registry.discover()
    if NAME not in found.working:
        broken = {entry.name: entry.reason for entry in found.broken}
        pytest.fail(f"{NAME} did not load: {broken.get(NAME, 'no such method directory')}")
    return found.working[NAME]


def test_the_spec_loads_without_torch(spec: MethodSpec) -> None:
    # Importing method.py must not pull in torch or transformers -- the heavy
    # imports live inside AnomalySegmenter for exactly this reason.
    assert spec.name == NAME
    assert spec.description


def test_it_declares_a_sigmoid_output(spec: MethodSpec) -> None:
    # The student's head ends in a sigmoid, so the preview needs no rescaling
    # and the declared range must say so.
    assert spec.score_range == (0.0, 1.0)
    assert 0.0 < spec.default_threshold < 1.0


def test_it_drops_small_components_by_default(spec: MethodSpec) -> None:
    # On this benchmark the minimum-area filter moves component F1 by tens of
    # points, so leaving it at zero would be a silent quality regression.
    assert spec.default_min_area > 0


def test_the_shipped_checkpoint_is_where_the_defaults_say(spec: MethodSpec) -> None:
    from methods.raas_distill.method import DEFAULT_CHECKPOINT

    assert DEFAULT_CHECKPOINT.is_file(), (
        f"{DEFAULT_CHECKPOINT} is missing. It is committed to this branch on "
        f"purpose -- check .gitignore is not excluding methods/*/weights/*.pt."
    )


def test_defaults_name_the_resolution_knob(spec: MethodSpec) -> None:
    assert spec.defaults["short_side"] >= 544, "below 544 small obstacles disappear"
