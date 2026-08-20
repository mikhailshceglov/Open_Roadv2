"""Report reads metrics.json and nothing else, so it can compare anything."""

from __future__ import annotations

from pathlib import Path

import pytest

from open_road.io import RunLayout, write_json
from open_road.stages.report import run_report, to_markdown


def _run(tmp_path: Path, method: str, ap: float) -> Path:
    layout = RunLayout(tmp_path / "runs" / method / "toy")
    write_json(
        layout.metrics,
        {
            "method": method,
            "dataset": "toy",
            "frames": 3,
            "pixel": {"AP": ap, "AUROC": 0.9, "FPR95": 0.1,
                      "precision": 0.8, "recall": 0.7, "F1": 0.75},
            "component": {"sIoU_gt": 0.5, "PPV": 0.6, "F1_star": 0.55},
        },
    )
    return layout.root


def test_two_runs_land_in_one_table(tmp_path: Path) -> None:
    runs = [_run(tmp_path, "alpha", 0.90), _run(tmp_path, "beta", 0.70)]

    result = run_report(runs, tmp_path / "out", report=lambda _: None)

    assert result["runs"] == ["alpha / toy", "beta / toy"]
    table = (tmp_path / "out" / "comparison.md").read_text(encoding="utf-8")
    assert "90.00" in table and "70.00" in table
    assert (tmp_path / "out" / "comparison.csv").is_file()
    assert (tmp_path / "out" / "comparison.json").is_file()


def test_a_missing_metric_renders_as_a_dash() -> None:
    rows = [("partial / toy", {"frames": 1, "pixel": {"AP": 0.5}})]

    assert "—" in to_markdown(rows)


def test_a_run_without_metrics_says_to_evaluate_first(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="open-road eval"):
        run_report([tmp_path / "nothing"], report=lambda _: None)
