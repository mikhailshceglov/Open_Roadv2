"""Put several runs' metrics side by side.

Reads only ``metrics.json``, so it compares runs produced weeks apart, by
different methods, in different environments -- nothing here imports a method
or touches a score map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from open_road.io import RunLayout, read_json, write_json

Reporter = Callable[[str], None]

COLUMNS: tuple[tuple[str, str, str], ...] = (
    # (header, section, key)
    ("AP", "pixel", "AP"),
    ("AUROC", "pixel", "AUROC"),
    ("FPR95", "pixel", "FPR95"),
    ("Precision", "pixel", "precision"),
    ("Recall", "pixel", "recall"),
    ("F1", "pixel", "F1"),
    ("sIoU_gt", "component", "sIoU_gt"),
    ("PPV", "component", "PPV"),
    ("F1*", "component", "F1_star"),
)


def _label(path: Path, metrics: dict[str, Any]) -> str:
    method = metrics.get("method")
    dataset = metrics.get("dataset")
    if method and dataset:
        return f"{method} / {dataset}"
    return path.parent.name or str(path)


def collect(runs: Iterable[str | Path]) -> list[tuple[str, dict[str, Any]]]:
    """Load ``metrics.json`` for each run directory, in the order given."""
    rows: list[tuple[str, dict[str, Any]]] = []
    for run in runs:
        path = Path(run)
        metrics_path = path if path.suffix == ".json" else RunLayout(path).metrics
        if not metrics_path.is_file():
            raise SystemExit(f"no metrics.json under {path}; run `open-road eval` there first")
        metrics = read_json(metrics_path)
        rows.append((_label(metrics_path, metrics), metrics))
    return rows


def to_markdown(rows: Sequence[tuple[str, dict[str, Any]]]) -> str:
    headers = ["run", "frames", *(header for header, _, _ in COLUMNS)]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for label, metrics in rows:
        cells = [label, str(metrics.get("frames", ""))]
        for _header, section, key in COLUMNS:
            value = metrics.get(section, {}).get(key)
            cells.append(f"{100 * value:.2f}" if isinstance(value, (int, float)) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_csv(rows: Sequence[tuple[str, dict[str, Any]]]) -> str:
    headers = ["run", "frames", *(header for header, _, _ in COLUMNS)]
    lines = [",".join(headers)]
    for label, metrics in rows:
        cells = [label, str(metrics.get("frames", ""))]
        for _header, section, key in COLUMNS:
            value = metrics.get(section, {}).get(key)
            cells.append(f"{100 * value:.4f}" if isinstance(value, (int, float)) else "")
        lines.append(",".join(cells))
    return "\n".join(lines)


def run_report(
    runs: Sequence[str | Path],
    out: Path | None = None,
    *,
    report: Reporter = print,
) -> dict[str, Any]:
    """Compare runs; print the table and, with ``out``, write csv + markdown."""
    rows = collect(runs)
    table = to_markdown(rows)
    report(table)

    written: dict[str, str] = {}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "comparison.md").write_text(table + "\n", encoding="utf-8")
        (out / "comparison.csv").write_text(to_csv(rows) + "\n", encoding="utf-8")
        write_json(out / "comparison.json", [metrics for _label, metrics in rows])
        written = {
            "markdown": str(out / "comparison.md"),
            "csv": str(out / "comparison.csv"),
            "json": str(out / "comparison.json"),
        }
        report(f"\nwritten to {out}")

    return {"runs": [label for label, _ in rows], "written": written}
