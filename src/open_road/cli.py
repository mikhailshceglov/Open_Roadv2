"""The command line: the same five verbs for every method.

    open-road methods
    open-road datasets
    open-road infer  --method M --dataset D
    open-road render --method M --dataset D
    open-road eval   --method M --dataset D
    open-road run    --method M --dataset D
    open-road report --runs runs/a/road_anomaly runs/b/road_anomaly

Only ``infer`` imports the method. ``render``, ``eval`` and ``report`` read the
score maps on disk, so they work in a plain environment even when the method's
own dependency pins are not installed there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import typer
import yaml

from open_road import registry
from open_road.dataset import DatasetSpec
from open_road.io import RunLayout
from open_road.method import MethodSpec
from open_road.paths import repo_root, run_dir
from open_road.stages.evaluate import run_evaluate
from open_road.stages.infer import run_infer
from open_road.stages.render import run_render
from open_road.stages.report import run_report
from open_road.stages.video import run_video

HELP = """The same five verbs for every method.

Start with `open-road methods` and `open-road datasets` to see what this branch
has, then `open-road run --method M --dataset D` to score, render and evaluate
in one go. Only `infer` imports the method: render, eval and report read the
score maps from disk, so they work even where the method's own pins are not
installed.
"""

app = typer.Typer(no_args_is_help=True, add_completion=False, help=HELP)

METHOD_OPTION = typer.Option(..., "--method", "-m", help="Directory name under methods/")
DATASET_OPTION = typer.Option("road_anomaly", "--dataset", "-d", help="Name under configs/datasets/")


def _datasets_dir() -> Path:
    return repo_root() / "configs" / "datasets"


def _method_config_path(name: str) -> Path:
    return repo_root() / "configs" / "methods" / f"{name}.yaml"


def _load_dataset(name: str) -> DatasetSpec:
    path = name if name.endswith((".yaml", ".yml")) else _datasets_dir() / f"{name}.yaml"
    try:
        return DatasetSpec.from_yaml(path)
    except FileNotFoundError:
        available = sorted(p.stem for p in _datasets_dir().glob("*.yaml"))
        raise typer.BadParameter(
            f"unknown dataset {name!r}; available: {', '.join(available) or 'none'}"
        ) from None


def _load_method(name: str) -> MethodSpec:
    try:
        return registry.load(name)
    except LookupError as error:
        raise typer.BadParameter(str(error)) from None


def _spec_if_available(name: str) -> Optional[MethodSpec]:
    """The spec, or None when the method's dependencies are absent.

    Used by `eval`, which needs nothing from the method but would still like
    its default min_area. Evaluation must keep working in an environment where
    the method itself cannot be imported.
    """
    try:
        return registry.load(name)
    except Exception:  # noqa: BLE001 -- absence is expected here, not exceptional
        return None


def _method_settings(name: str, override: Optional[Path]) -> dict[str, Any]:
    path = Path(override) if override else _method_config_path(name)
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _layout(method: str, dataset: str) -> RunLayout:
    return RunLayout(run_dir(method, dataset))


@app.command()
def methods() -> None:
    """List the methods present on this branch."""
    found = registry.discover()
    if not found.working and not found.broken:
        typer.echo(
            "no methods on this branch. `main` carries the skeleton only — "
            "check out a method branch, e.g. `git checkout raas-distill`."
        )
        raise typer.Exit()

    for name in found.names:
        spec = found.working[name]
        typer.echo(f"{name:<20} {spec.description}")
        typer.echo(
            f"{'':<20} threshold {spec.default_threshold:g}  "
            f"min_area {spec.default_min_area}  "
            f"score range [{spec.score_range[0]:g}, {spec.score_range[1]:g}]"
        )

    for broken in found.broken:
        typer.echo(f"{broken.name:<20} UNAVAILABLE — {broken.reason}", err=True)
        typer.echo(
            f"{'':<20} install it with: "
            f"pip install -r methods/{broken.name}/requirements.txt",
            err=True,
        )


@app.command()
def datasets() -> None:
    """List the datasets configured in configs/datasets/."""
    paths = sorted(_datasets_dir().glob("*.yaml"))
    if not paths:
        typer.echo("no dataset configs in configs/datasets/")
        raise typer.Exit()
    for path in paths:
        spec = DatasetSpec.from_yaml(path)
        state = "ready" if spec.images_path.is_dir() else "MISSING"
        labels = "labelled" if spec.has_labels else "unlabelled"
        typer.echo(f"{spec.name:<20} {state:<8} {labels:<11} {spec.root}")


@app.command()
def infer(
    method: str = METHOD_OPTION,
    dataset: str = DATASET_OPTION,
    config: Optional[Path] = typer.Option(None, help="Method YAML; defaults to configs/methods/<method>.yaml"),
    limit: int = typer.Option(0, help="Score only the first N frames (0 = all)"),
    overwrite: bool = typer.Option(False, help="Re-score frames that already have a map"),
    save_intermediate: bool = typer.Option(
        True, help="Write the method's internal maps under intermediate/, when it exposes any"
    ),
) -> None:
    """Score every frame of a dataset and store the raw maps."""
    spec = _load_method(method)
    data = _load_dataset(dataset)
    layout = _layout(method, data.name)
    summary = run_infer(
        spec,
        data,
        layout,
        _method_settings(method, config),
        limit=limit,
        overwrite=overwrite,
        save_intermediate=save_intermediate,
        report=typer.echo,
    )
    typer.echo(f"scores  -> {layout.score_raw}")
    typer.echo(f"preview -> {layout.soft_mask}")
    if summary.get("intermediate"):
        typer.echo(f"internals -> {layout.intermediate}")


@app.command()
def render(
    method: str = METHOD_OPTION,
    dataset: str = DATASET_OPTION,
    threshold: Optional[float] = typer.Option(None, help="Score cut; defaults to the method's"),
    min_area: Optional[int] = typer.Option(None, help="Drop components smaller than this"),
    draw: str = typer.Option("seg", help="seg, boxes, or both"),
    alpha: float = typer.Option(0.5, help="Fill opacity"),
    labels: bool = typer.Option(True, help="Outline ground truth when the dataset has it"),
) -> None:
    """Threshold the score maps into masks, regions and overlays."""
    if draw not in {"seg", "boxes", "both"}:
        raise typer.BadParameter("draw must be seg, boxes, or both")
    spec = _load_method(method)
    data = _load_dataset(dataset)
    run_render(
        spec,
        data,
        _layout(method, data.name),
        threshold=threshold,
        min_area=min_area,
        mode=draw,
        alpha=alpha,
        draw_labels=labels,
        report=typer.echo,
    )


@app.command("eval")
def evaluate(
    method: str = METHOD_OPTION,
    dataset: str = DATASET_OPTION,
    threshold: Optional[float] = typer.Option(
        None, help="Operating point for the component metrics; default is the best-F1 threshold"
    ),
    min_area: Optional[int] = typer.Option(
        None, help="Component filter; must match render's. Defaults to the method's"
    ),
) -> None:
    """Score the stored maps against the dataset's labels."""
    data = _load_dataset(dataset)
    layout = _layout(method, data.name)
    if min_area is None:
        spec = _spec_if_available(method)
        min_area = spec.default_min_area if spec else 0
        if spec is None:
            typer.echo(
                f"{method} could not be imported, so its default min_area is unknown; "
                f"using 0. Pass --min-area to match what render used.",
                err=True,
            )
    run_evaluate(
        data, layout, threshold=threshold, min_area=min_area, method=method, report=typer.echo
    )
    typer.echo(f"\nmetrics -> {layout.metrics}")


@app.command()
def run(
    method: str = METHOD_OPTION,
    dataset: str = DATASET_OPTION,
    config: Optional[Path] = typer.Option(None, help="Method YAML"),
    limit: int = typer.Option(0, help="Score only the first N frames (0 = all)"),
    overwrite: bool = typer.Option(False, help="Re-score frames that already have a map"),
    draw: str = typer.Option("seg", help="seg, boxes, or both"),
    save_intermediate: bool = typer.Option(
        True, help="Write the method's internal maps under intermediate/"
    ),
    video_fps: float = typer.Option(
        0.0, help="Also encode the overlays into an mp4 at this rate (0 = skip)"
    ),
) -> None:
    """infer, then eval, then render at the operating point eval found.

    Eval runs before render on purpose. It sweeps the threshold and reports the
    best-F1 operating point; rendering at the method's fixed default instead
    would draw one mask and score a different one.
    """
    spec = _load_method(method)
    data = _load_dataset(dataset)
    layout = _layout(method, data.name)

    typer.echo("== infer ==")
    summary = run_infer(
        spec, data, layout, _method_settings(method, config),
        limit=limit, overwrite=overwrite, save_intermediate=save_intermediate,
        report=typer.echo,
    )

    threshold = spec.default_threshold
    if data.has_labels:
        typer.echo("\n== eval ==")
        metrics = run_evaluate(
            data, layout, min_area=spec.default_min_area, method=method, report=typer.echo
        )
        threshold = metrics["threshold"]
    else:
        typer.echo("\ndataset is unlabelled; skipping eval")

    typer.echo(f"\n== render (threshold {threshold:.4f}) ==")
    run_render(spec, data, layout, threshold=threshold, mode=draw, report=typer.echo)

    if video_fps > 0:
        typer.echo(f"\n== video ==")
        run_video(data, layout, fps=video_fps, report=typer.echo)

    typer.echo(f"\nrun      -> {layout.root}")
    typer.echo(f"overlays -> {layout.overlay}")
    typer.echo(f"masks    -> {layout.mask}")
    if summary.get("intermediate"):
        typer.echo(f"internals-> {layout.intermediate}")


@app.command()
def video(
    method: str = METHOD_OPTION,
    dataset: str = DATASET_OPTION,
    fps: float = typer.Option(10.0, help="Playback rate of the encoded clip"),
    source: str = typer.Option("overlay", help="overlay or mask"),
    out: Optional[Path] = typer.Option(None, help="Destination .mp4"),
) -> None:
    """Encode a rendered clip back into a video, in the dataset's frame order."""
    data = _load_dataset(dataset)
    run_video(
        data,
        _layout(method, data.name),
        fps=fps,
        source=source,
        destination=out,
        report=typer.echo,
    )


@app.command()
def report(
    runs: list[Path] = typer.Option(..., "--runs", help="Run directories to compare"),
    out: Optional[Path] = typer.Option(None, help="Write comparison.{md,csv,json} here"),
) -> None:
    """Put several runs' metrics side by side."""
    run_report(runs, out, report=typer.echo)


if __name__ == "__main__":
    app()
