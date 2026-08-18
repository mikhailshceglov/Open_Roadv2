"""Run RAAS inference and the official SegmentMeIfYouCan validation metrics.

The official benchmark is included as a pinned git submodule.  This wrapper
sets its dataset/output roots before importing it, feeds RAAS anomaly maps to
its Evaluation API, and writes a compact CSV/JSON summary.
"""

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MASKOMALY_DIR = SCRIPT_DIR.parent
RAAS_DIR = MASKOMALY_DIR.parent
BENCHMARK_DIR = RAAS_DIR / "third_party" / "road-anomaly-benchmark"
DEFAULT_CONFIG = (
    RAAS_DIR
    / "Mask2Former/configs/cityscapes/semantic-segmentation/swin/"
    / "maskformer2_swin_large_IN21k_384_bs16_90k.yaml"
)
DEFAULT_WEIGHTS = MASKOMALY_DIR / "maskomaly/ckpt/model_final_17c1ee.pkl"

MODEL_MODULES = {
    "maskomaly": "model_ori",
    "maskomaly_id": "model_id",
    "maskomaly_ood": "model_ood",
}

DATASETS = {
    "AnomalyTrack-validation": {
        "directory": "dataset_AnomalyTrack",
        "segment_metric": "SegEval-AnomalyTrack",
    },
    "ObstacleTrack-validation": {
        "directory": "dataset_ObstacleTrack",
        "segment_metric": "SegEval-ObstacleTrack",
    },
}

SUMMARY_COLUMNS = (
    "model",
    "dataset",
    "AUPR",
    "FPR@95",
    "sIoU GT",
    "PPV",
    "mean F1",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAAS evaluation with the official SMIYC validation protocol",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets-root",
        required=True,
        type=Path,
        help="Parent of dataset_AnomalyTrack and dataset_ObstacleTrack.",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Official outputs and summary root."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_MODULES),
        default=list(MODEL_MODULES),
    )
    parser.add_argument(
        "--phase", choices=("all", "inference", "metrics"), default="all"
    )
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save official per-frame metric visualizations.",
    )
    parser.add_argument("--masks", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--analysis-file", default=None, help=argparse.SUPPRESS)
    return parser


def configure_import_paths() -> None:
    """Make the repo-local model sources and official benchmark importable."""
    ordered_paths = (
        MASKOMALY_DIR / "detectron2_replacements",
        MASKOMALY_DIR / "maskomaly",
        RAAS_DIR / "detectron2",
        RAAS_DIR / "Mask2Former",
        BENCHMARK_DIR,
    )
    for path in reversed(ordered_paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def validate_dataset_layout(datasets_root: Path) -> None:
    errors = []
    for spec in DATASETS.values():
        root = datasets_root / spec["directory"]
        for child in ("images", "labels_masks"):
            path = root / child
            if not path.is_dir():
                errors.append("missing directory: {}".format(path))
    if errors:
        raise FileNotFoundError("Invalid SMIYC dataset layout:\n- " + "\n- ".join(errors))


def preflight(args: argparse.Namespace) -> None:
    datasets_root = args.datasets_root.expanduser().resolve()
    validate_dataset_layout(datasets_root)

    benchmark_package = BENCHMARK_DIR / "road_anomaly_benchmark" / "evaluation.py"
    if not benchmark_package.is_file():
        raise FileNotFoundError(
            "Official benchmark submodule is missing. Run: "
            "git submodule update --init --recursive"
        )

    if args.phase in ("all", "inference"):
        for label, path in (
            ("Mask2Former config", args.config_file),
            ("model weights", args.weights),
        ):
            resolved = path.expanduser().resolve()
            if not resolved.is_file():
                raise FileNotFoundError("{} not found: {}".format(label, resolved))


def import_official_evaluation(datasets_root: Path, output: Path):
    """Import Evaluation only after configuring its environment-backed paths."""
    os.environ["DIR_DATASETS"] = str(datasets_root.expanduser().resolve())
    os.environ["DIR_OUTPUTS"] = str(output.expanduser().resolve())
    configure_import_paths()
    try:
        module = importlib.import_module("road_anomaly_benchmark.evaluation")
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import the official SMIYC evaluator. Install the minimal "
            "dependencies with `pip install -r Maskomaly/requirements-smiyc.txt`. "
            "Original error: {}".format(exc)
        ) from exc
    install_hdf5_numpy_scalar_compatibility()
    return module.Evaluation


def install_hdf5_numpy_scalar_compatibility() -> None:
    """Allow the upstream HDF5 writer to persist NumPy scalar aggregates.

    The pinned benchmark's component metric produces ``np.int64`` counters,
    while its serializer accepts only Python ``int``/``float``.  Converting a
    scalar with ``item()`` changes serialization only, not metric computation.
    """
    dataset_io = importlib.import_module(
        "road_anomaly_benchmark.datasets.dataset_io"
    )
    current = dataset_io.hdf5_write_hierarchy_to_group
    if getattr(current, "_raas_numpy_scalar_compatible", False):
        return

    def compatible_writer(group, hierarchy):
        normalized = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in hierarchy.items()
        }
        return current(group, normalized)

    compatible_writer._raas_numpy_scalar_compatible = True
    dataset_io.hdf5_write_hierarchy_to_group = compatible_writer


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB HxWx3 image, got {}".format(image.shape))
    return np.ascontiguousarray(image[:, :, ::-1])


def prepare_anomaly_map(
    prediction: np.ndarray,
    target_hw: Tuple[int, int],
    resize_fn=None,
) -> np.ndarray:
    """Validate, resize and clip a model prediction for the benchmark API."""
    if isinstance(prediction, tuple):
        prediction = prediction[0]
    anomaly_map = np.asarray(prediction, dtype=np.float32)
    if anomaly_map.ndim > 2:
        anomaly_map = anomaly_map.squeeze()
    if anomaly_map.ndim != 2:
        raise ValueError("Expected a 2-D anomaly map, got {}".format(anomaly_map.shape))
    if not np.all(np.isfinite(anomaly_map)):
        raise ValueError("Anomaly map contains NaN or infinity")

    if anomaly_map.shape != target_hw:
        if resize_fn is None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("OpenCV is required to resize anomaly maps") from exc
            resize_fn = lambda value, hw: cv2.resize(
                value, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR
            )
        anomaly_map = np.asarray(resize_fn(anomaly_map, target_hw), dtype=np.float32)

    if anomaly_map.shape != target_hw:
        raise ValueError(
            "Resized anomaly map has {}, expected {}".format(
                anomaly_map.shape, target_hw
            )
        )
    return np.ascontiguousarray(np.clip(anomaly_map, 0.0, 1.0), dtype=np.float32)


def model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        config_file=str(args.config_file.expanduser().resolve()),
        opts=["MODEL.WEIGHTS", str(args.weights.expanduser().resolve())],
        masks=args.masks,
        analysis_file=args.analysis_file,
    )


def load_model(model_name: str, args: argparse.Namespace):
    configure_import_paths()
    module = importlib.import_module(MODEL_MODULES[model_name])
    return module.Maskomaly(model_args(args))


def run_inference(evaluation, model) -> int:
    expected = len(evaluation)
    processed = 0
    for frame in evaluation.get_frames():
        bgr_image = rgb_to_bgr(frame.image)
        prediction = model.get_soft_mask(bgr_image)
        anomaly_map = prepare_anomaly_map(prediction, frame.image.shape[:2])
        evaluation.save_output(frame, anomaly_map)
        processed += 1

    evaluation.wait_to_finish_saving()
    if processed != expected:
        raise RuntimeError(
            "Saved {} predictions, but dataset contains {} frames".format(
                processed, expected
            )
        )
    return processed


def expected_prediction_paths(evaluation, output: Path, model_name: str) -> List[Path]:
    paths = []
    for frame in evaluation.get_frames():
        paths.append(
            output
            / "anomaly_p"
            / model_name
            / frame.dset_name
            / "{}.hdf5".format(frame.fid)
        )
    return paths


def require_saved_predictions(evaluation, output: Path, model_name: str) -> None:
    missing = [
        path
        for path in expected_prediction_paths(evaluation, output, model_name)
        if not path.is_file()
    ]
    if missing:
        preview = "\n- ".join(str(path) for path in missing[:5])
        suffix = "\n... and {} more".format(len(missing) - 5) if len(missing) > 5 else ""
        raise FileNotFoundError(
            "Missing saved predictions for {}:\n- {}{}".format(
                model_name, preview, suffix
            )
        )


def _as_percent(value) -> Optional[float]:
    value = float(value)
    return 100.0 * value if np.isfinite(value) else None


def calculate_metrics(evaluation, segment_metric: str, visualize: bool) -> Dict[str, Optional[float]]:
    pixel = evaluation.calculate_metric_from_saved_outputs(
        "PixBinaryClass", parallel=False, frame_vis=visualize
    )
    segment = evaluation.calculate_metric_from_saved_outputs(
        segment_metric, parallel=False, frame_vis=visualize
    )
    return {
        "AUPR": _as_percent(pixel.area_PRC),
        "FPR@95": _as_percent(pixel.tpr95_fpr),
        "sIoU GT": _as_percent(segment.sIoU_gt),
        "PPV": _as_percent(segment.prec_pred),
        "mean F1": _as_percent(segment.f1_mean),
    }


def write_summary(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "summary.csv"
    json_path = output / "summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    payload = {"unit": "percent", "results": list(rows)}
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def print_summary(rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    formatted = []
    for row in rows:
        metric_values = []
        for key in SUMMARY_COLUMNS[2:]:
            value = row[key]
            metric_values.append("n/a" if value is None else "{:.2f}".format(float(value)))
        formatted.append(
            [
                str(row["model"]),
                str(row["dataset"]),
                *metric_values,
            ]
        )
    widths = [len(name) for name in SUMMARY_COLUMNS]
    for row in formatted:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    def render(values: Iterable[str]) -> str:
        return " | ".join(value.ljust(width) for value, width in zip(values, widths))

    print("\nOfficial SMIYC validation metrics (percent)")
    print(render(SUMMARY_COLUMNS))
    print("-+-".join("-" * width for width in widths))
    for row in formatted:
        print(render(row))


def execute(args: argparse.Namespace, evaluation_class=None, model_loader=load_model) -> List[Dict[str, object]]:
    args.datasets_root = args.datasets_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.config_file = args.config_file.expanduser().resolve()
    args.weights = args.weights.expanduser().resolve()
    preflight(args)
    args.output.mkdir(parents=True, exist_ok=True)

    if evaluation_class is None:
        evaluation_class = import_official_evaluation(args.datasets_root, args.output)

    rows = []
    for model_name in args.models:
        model = None
        if args.phase in ("all", "inference"):
            print("\nLoading model: {}".format(model_name))
            model = model_loader(model_name, args)

        for dataset_name, spec in DATASETS.items():
            evaluation = evaluation_class(
                method_name=model_name,
                dataset_name=dataset_name,
                threaded_saver=False,
            )
            if model is not None:
                count = run_inference(evaluation, model)
                print("{} / {}: saved {} predictions".format(model_name, dataset_name, count))

            if args.phase in ("all", "metrics"):
                require_saved_predictions(evaluation, args.output, model_name)
                metrics = calculate_metrics(
                    evaluation, spec["segment_metric"], args.visualize
                )
                row = {"model": model_name, "dataset": dataset_name}
                row.update(metrics)
                rows.append(row)

        if model is not None:
            del model
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    if rows:
        write_summary(rows, args.output)
        print_summary(rows)
        print("\nSummary: {}".format(args.output / "summary.csv"))
    return rows


def main() -> None:
    args = build_parser().parse_args()
    execute(args)


if __name__ == "__main__":
    main()
