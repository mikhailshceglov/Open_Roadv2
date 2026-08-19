"""Import refined Objectomaly maps into the official SMIYC evaluator."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MASKOMALY_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MASKOMALY_DIR / "maskomaly"))
sys.path.insert(0, str(SCRIPT_DIR))

from objectomaly_cache import (
    index_entries,
    read_manifest,
    resolve_cached_path,
    validate_anomaly_map,
)
import run_smiyc_eval as smiyc


def merge_summary_rows(output: Path, rows):
    """Merge repeated model imports into one stable model/dataset summary."""
    existing = []
    summary_path = Path(output) / "summary.json"
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("unit") != "percent" or not isinstance(payload.get("results"), list):
            raise ValueError("Unexpected existing summary format: {}".format(summary_path))
        existing = payload["results"]
    merged = {}
    order = []
    for row in list(existing) + list(rows):
        key = (str(row["model"]), str(row["dataset"]))
        if key not in merged:
            order.append(key)
        merged[key] = row
    return [merged[key] for key in order]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import Objectomaly predictions and calculate official SMIYC metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--datasets-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method-name")
    parser.add_argument("--phase", choices=("all", "import", "metrics"), default="all")
    parser.add_argument("--visualize", action="store_true")
    return parser


def import_predictions(evaluation, entry_index, manifest_path: Path) -> int:
    processed = 0
    expected = len(evaluation)
    for frame in evaluation.get_frames():
        key = (str(frame.dset_name), str(frame.fid))
        try:
            entry = entry_index[key]
        except KeyError as exc:
            raise KeyError("Refined manifest is missing {} / {}".format(*key)) from exc
        refined_rel = entry.get("refined_map")
        if not refined_rel:
            raise ValueError(
                "Entry {} / {} has no refined_map; run refinement phase first".format(*key)
            )
        prediction = np.load(
            str(resolve_cached_path(manifest_path, refined_rel)), allow_pickle=False
        )
        prediction = validate_anomaly_map(prediction, frame.image.shape[:2])
        evaluation.save_output(frame, prediction)
        processed += 1
    evaluation.wait_to_finish_saving()
    if processed != expected:
        raise RuntimeError("Imported {} of {} predictions".format(processed, expected))
    return processed


def execute(args, evaluation_class=None):
    args.manifest = args.manifest.expanduser().resolve()
    args.datasets_root = args.datasets_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    smiyc.validate_dataset_layout(args.datasets_root)
    payload = read_manifest(args.manifest)
    if payload.get("kind") != "raas-objectomaly-refined":
        raise ValueError("Unexpected manifest kind: {}".format(payload.get("kind")))
    entry_index = index_entries(payload["entries"])
    method_name = args.method_name or "objectomaly_{}".format(payload["source_model"])
    args.output.mkdir(parents=True, exist_ok=True)
    if evaluation_class is None:
        evaluation_class = smiyc.import_official_evaluation(args.datasets_root, args.output)

    rows = []
    for dataset_name, spec in smiyc.DATASETS.items():
        evaluation = evaluation_class(
            method_name=method_name,
            dataset_name=dataset_name,
            threaded_saver=False,
        )
        if args.phase in ("all", "import"):
            count = import_predictions(evaluation, entry_index, args.manifest)
            print("{} / {}: imported {} predictions".format(method_name, dataset_name, count))
        if args.phase in ("all", "metrics"):
            smiyc.require_saved_predictions(evaluation, args.output, method_name)
            metrics = smiyc.calculate_metrics(
                evaluation, spec["segment_metric"], args.visualize
            )
            row = {"model": method_name, "dataset": dataset_name}
            row.update(metrics)
            rows.append(row)

    if rows:
        combined_rows = merge_summary_rows(args.output, rows)
        smiyc.write_summary(combined_rows, args.output)
        smiyc.print_summary(combined_rows)
        print("Summary: {}".format(args.output / "summary.csv"))
    return rows


def main() -> None:
    execute(build_parser().parse_args())


if __name__ == "__main__":
    main()
