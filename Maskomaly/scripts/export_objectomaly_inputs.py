"""Export lossless BGR images and float32 Maskomaly maps for Objectomaly."""

import argparse
import gc
from pathlib import Path
import sys
import time

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MASKOMALY_DIR = SCRIPT_DIR.parent
RAAS_DIR = MASKOMALY_DIR.parent
sys.path.insert(0, str(MASKOMALY_DIR / "maskomaly"))
sys.path.insert(0, str(SCRIPT_DIR))

from objectomaly_cache import (
    validate_anomaly_map,
    validate_frame_id,
    validate_semantic_map,
    write_manifest,
)
from inference_timing import runtime_environment, timed_call
import run_smiyc_eval as smiyc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export RAAS coarse maps for the cache-first Objectomaly pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--models", nargs="+", choices=tuple(smiyc.MODEL_MODULES), default=["maskomaly"]
    )
    parser.add_argument("--config-file", type=Path, default=smiyc.DEFAULT_CONFIG)
    parser.add_argument("--weights", type=Path, default=smiyc.DEFAULT_WEIGHTS)
    parser.add_argument("--masks", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--analysis-file", default=None, help=argparse.SUPPRESS)
    return parser


def export_evaluation(evaluation, model, model_name: str, output: Path):
    import cv2

    entries = []
    expected = len(evaluation)
    for frame in evaluation.get_frames():
        fid = validate_frame_id(frame.fid)
        dataset = validate_frame_id(str(frame.dset_name))
        bgr = smiyc.rgb_to_bgr(frame.image)
        prediction, raas_inference_s = timed_call(lambda: model.get_soft_mask(bgr))
        postprocess_started = time.perf_counter()
        coarse = smiyc.prepare_anomaly_map(prediction, bgr.shape[:2])
        coarse = validate_anomaly_map(coarse, bgr.shape[:2])
        semantic = getattr(model, "last_semantic_segmentation", None)
        if semantic is None:
            raise RuntimeError("Model did not expose last_semantic_segmentation")
        semantic = np.asarray(semantic)
        if semantic.shape != bgr.shape[:2]:
            semantic = cv2.resize(
                semantic.astype(np.int16),
                (bgr.shape[1], bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        semantic = validate_semantic_map(semantic, bgr.shape[:2])
        raas_postprocess_s = float(time.perf_counter() - postprocess_started)

        image_rel = Path("images") / dataset / (fid + ".npy")
        coarse_rel = Path("coarse") / model_name / dataset / (fid + ".npy")
        semantic_rel = Path("semantic") / model_name / dataset / (fid + ".npy")
        for rel, value in (
            (image_rel, bgr),
            (coarse_rel, coarse),
            (semantic_rel, semantic),
        ):
            path = output / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(path), value, allow_pickle=False)

        entries.append(
            {
                "dataset": dataset,
                "evaluation_dataset": evaluation.dataset_name,
                "fid": fid,
                "height": int(bgr.shape[0]),
                "width": int(bgr.shape[1]),
                "image_bgr": str(image_rel),
                "coarse_map": str(coarse_rel),
                "semantic_map": str(semantic_rel),
                "timings": {
                    "raas_inference_s": raas_inference_s,
                    "raas_postprocess_s": raas_postprocess_s,
                },
            }
        )
    if len(entries) != expected:
        raise RuntimeError("Exported {} of {} frames".format(len(entries), expected))
    return entries


def execute(args, evaluation_class=None, model_loader=smiyc.load_model):
    args.datasets_root = args.datasets_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.config_file = args.config_file.expanduser().resolve()
    args.weights = args.weights.expanduser().resolve()
    smiyc.preflight(argparse.Namespace(**dict(vars(args), phase="inference")))
    args.output.mkdir(parents=True, exist_ok=True)
    if evaluation_class is None:
        evaluation_class = smiyc.import_official_evaluation(args.datasets_root, args.output)

    manifests = []
    for model_name in args.models:
        print("Loading model: {}".format(model_name))
        model, model_load_s = timed_call(lambda: model_loader(model_name, args))
        entries = []
        for dataset_name in smiyc.DATASETS:
            evaluation = evaluation_class(
                method_name=model_name,
                dataset_name=dataset_name,
                threaded_saver=False,
            )
            exported = export_evaluation(evaluation, model, model_name, args.output)
            entries.extend(exported)
            print("{} / {}: exported {} frames".format(model_name, dataset_name, len(exported)))
        manifest_path = args.output / ("manifest-{}.json".format(model_name))
        write_manifest(
            manifest_path,
            {
                "kind": "raas-objectomaly-inputs",
                "source_model": model_name,
                "objectomaly_commit": "66d2ad2a1b02d79389f4265d9d1d99ab6412324f",
                "setup_timings_s": {"raas_model_load_s": model_load_s},
                "runtime": runtime_environment(),
                "entries": entries,
            },
        )
        manifests.append(manifest_path)
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return manifests


def main() -> None:
    execute(build_parser().parse_args())


if __name__ == "__main__":
    main()
