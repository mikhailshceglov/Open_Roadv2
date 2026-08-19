"""RAAS + Objectomaly inference on an arbitrary image folder without GT.

The two subcommands intentionally run in separate conda environments:

* ``export`` runs RAAS and writes lossless images plus float32 coarse maps;
* ``refine`` runs SAM/OASC/MBP and renders inspection-ready visualizations.
"""

import argparse
import gc
import re
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MASKOMALY_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MASKOMALY_DIR / "maskomaly"))
sys.path.insert(0, str(SCRIPT_DIR))

from objectomaly_cache import (
    read_manifest,
    resolve_cached_path,
    validate_anomaly_map,
    validate_bgr_image,
    validate_semantic_map,
    write_manifest,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CUSTOM_DATASET_NAME = "custom-folder"
DEFAULT_FOLDER_CONFIG = MASKOMALY_DIR / "configs" / "objectomaly_global_fusion.json"


def build_parser():
    parser = argparse.ArgumentParser(
        description="RAAS + Objectomaly inference on images without ground truth"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser(
        "export", help="Run RAAS and cache float anomaly maps"
    )
    export.add_argument(
        "--model",
        choices=("maskomaly", "maskomaly_id", "maskomaly_ood"),
        required=True,
    )
    export.add_argument("--input", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--recursive", action="store_true")
    export.add_argument("--config-file", type=Path)
    export.add_argument("--weights", type=Path)
    export.add_argument("--masks", type=int, default=4, help=argparse.SUPPRESS)
    export.add_argument("--analysis-file", default=None, help=argparse.SUPPRESS)

    refine = commands.add_parser(
        "refine", help="Run SAM/OASC/MBP and save visualizations"
    )
    refine.add_argument("--manifest", required=True, type=Path)
    refine.add_argument("--output", required=True, type=Path)
    refine.add_argument("--config", type=Path, default=DEFAULT_FOLDER_CONFIG)
    refine.add_argument("--sam-checkpoint", required=True, type=Path)
    refine.add_argument("--device", default="cuda")
    refine.add_argument("--phase", choices=("all", "masks", "refine"), default="all")
    refine.add_argument("--threshold", type=float, default=0.5)
    return parser


def discover_images(root: Path, recursive: bool):
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("Input image directory not found: {}".format(root))
    candidates = root.rglob("*") if recursive else root.iterdir()
    images = sorted(
        (path for path in candidates if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not images:
        raise FileNotFoundError("No JPG, JPEG or PNG images found in {}".format(root))
    return root, images


def make_frame_id(index: int, relative_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", relative_path.stem).strip("._-")
    if not stem:
        stem = "image"
    return "{:06d}_{}".format(index, stem[:80])


def export_folder(args, model_loader=None):
    import cv2
    import run_smiyc_eval as smiyc

    input_root, image_paths = discover_images(args.input, args.recursive)
    output = args.output.expanduser().resolve()
    config = (args.config_file or smiyc.DEFAULT_CONFIG).expanduser().resolve()
    weights = (args.weights or smiyc.DEFAULT_WEIGHTS).expanduser().resolve()
    for label, path in (("Mask2Former config", config), ("model weights", weights)):
        if not path.is_file():
            raise FileNotFoundError("{} not found: {}".format(label, path))

    model_args = SimpleNamespace(
        config_file=config,
        weights=weights,
        masks=args.masks,
        analysis_file=args.analysis_file,
    )
    model = (model_loader or smiyc.load_model)(args.model, model_args)
    entries = []
    for index, image_path in enumerate(image_paths):
        relative = image_path.relative_to(input_root)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Cannot decode image: {}".format(image_path))
        image = validate_bgr_image(image)
        prediction = model.get_soft_mask(image)
        coarse = smiyc.prepare_anomaly_map(prediction, image.shape[:2])
        coarse = validate_anomaly_map(coarse, image.shape[:2])
        semantic = getattr(model, "last_semantic_segmentation", None)
        if semantic is None:
            raise RuntimeError("Model did not expose last_semantic_segmentation")
        semantic = np.asarray(semantic)
        if semantic.shape != image.shape[:2]:
            semantic = cv2.resize(
                semantic.astype(np.int16),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        semantic = validate_semantic_map(semantic, image.shape[:2])
        fid = make_frame_id(index, relative)

        image_rel = Path("images") / CUSTOM_DATASET_NAME / (fid + ".npy")
        coarse_rel = Path("coarse") / args.model / CUSTOM_DATASET_NAME / (fid + ".npy")
        semantic_rel = Path("semantic") / args.model / CUSTOM_DATASET_NAME / (fid + ".npy")
        for relative_cache, value in (
            (image_rel, image),
            (coarse_rel, coarse),
            (semantic_rel, semantic),
        ):
            destination = output / relative_cache
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(destination), value, allow_pickle=False)

        entries.append(
            {
                "dataset": CUSTOM_DATASET_NAME,
                "fid": fid,
                "source_name": relative.as_posix(),
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
                "image_bgr": str(image_rel),
                "coarse_map": str(coarse_rel),
                "semantic_map": str(semantic_rel),
            }
        )
        print("[{}/{}] {}".format(index + 1, len(image_paths), relative.as_posix()))

    manifest = output / ("manifest-{}.json".format(args.model))
    write_manifest(
        manifest,
        {
            "kind": "raas-objectomaly-folder-inputs",
            "source_model": args.model,
            "input_root": str(input_root),
            "entries": entries,
        },
    )
    print("Manifest: {}".format(manifest))
    return manifest


def _score_to_uint8(score):
    return np.rint(np.clip(score, 0.0, 1.0) * 255.0).astype(np.uint8)


def _overlay(image, score, cv2):
    heatmap = cv2.applyColorMap(_score_to_uint8(score), cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.6, heatmap, 0.4, 0.0), heatmap


def _label(image, value, cv2):
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (240, 36), (0, 0, 0), thickness=-1)
    cv2.putText(
        labeled,
        value,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def _semantic_color(semantic):
    palette = np.array(
        [
            [128, 64, 128], [232, 35, 244], [70, 70, 70], [156, 102, 102],
            [153, 153, 190], [153, 153, 153], [30, 170, 250], [0, 220, 220],
            [35, 142, 107], [152, 251, 152], [180, 130, 70], [60, 20, 220],
            [0, 0, 255], [142, 0, 0], [70, 0, 0], [100, 60, 0], [100, 80, 0],
            [230, 0, 0], [32, 11, 119],
        ],
        dtype=np.uint8,
    )
    result = np.zeros(semantic.shape + (3,), dtype=np.uint8)
    valid = np.logical_and(semantic >= 0, semantic < len(palette))
    result[valid] = palette[semantic[valid]]
    return result


def render_outputs(manifest_path: Path, output: Path, threshold: float):
    import cv2

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    payload = read_manifest(manifest_path)
    if payload.get("kind") != "raas-objectomaly-folder-refined":
        raise ValueError("Unexpected refined manifest kind: {}".format(payload.get("kind")))
    input_manifest_value = payload.get("input_manifest")
    if not input_manifest_value:
        raise ValueError("Refined manifest has no input_manifest")
    input_manifest = Path(str(input_manifest_value)).expanduser().resolve()
    source_payload = read_manifest(input_manifest)
    if source_payload.get("kind") != "raas-objectomaly-folder-inputs":
        raise ValueError("Unexpected source manifest kind: {}".format(source_payload.get("kind")))
    visual_root = output / "visualizations"
    for entry in payload["entries"]:
        if not entry.get("refined_map"):
            continue
        target_hw = (int(entry["height"]), int(entry["width"]))
        image = validate_bgr_image(
            np.load(str(resolve_cached_path(input_manifest, entry["image_bgr"])), allow_pickle=False),
            target_hw,
        )
        coarse = validate_anomaly_map(
            np.load(str(resolve_cached_path(input_manifest, entry["coarse_map"])), allow_pickle=False),
            target_hw,
        )
        semantic = validate_semantic_map(
            np.load(
                str(resolve_cached_path(input_manifest, entry["semantic_map"])),
                allow_pickle=False,
            ),
            target_hw,
        )
        if entry.get("fused_map"):
            fused = validate_anomaly_map(
                np.load(
                    str(resolve_cached_path(manifest_path, entry["fused_map"])),
                    allow_pickle=False,
                ),
                target_hw,
            )
        else:
            fused = coarse
        refined = validate_anomaly_map(
            np.load(str(resolve_cached_path(manifest_path, entry["refined_map"])), allow_pickle=False),
            target_hw,
        )
        coarse_overlay, coarse_heatmap = _overlay(image, coarse, cv2)
        fused_overlay, fused_heatmap = _overlay(image, fused, cv2)
        refined_overlay, refined_heatmap = _overlay(image, refined, cv2)
        comparison = np.concatenate(
            (
                _label(image, "Input", cv2),
                _label(coarse_overlay, "RAAS", cv2),
                _label(fused_overlay, "Global fusion", cv2),
                _label(refined_overlay, "Final", cv2),
            ),
            axis=1,
        )
        if entry.get("protected_candidates"):
            protected = np.load(
                str(resolve_cached_path(manifest_path, entry["protected_candidates"])),
                allow_pickle=False,
            ).astype(bool)
        else:
            protected = np.zeros(target_hw, dtype=bool)
        if protected.shape != target_hw:
            raise ValueError("Protected candidate mask has invalid shape")
        candidate_overlay = image.copy()
        candidate_overlay[protected] = (
            0.35 * candidate_overlay[protected] + 0.65 * np.array([0, 0, 255])
        ).astype(np.uint8)
        name = entry["fid"]
        destinations = {
            visual_root / "coarse_mask" / (name + ".png"): _score_to_uint8(coarse),
            visual_root / "fused_mask" / (name + ".png"): _score_to_uint8(fused),
            visual_root / "refined_mask" / (name + ".png"): _score_to_uint8(refined),
            visual_root / "binary_mask" / (name + ".png"): (refined >= threshold).astype(np.uint8) * 255,
            visual_root / "coarse_heatmap" / (name + ".png"): coarse_heatmap,
            visual_root / "fused_heatmap" / (name + ".png"): fused_heatmap,
            visual_root / "refined_heatmap" / (name + ".png"): refined_heatmap,
            visual_root / "coarse_overlay" / (name + ".jpg"): coarse_overlay,
            visual_root / "fused_overlay" / (name + ".jpg"): fused_overlay,
            visual_root / "refined_overlay" / (name + ".jpg"): refined_overlay,
            visual_root / "semantic" / (name + ".png"): _semantic_color(semantic),
            visual_root / "protected_candidates" / (name + ".png"): candidate_overlay,
            visual_root / "comparison" / (name + ".jpg"): comparison,
        }
        for destination, value in destinations.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), value):
                raise RuntimeError("Cannot write visualization: {}".format(destination))
    print("Visualizations: {}".format(visual_root))


def refine_folder(args):
    import run_objectomaly_refinement as refinement

    manifest = args.manifest.expanduser().resolve()
    source = read_manifest(manifest)
    if source.get("kind") != "raas-objectomaly-folder-inputs":
        raise ValueError("Expected a custom-folder input manifest")
    output = args.output.expanduser().resolve()
    refine_args = SimpleNamespace(
        manifest=manifest,
        output=output,
        config=args.config,
        sam_checkpoint=args.sam_checkpoint,
        device=args.device,
        phase=args.phase,
    )
    if args.phase == "all":
        # SAM ViT-H and CLIP are intentionally kept out of GPU memory at the
        # same time. The subprocess-free two-pass flow also makes the SAM cache
        # immediately reusable for prompt/fusion experiments.
        refine_args.phase = "masks"
        refinement.execute(refine_args)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        refine_args.phase = "refine"
    refined_manifest = refinement.execute(refine_args)
    if refine_args.phase == "refine":
        render_outputs(refined_manifest, output, args.threshold)
    return refined_manifest


def main():
    args = build_parser().parse_args()
    if args.command == "export":
        export_folder(args)
    else:
        refine_folder(args)


if __name__ == "__main__":
    main()
