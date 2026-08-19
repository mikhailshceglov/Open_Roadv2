"""Generate SAM masks and refine cached RAAS maps with official Objectomaly."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MASKOMALY_DIR = SCRIPT_DIR.parent
RAAS_DIR = MASKOMALY_DIR.parent
OBJECTOMALY_DIR = RAAS_DIR / "third_party" / "Objectomaly"
OBJECTOMALY_COMMIT = "66d2ad2a1b02d79389f4265d9d1d99ab6412324f"
DEFAULT_CONFIG = MASKOMALY_DIR / "configs" / "objectomaly_smiyc.json"
sys.path.insert(0, str(MASKOMALY_DIR / "maskomaly"))

from objectomaly_cache import (
    read_manifest,
    resolve_cached_path,
    validate_anomaly_map,
    validate_bgr_image,
    validate_frame_id,
    validate_semantic_map,
    write_manifest,
)
from global_fusion import apply_global_fusion, build_clip_validator
from inference_timing import runtime_environment, timed_call, write_timing_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Official Objectomaly refinement over cached RAAS predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase", choices=("all", "masks", "refine"), default="all")
    return parser


def load_config(path: Path) -> Mapping[str, object]:
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    for section in ("sam", "postprocess", "oasc", "mbp"):
        if not isinstance(config.get(section), dict):
            raise ValueError("Missing config section: {}".format(section))
    return config


def configure_objectomaly_imports() -> None:
    package = OBJECTOMALY_DIR / "objectomaly" / "__init__.py"
    if not package.is_file():
        raise FileNotFoundError(
            "Objectomaly submodule is missing. Run: git submodule update --init --recursive"
        )
    value = str(OBJECTOMALY_DIR)
    if value not in sys.path:
        sys.path.insert(0, value)


def verify_objectomaly_commit() -> None:
    gitlink = RAAS_DIR / ".git" / "modules" / "third_party" / "Objectomaly" / "HEAD"
    # Worktree archives may not include nested git metadata. Source provenance
    # remains recorded in both .gitmodules/gitlink and the manifests.
    if not gitlink.exists() and not (OBJECTOMALY_DIR / ".git").exists():
        return
    import subprocess

    actual = subprocess.check_output(
        ["git", "-C", str(OBJECTOMALY_DIR), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != OBJECTOMALY_COMMIT:
        raise RuntimeError(
            "Objectomaly commit mismatch: expected {}, got {}".format(
                OBJECTOMALY_COMMIT, actual
            )
        )


def load_dependencies():
    configure_objectomaly_imports()
    from objectomaly.masks.cache import has_bundle, load_bundle, save_bundle
    from objectomaly.masks.postprocess import postprocess
    from objectomaly.masks.sam import SAMMaskGenerator
    from objectomaly.refinement.mbp import apply_mbp
    from objectomaly.refinement.oasc import apply_oasc

    return {
        "SAMMaskGenerator": SAMMaskGenerator,
        "postprocess": postprocess,
        "save_bundle": save_bundle,
        "load_bundle": load_bundle,
        "has_bundle": has_bundle,
        "apply_oasc": apply_oasc,
        "apply_mbp": apply_mbp,
    }


def install_sam_cpu_nms_workaround() -> None:
    """Avoid torchvision 0.16 mixed-device indexing for >4000 SAM boxes.

    A 64x64 point grid can enter torchvision's vanilla batched-NMS branch.
    On the pinned torch/torchvision stack that branch may create a CPU keep
    mask while retaining CUDA indices. Running only NMS on CPU is deterministic
    and leaves SAM encoding/mask decoding on the requested device.
    """
    from segment_anything import automatic_mask_generator as sam_amg

    current = sam_amg.batched_nms
    if getattr(current, "_raas_cpu_nms", False):
        return

    def cpu_batched_nms(boxes, scores, idxs, iou_threshold):
        device = boxes.device
        keep = current(
            boxes.detach().cpu(),
            scores.detach().cpu(),
            idxs.detach().cpu(),
            iou_threshold,
        )
        return keep.to(device=device)

    cpu_batched_nms._raas_cpu_nms = True
    sam_amg.batched_nms = cpu_batched_nms


def _make_generator(args, config, dependencies):
    if args.sam_checkpoint is None:
        raise ValueError("--sam-checkpoint is required for phase {}".format(args.phase))
    checkpoint = args.sam_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("SAM checkpoint not found: {}".format(checkpoint))
    sam = config["sam"]
    if bool(sam.get("force_cpu_nms", False)):
        install_sam_cpu_nms_workaround()
    expected_md5 = str(sam.get("checkpoint_md5", "")).lower()
    if expected_md5:
        digest = hashlib.md5()
        with checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual_md5 = digest.hexdigest()
        if actual_md5 != expected_md5:
            raise RuntimeError(
                "SAM checkpoint MD5 mismatch: expected {}, got {}".format(
                    expected_md5, actual_md5
                )
            )
    return dependencies["SAMMaskGenerator"](
        checkpoint=str(checkpoint),
        variant=sam["variant"],
        device=args.device,
        points_per_side=int(sam["points_per_side"]),
        pred_iou_thresh=float(sam["pred_iou_thresh"]),
        stability_score_thresh=float(sam["stability_score_thresh"]),
    )


def refine_entry(
    entry,
    input_manifest: Path,
    output: Path,
    config,
    dependencies,
    phase: str,
    generator=None,
    clip_validator=None,
):
    fid = validate_frame_id(entry["fid"])
    dataset = validate_frame_id(str(entry["dataset"]))
    target_hw = (int(entry["height"]), int(entry["width"]))
    image = validate_bgr_image(
        np.load(str(resolve_cached_path(input_manifest, entry["image_bgr"])), allow_pickle=False),
        target_hw,
    )
    coarse = validate_anomaly_map(
        np.load(str(resolve_cached_path(input_manifest, entry["coarse_map"])), allow_pickle=False),
        target_hw,
    )

    cache_root = output / "mask_cache"
    generator_tag = "sam_{}".format(config["sam"]["variant"])
    has_bundle = dependencies["has_bundle"](
        dataset, generator_tag, fid, cache_root=str(cache_root)
    )
    if phase in ("all", "masks"):
        if generator is None:
            raise RuntimeError("SAM generator is not initialized")
        raw, sam_generate_s = timed_call(lambda: generator.generate(image))
        bundle, sam_postprocess_s = timed_call(
            lambda: dependencies["postprocess"](raw, **config["postprocess"])
        )
        bundle.timings = dict(getattr(bundle, "timings", {}))
        bundle.timings["sam_generate_s"] = sam_generate_s
        bundle.timings["sam_postprocess_s"] = sam_postprocess_s
        dependencies["save_bundle"](
            bundle,
            dataset,
            generator_tag,
            fid,
            cache_root=str(cache_root),
        )
    elif not has_bundle:
        raise FileNotFoundError(
            "Missing cached masks for {} / {}; run --phase masks first".format(
                dataset, fid
            )
        )
    else:
        bundle = dependencies["load_bundle"](
            dataset, generator_tag, fid, cache_root=str(cache_root)
        )

    result = dict(entry)
    result["n_masks"] = int(bundle.n)
    timings = dict(entry.get("timings", {}))
    for key in ("sam_generate_s", "sam_postprocess_s"):
        if key in getattr(bundle, "timings", {}):
            timings[key] = float(bundle.timings[key])
    if phase in ("all", "refine"):
        oasc_cfg = config["oasc"]
        calibrated, timings["oasc_s"] = timed_call(
            lambda: dependencies["apply_oasc"](
                coarse, bundle, oasc_cfg["variant"], **oasc_cfg.get("params", {})
            )
        )
        fusion_cfg = config.get("global_fusion", {})
        if isinstance(fusion_cfg, dict) and fusion_cfg.get("enabled", False):
            semantic_rel = entry.get("semantic_map")
            if not semantic_rel:
                raise ValueError(
                    "Global fusion requires semantic_map in every input entry; "
                    "rerun the RAAS export stage"
                )
            semantic = validate_semantic_map(
                np.load(
                    str(resolve_cached_path(input_manifest, semantic_rel)),
                    allow_pickle=False,
                ),
                target_hw,
            )
            fusion_result, timings["global_fusion_s"] = timed_call(
                lambda: apply_global_fusion(
                    calibrated,
                    semantic,
                    image,
                    bundle,
                    fusion_cfg,
                    clip_validator=clip_validator,
                )
            )
            calibrated, protected, diagnostics = fusion_result
            fused_rel = Path("fused") / str(entry["source_model"]) / dataset / (fid + ".npy")
            protected_rel = (
                Path("protected_candidates")
                / str(entry["source_model"])
                / dataset
                / (fid + ".npy")
            )
            for rel, value in (
                (fused_rel, calibrated),
                (protected_rel, protected.astype(np.uint8)),
            ):
                path = output / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(path), value, allow_pickle=False)
            result["fused_map"] = str(fused_rel)
            result["protected_candidates"] = str(protected_rel)
            result["global_fusion"] = diagnostics
        else:
            timings["global_fusion_s"] = 0.0
        mbp_cfg = config["mbp"]
        mbp_params = dict(mbp_cfg.get("params", {}))
        if "gaussian_ksize" in mbp_params:
            mbp_params["gaussian_ksize"] = tuple(mbp_params["gaussian_ksize"])
        refined, timings["mbp_s"] = timed_call(
            lambda: dependencies["apply_mbp"](
                calibrated,
                mbp_cfg["variant"],
                image_bgr=image,
                bundle=bundle,
                **mbp_params
            )
        )
        refined = validate_anomaly_map(refined, target_hw)
        refined_rel = Path("refined") / str(entry["source_model"]) / dataset / (fid + ".npy")
        refined_path = output / refined_rel
        refined_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(refined_path), refined, allow_pickle=False)
        result["refined_map"] = str(refined_rel)
        timings["objectomaly_refine_s"] = sum(
            timings.get(key, 0.0)
            for key in ("oasc_s", "global_fusion_s", "mbp_s")
        )
        timings["end_to_end_compute_s"] = sum(
            timings.get(key, 0.0)
            for key in (
                "raas_inference_s",
                "raas_postprocess_s",
                "sam_generate_s",
                "sam_postprocess_s",
                "oasc_s",
                "global_fusion_s",
                "mbp_s",
            )
        )
    result["timings"] = timings
    return result


def execute(args, dependencies=None, generator=None):
    args.manifest = args.manifest.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    source = read_manifest(args.manifest)
    source_kind = source.get("kind")
    supported_kinds = {
        "raas-objectomaly-inputs": "raas-objectomaly-refined",
        "raas-objectomaly-folder-inputs": "raas-objectomaly-folder-refined",
    }
    if source_kind not in supported_kinds:
        raise ValueError("Unexpected input manifest kind: {}".format(source.get("kind")))
    source_model = validate_frame_id(str(source["source_model"]))
    verify_objectomaly_commit()
    config = load_config(args.config)
    mask_cache_config_path = args.output / "mask-cache-config.json"
    mask_cache_config = {
        "objectomaly_commit": OBJECTOMALY_COMMIT,
        "sam": config["sam"],
        "postprocess": config["postprocess"],
    }
    if args.phase in ("all", "masks"):
        mask_cache_config_path.parent.mkdir(parents=True, exist_ok=True)
        with mask_cache_config_path.open("w", encoding="utf-8") as handle:
            json.dump(mask_cache_config, handle, indent=2, sort_keys=True)
            handle.write("\n")
    else:
        if not mask_cache_config_path.is_file():
            raise FileNotFoundError(
                "Missing {}; run --phase masks first".format(mask_cache_config_path)
            )
        with mask_cache_config_path.open(encoding="utf-8") as handle:
            cached_config = json.load(handle)
        if cached_config != mask_cache_config:
            raise RuntimeError(
                "Mask cache config differs from the current SAM/postprocess config"
            )
    dependencies = dependencies or load_dependencies()
    setup_timings = dict(source.get("setup_timings_s", {}))
    if generator is None and args.phase in ("all", "masks"):
        generator, setup_timings["sam_generator_setup_s"] = timed_call(
            lambda: _make_generator(args, config, dependencies)
        )
    clip_validator = None
    fusion_cfg = config.get("global_fusion", {})
    if (
        args.phase in ("all", "refine")
        and isinstance(fusion_cfg, dict)
        and fusion_cfg.get("enabled", False)
    ):
        clip_validator, setup_timings["clip_model_load_s"] = timed_call(
            lambda: build_clip_validator(fusion_cfg, args.device)
        )

    entries = []
    for source_entry in source["entries"]:
        entry = dict(source_entry)
        entry["source_model"] = source_model
        entries.append(
            refine_entry(
                entry,
                args.manifest,
                args.output,
                config,
                dependencies,
                args.phase,
                generator,
                clip_validator,
            )
        )
        print("{} / {}: masks={}".format(entry["dataset"], entry["fid"], entries[-1]["n_masks"]))

    manifest_path = args.output / ("manifest-objectomaly-{}.json".format(source_model))
    write_manifest(
        manifest_path,
        {
            "kind": supported_kinds[source_kind],
            "source_model": source_model,
            "input_manifest": str(args.manifest),
            "objectomaly_commit": OBJECTOMALY_COMMIT,
            "phase": args.phase,
            "config": config,
            "setup_timings_s": setup_timings,
            "runtime": runtime_environment(),
            "entries": entries,
        },
    )
    if args.phase in ("all", "refine"):
        csv_path, json_path = write_timing_report(
            entries,
            args.output,
            source_model,
            setup_timings=setup_timings,
            runtime=runtime_environment(),
        )
        print("Timings: {} / {}".format(csv_path, json_path))
    print("Manifest: {}".format(manifest_path))
    return manifest_path


def main() -> None:
    execute(build_parser().parse_args())


if __name__ == "__main__":
    main()
