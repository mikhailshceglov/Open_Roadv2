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
    write_manifest,
)


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


def _make_generator(args, config, dependencies):
    if args.sam_checkpoint is None:
        raise ValueError("--sam-checkpoint is required for phase {}".format(args.phase))
    checkpoint = args.sam_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("SAM checkpoint not found: {}".format(checkpoint))
    sam = config["sam"]
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
        raw = generator.generate(image)
        bundle = dependencies["postprocess"](raw, **config["postprocess"])
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
    if phase in ("all", "refine"):
        oasc_cfg = config["oasc"]
        calibrated = dependencies["apply_oasc"](
            coarse, bundle, oasc_cfg["variant"], **oasc_cfg.get("params", {})
        )
        mbp_cfg = config["mbp"]
        mbp_params = dict(mbp_cfg.get("params", {}))
        if "gaussian_ksize" in mbp_params:
            mbp_params["gaussian_ksize"] = tuple(mbp_params["gaussian_ksize"])
        refined = dependencies["apply_mbp"](
            calibrated,
            mbp_cfg["variant"],
            image_bgr=image,
            bundle=bundle,
            **mbp_params
        )
        refined = validate_anomaly_map(refined, target_hw)
        refined_rel = Path("refined") / str(entry["source_model"]) / dataset / (fid + ".npy")
        refined_path = output / refined_rel
        refined_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(refined_path), refined, allow_pickle=False)
        result["refined_map"] = str(refined_rel)
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
    if generator is None and args.phase in ("all", "masks"):
        generator = _make_generator(args, config, dependencies)

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
            "entries": entries,
        },
    )
    print("Manifest: {}".format(manifest_path))
    return manifest_path


def main() -> None:
    execute(build_parser().parse_args())


if __name__ == "__main__":
    main()
