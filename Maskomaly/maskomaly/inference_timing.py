"""CUDA-aware per-frame timing reports for cache-first inference."""

import csv
import json
import platform
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import numpy as np


TIMING_KEYS = (
    "raas_inference_s",
    "raas_postprocess_s",
    "sam_generate_s",
    "sam_postprocess_s",
    "oasc_s",
    "global_fusion_s",
    "mbp_s",
    "objectomaly_refine_s",
    "end_to_end_compute_s",
)


def synchronize_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def timed_call(function: Callable):
    synchronize_cuda()
    started = time.perf_counter()
    result = function()
    synchronize_cuda()
    return result, float(time.perf_counter() - started)


def runtime_environment() -> Mapping[str, object]:
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        result.update(
            {
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_runtime": getattr(torch.version, "cuda", None),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        result["torch"] = None
    try:
        import torchvision

        result["torchvision"] = torchvision.__version__
    except (ImportError, RuntimeError):
        result["torchvision"] = None
    return result


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return None
    return {
        "mean_ms": float(values.mean() * 1000.0),
        "p50_ms": float(np.percentile(values, 50) * 1000.0),
        "p95_ms": float(np.percentile(values, 95) * 1000.0),
        "max_ms": float(values.max() * 1000.0),
        "fps_from_mean": float(1.0 / values.mean()) if values.mean() > 0 else None,
    }


def write_timing_report(
    entries: Sequence[Mapping[str, object]],
    output: Path,
    source_model: str,
    setup_timings=None,
    runtime=None,
):
    """Write per-frame CSV and all/steady-state aggregate JSON.

    Steady state excludes the first frame, whose SAM timing includes lazy model
    construction and CUDA warm-up. Model setup is reported separately.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / ("timings-{}.csv".format(source_model))
    json_path = output / ("timings-summary-{}.json".format(source_model))
    rows = []
    for entry in entries:
        timings = dict(entry.get("timings", {}))
        row = {
            "dataset": str(entry.get("dataset", "")),
            "fid": str(entry.get("fid", "")),
        }
        for key in TIMING_KEYS:
            value = timings.get(key)
            row[key.replace("_s", "_ms")] = (
                None if value is None else float(value) * 1000.0
            )
        rows.append(row)

    fieldnames = ["dataset", "fid"] + [
        key.replace("_s", "_ms") for key in TIMING_KEYS
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    all_entries = list(entries)
    steady_entries = all_entries[1:] if len(all_entries) > 1 else all_entries

    def summarize(selected):
        stages = {}
        for key in TIMING_KEYS:
            values = [
                float(entry.get("timings", {}).get(key))
                for entry in selected
                if entry.get("timings", {}).get(key) is not None
            ]
            stages[key] = _stats(values)
        return {"frames": len(selected), "stages": stages}

    payload = {
        "source_model": source_model,
        "units": "milliseconds",
        "definition": (
            "CUDA-synchronized compute latency; dataset/image/cache IO and metric "
            "calculation are excluded. end_to_end_compute_s is the sequential sum "
            "of RAAS, SAM, OASC, global fusion/CLIP and MBP stages."
        ),
        "setup_timings_s": dict(setup_timings or {}),
        "runtime": dict(runtime or runtime_environment()),
        "all_frames": summarize(all_entries),
        "steady_state": summarize(steady_entries),
        "by_dataset": {
            dataset: summarize(
                [entry for entry in all_entries if entry.get("dataset") == dataset]
            )
            for dataset in sorted(
                {str(entry.get("dataset", "")) for entry in all_entries}
            )
        },
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")

    end_to_end = payload["steady_state"]["stages"]["end_to_end_compute_s"]
    if end_to_end is not None:
        print(
            "Timing {} (steady state, {} frames): mean={:.1f} ms, "
            "p95={:.1f} ms, {:.2f} FPS from mean".format(
                source_model,
                payload["steady_state"]["frames"],
                end_to_end["mean_ms"],
                end_to_end["p95_ms"],
                end_to_end["fps_from_mean"],
            )
        )
    return csv_path, json_path
