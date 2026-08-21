"""Stage 4 — export the student to ONNX and measure what it actually costs.

The number that matters is milliseconds per frame at the resolution the student
will really run at, next to the teacher's measured 2.3 s (AnomalyTrack) and
5.2 s (ObstacleTrack) per frame.
"""

import argparse
import json
import os
from pathlib import Path
import time

import torch

from ...student.model import AnomalyOnly, Student
from ..common import out_root

TEACHER_BASELINE_SECONDS = {"AnomalyTrack": 2.3, "ObstacleTrack": 5.2}


def load_student(checkpoint: Path, device):
    model = Student().to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    return AnomalyOnly(model).to(device).eval()


@torch.no_grad()
def benchmark(model, device, height: int, width: int, warmup: int = 5, runs: int = 20):
    sample = torch.randn(1, 3, height, width, device=device)
    for _ in range(warmup):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()

    started = time.time()
    for _ in range(runs):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - started) / runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=out_root() / "checkpoints" / "student_final.pt")
    parser.add_argument("--output", type=Path, default=out_root() / "export")
    parser.add_argument("--height", type=int, default=int(os.environ.get("EXPORT_HEIGHT", "1024")))
    parser.add_argument("--width", type=int, default=int(os.environ.get("EXPORT_WIDTH", "2048")))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError("Student checkpoint not found: {}".format(args.checkpoint))
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_student(args.checkpoint, device)
    parameters = sum(p.numel() for p in model.parameters())

    onnx_path = args.output / "student_anomaly.onnx"
    torch.onnx.export(
        model,
        torch.randn(1, 3, args.height, args.width, device=device),
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["anomaly"],
        opset_version=args.opset,
        dynamic_axes={"pixel_values": {0: "batch", 2: "height", 3: "width"},
                      "anomaly": {0: "batch", 2: "height", 3: "width"}},
    )
    print("exported {} ({:.1f} MB)".format(onnx_path.name, onnx_path.stat().st_size / 2**20), flush=True)

    seconds = benchmark(model, device, args.height, args.width)
    stats = {
        "parameters": parameters,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "resolution": [args.height, args.width],
        "seconds_per_frame": seconds,
        "ms_per_frame": seconds * 1000.0,
        "fps": 1.0 / seconds,
        "speedup_vs_teacher": {
            name: baseline / seconds for name, baseline in TEACHER_BASELINE_SECONDS.items()
        },
        "onnx": str(onnx_path),
    }
    (args.output / "export_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(
        "student: {:.1f} ms/frame at {}x{} ({:.1f} FPS), {:.1f}M parameters".format(
            stats["ms_per_frame"], args.height, args.width, stats["fps"], parameters / 1e6
        ),
        flush=True,
    )
    for name, factor in stats["speedup_vs_teacher"].items():
        print("  vs teacher on {}: x{:.0f}".format(name, factor), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
