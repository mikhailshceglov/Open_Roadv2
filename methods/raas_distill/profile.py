"""Latency profiler for the distilled student.

Timing the forward pass alone understates what a deployment actually pays: a
frame arrives as a BGR array and has to be resized, normalised, moved to the
device, and the 1/4-scale logit has to be upsampled and squashed back into a
full-resolution probability map. On a 1080p frame those steps are not rounding
error -- on an A100 the forward pass takes 20.7 ms and the surrounding work
another 36.8 -- so they are reported separately and only then added up.

Two other things a naive timing loop gets wrong:

* **Percentiles, not the mean.** A real-time consumer drops frames on the slow
  tail, so p99 decides whether the pipeline holds 30 FPS.
* **Synchronisation.** CUDA and MPS queue work asynchronously; without an
  explicit sync the timer measures how fast Python enqueues, which is a number
  that looks wonderful and means nothing.

    python -m methods.raas_distill.profile --checkpoint methods/raas_distill/weights/student_final.pt
    python -m methods.raas_distill.profile --checkpoint ... --sizes 1088x1920 736x1280
    python -m methods.raas_distill.profile --checkpoint ... --precision fp16 --per-module
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from open_road.device import resolve_device

from .student.model import Student
from .student.preprocess import SIZE_MULTIPLE, normalise


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class Timer:
    """Wall time per named step, synchronising around each one."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.samples: dict[str, list[float]] = {}

    @contextlib.contextmanager
    def step(self, name: str):
        synchronize(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            synchronize(self.device)
            self.samples.setdefault(name, []).append((time.perf_counter() - start) * 1000)

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "mean": float(np.mean(values)),
                "p50": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)),
                "p99": float(np.percentile(values, 99)),
                "std": float(np.std(values)),
            }
            for name, values in self.samples.items()
        }


def profile_pipeline(model, device, height, width, runs, warmup, half):
    """Time the full BGR-frame-in, probability-map-out path, step by step."""
    import cv2

    frame = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
    dtype = torch.half if half else torch.float
    timer = Timer(device)

    for index in range(runs + warmup):
        recording = index >= warmup
        keep = timer.step if recording else (lambda _name: contextlib.nullcontext())

        with keep("1_resize"):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            target_h = int(np.ceil(height / SIZE_MULTIPLE) * SIZE_MULTIPLE)
            target_w = int(np.ceil(width / SIZE_MULTIPLE) * SIZE_MULTIPLE)
            resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        with keep("2_normalise"):
            tensor = torch.from_numpy(np.ascontiguousarray(normalise(resized).transpose(2, 0, 1)))[None]

        with keep("3_to_device"):
            tensor = tensor.to(device, dtype=dtype, non_blocking=True)

        with keep("4_forward"):
            with torch.no_grad():
                _semantic, anomaly = model(tensor)

        with keep("5_upsample"):
            full = F.interpolate(
                anomaly.float(), size=(height, width), mode="bilinear", align_corners=False
            )
            probability = torch.sigmoid(full)

        with keep("6_to_host"):
            probability[0, 0].cpu().numpy()

    return timer.summary()


def per_module(model, device, height, width, half, runs: int = 20):
    """Which submodules dominate: the encoder stages or the decode head."""
    dtype = torch.half if half else torch.float
    batch = torch.randn(1, 3, height, width, device=device, dtype=dtype)
    totals: dict[str, list[float]] = {}
    started: dict[str, float] = {}
    handles = []

    def pre_hook(name):
        def hook(_module, _inputs):
            synchronize(device)
            started[name] = time.perf_counter()
        return hook

    def post_hook(name):
        def hook(_module, _inputs, _output):
            synchronize(device)
            totals.setdefault(name, []).append((time.perf_counter() - started[name]) * 1000)
        return hook

    inner = model.model
    for name, module in (("encoder", inner.segformer), ("decode_head", inner.decode_head)):
        handles.append(module.register_forward_pre_hook(pre_hook(name)))
        handles.append(module.register_forward_hook(post_hook(name)))

    with torch.no_grad():
        for _ in range(5):
            model(batch)
        totals.clear()
        for _ in range(runs):
            model(batch)

    for handle in handles:
        handle.remove()
    return {name: float(np.mean(values)) for name, values in totals.items()}


def parse_size(text: str) -> tuple[int, int]:
    height, width = text.lower().split("x")
    return int(height), int(width)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--sizes", nargs="+", default=["1088x1920", "1024x1824", "736x1280", "544x960"],
        help="HxW inputs to profile",
    )
    parser.add_argument("--precision", choices=("fp32", "fp16", "both"), default="both")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--per-module", action="store_true",
                        help="also split the forward pass into encoder and decode head")
    parser.add_argument("--json", type=Path, help="write the full measurements here")
    args = parser.parse_args()

    device = torch.device(resolve_device(args.device))
    model = Student(pretrained=False).to(device).eval()
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state))
    parameters = sum(p.numel() for p in model.parameters())

    name = torch.cuda.get_device_name(0) if device.type == "cuda" else device.type
    print(f"device: {name} | parameters: {parameters / 1e6:.2f}M | torch {torch.__version__}")

    # fp16 on CPU is emulated and meaningless to time.
    if device.type == "cpu":
        precisions = ["fp32"]
    elif args.precision == "both":
        precisions = ["fp32", "fp16"]
    else:
        precisions = [args.precision]

    report = {"device": name, "torch": torch.__version__, "parameters": parameters, "runs": {}}
    for size in args.sizes:
        height, width = parse_size(size)
        for precision in precisions:
            half = precision == "fp16"
            model.half() if half else model.float()
            steps = profile_pipeline(model, device, height, width, args.runs, args.warmup, half)
            total = sum(entry["mean"] for entry in steps.values())
            forward = steps["4_forward"]["mean"]
            tail = sum(entry["p99"] for entry in steps.values())

            print(f"\n{height}x{width}  {precision}")
            print("  {:<14}{:>9}{:>9}{:>9}".format("step", "mean", "p90", "p99"))
            for step_name in sorted(steps):
                entry = steps[step_name]
                print("  {:<14}{:>9.2f}{:>9.2f}{:>9.2f}".format(
                    step_name[2:], entry["mean"], entry["p90"], entry["p99"]))
            print("  {:<14}{:>9.2f}{:>18.2f}".format("TOTAL", total, tail))
            print(f"  forward is {100 * forward / total:.0f}% of the time | "
                  f"{1000 / total:.1f} FPS at the mean, {1000 / tail:.1f} at p99")

            entry = {"steps": steps, "total_mean_ms": total, "total_p99_ms": tail,
                     "fps_mean": 1000 / total, "fps_p99": 1000 / tail}
            if args.per_module:
                entry["per_module_ms"] = per_module(model, device, height, width, half)
                print("  per module: " + ", ".join(
                    f"{key} {value:.2f} ms" for key, value in entry["per_module_ms"].items()))
            report["runs"][f"{height}x{width}_{precision}"] = entry

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nfull measurements: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
