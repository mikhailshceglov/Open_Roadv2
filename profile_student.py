"""Latency profiler for the distilled student.

Existing benchmarks in this project timed the forward pass alone, which
understates what a deployment actually pays: a frame arrives as a BGR array and
has to be resized, normalised, moved to the device, and the 1/4-scale logit has
to be upsampled and squashed back to a full-resolution probability map.  On a
1080p frame those steps are not rounding error, so this profiler reports them
separately and only then adds them up.

Two other things it does that a naive timing loop does not:

* **Percentiles, not just the mean.**  A real-time consumer drops frames on the
  slow tail, so p99 decides whether the pipeline holds 30 FPS, not the average.
* **Correct synchronisation.**  CUDA and MPS queue work asynchronously; without
  an explicit sync the timer measures how fast Python can enqueue, which is a
  number that looks wonderful and means nothing.

Usage::

    python3 distill/profile_student.py --checkpoint results/student/student_final.pt
    python3 distill/profile_student.py --checkpoint ... --sizes 1088x1920 736x1280
    python3 distill/profile_student.py --checkpoint ... --precision fp16 --per-module
"""

import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# Importable both from the repository and from inside the job container,
# where this file is copied next to the job rather than into the package.
for _base in (Path(__file__).resolve().parent, Path("/opt/raas/distill"), Path("/opt/distill")):
    if (_base / "student").is_dir():
        sys.path.insert(0, str(_base))
        break

from student.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from student.model import Student  # noqa: E402

SIZE_MULTIPLE = 32


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class Timer:
    """Accumulates wall time per named step, synchronising around each one."""

    def __init__(self, device: torch.device):
        self.device = device
        self.samples = {}

    @contextlib.contextmanager
    def step(self, name: str):
        synchronize(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            synchronize(self.device)
            self.samples.setdefault(name, []).append((time.perf_counter() - start) * 1000)

    def summary(self):
        out = {}
        for name, values in self.samples.items():
            arr = np.asarray(values)
            out[name] = {
                "mean": float(arr.mean()),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "p99": float(np.percentile(arr, 99)),
                "std": float(arr.std()),
            }
        return out


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
            normalised = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            tensor = torch.from_numpy(normalised.transpose(2, 0, 1))[None]

        with keep("3_to_device"):
            tensor = tensor.to(device, dtype=dtype, non_blocking=True)

        with keep("4_forward"):
            with torch.no_grad():
                _, anomaly = model(tensor)

        with keep("5_upsample"):
            full = F.interpolate(
                anomaly.float(), size=(height, width), mode="bilinear", align_corners=False
            )
            probability = torch.sigmoid(full)

        with keep("6_to_host"):
            probability[0, 0].cpu().numpy()

    return timer.summary()


def per_module(model, device, height, width, half, runs=20):
    """Which submodules dominate: encoder stages versus the decode head."""
    dtype = torch.half if half else torch.float
    x = torch.randn(1, 3, height, width, device=device, dtype=dtype)
    totals = {}
    handles = []
    state = {}

    def pre_hook(name):
        def hook(_module, _inputs):
            synchronize(device)
            state[name] = time.perf_counter()
        return hook

    def post_hook(name):
        def hook(_module, _inputs, _output):
            synchronize(device)
            totals.setdefault(name, []).append((time.perf_counter() - state[name]) * 1000)
        return hook

    inner = model.model
    watched = [("encoder", inner.segformer), ("decode_head", inner.decode_head)]
    for name, module in watched:
        handles.append(module.register_forward_pre_hook(pre_hook(name)))
        handles.append(module.register_forward_hook(post_hook(name)))

    with torch.no_grad():
        for _ in range(5):
            model(x)
        totals.clear()
        for _ in range(runs):
            model(x)

    for handle in handles:
        handle.remove()
    return {name: float(np.mean(values)) for name, values in totals.items()}


def parse_size(text: str):
    height, width = text.lower().split("x")
    return int(height), int(width)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", default=["1088x1920", "1024x1824", "736x1280", "544x960"],
                        help="HxW inputs to profile")
    parser.add_argument("--precision", choices=("fp32", "fp16", "both"), default="both")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--per-module", action="store_true",
                        help="also break the forward pass into encoder and decode head")
    parser.add_argument("--json", type=Path, help="write the full measurements here")
    args = parser.parse_args()

    device = pick_device(args.device)
    model = Student(pretrained=False).to(device).eval()
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    parameters = sum(p.numel() for p in model.parameters())

    name = torch.cuda.get_device_name(0) if device.type == "cuda" else device.type
    print("устройство: {} | параметров: {:.2f}M | torch {}".format(name, parameters / 1e6, torch.__version__))
    # fp16 on CPU is emulated and meaningless to time.
    precisions = ["fp32"] if device.type == "cpu" else (
        ["fp32", "fp16"] if args.precision == "both" else [args.precision]
    )

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

            print("\n{}x{}  {}".format(height, width, precision))
            print("  {:<14}{:>9}{:>9}{:>9}".format("шаг", "среднее", "p90", "p99"))
            for step_name in sorted(steps):
                entry = steps[step_name]
                print("  {:<14}{:>9.2f}{:>9.2f}{:>9.2f}".format(
                    step_name[2:], entry["mean"], entry["p90"], entry["p99"]))
            print("  {:<14}{:>9.2f}{:>18.2f}".format("ИТОГО", total, tail))
            print("  forward — {:.0f}% полного времени | {:.1f} FPS по среднему, {:.1f} по p99".format(
                100 * forward / total, 1000 / total, 1000 / tail))

            entry = {"steps": steps, "total_mean_ms": total, "total_p99_ms": tail,
                     "fps_mean": 1000 / total, "fps_p99": 1000 / tail}
            if args.per_module:
                entry["per_module_ms"] = per_module(model, device, height, width, half)
                print("  по модулям: " + ", ".join(
                    "{} {:.2f} мс".format(k, v) for k, v in entry["per_module_ms"].items()))
            report["runs"]["{}x{}_{}".format(height, width, precision)] = entry

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("\nполные измерения: {}".format(args.json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
