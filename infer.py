"""Run the distilled student on an image or a folder of images.

This is the entry point for using the released weights; it needs nothing from
the training pipeline, the teacher, or the cloud -- just torch, transformers
and OpenCV.

    python3 infer.py --input frame.jpg --output out/
    python3 infer.py --input frames/ --output out/ --short-side 736 --half

Writes, per frame:

* ``<name>_score.npy``   raw anomaly probability in [0, 1], full resolution
* ``<name>_heat.png``    the same as a colour map, for looking at
* ``<name>_overlay.png`` heat map blended over the frame

``--short-side`` trades accuracy for speed and is worth choosing deliberately:
1024 is the best measured setting for small road obstacles, 736 roughly halves
the cost, and below ~544 tens-of-pixels obstacles start disappearing.
"""

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from student.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from student.model import Student  # noqa: E402

SIZE_MULTIPLE = 32
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AnomalySegmenter:
    """Frame in (BGR, as OpenCV reads it), anomaly probability out."""

    def __init__(self, checkpoint, device="auto", short_side=1024, half=False):
        self.device = pick_device(device)
        self.short_side = short_side
        self.half = half and self.device.type != "cpu"
        # No network: the checkpoint replaces every weight anyway.
        self.model = Student(pretrained=False).to(self.device).eval()
        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state["model"] if "model" in state else state)
        if self.half:
            self.model.half()

    @torch.no_grad()
    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        scale = self.short_side / min(height, width)
        target_h = int(np.ceil(height * scale / SIZE_MULTIPLE) * SIZE_MULTIPLE)
        target_w = int(np.ceil(width * scale / SIZE_MULTIPLE) * SIZE_MULTIPLE)
        resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        normalised = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(normalised.transpose(2, 0, 1))[None].to(
            self.device, dtype=torch.half if self.half else torch.float
        )

        _, anomaly = self.model(tensor)
        # The head predicts at 1/4 scale; go back to the frame the caller gave us.
        anomaly = F.interpolate(
            anomaly.float(), size=(height, width), mode="bilinear", align_corners=False
        )
        return torch.sigmoid(anomaly)[0, 0].cpu().numpy()


def render(image_bgr, score):
    heat = cv2.applyColorMap((np.clip(score, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return heat, cv2.addWeighted(image_bgr, 0.45, heat, 0.55, 0)


def frames(path: Path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="image file or folder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).resolve().parent / "weights" / "student_final.pt")
    parser.add_argument("--short-side", type=int, default=1024)
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--half", action="store_true", help="fp16 inference (GPU only)")
    parser.add_argument("--no-images", action="store_true", help="write only .npy scores")
    args = parser.parse_args()

    paths = frames(args.input)
    if not paths:
        raise SystemExit("No images found under {}".format(args.input))
    args.output.mkdir(parents=True, exist_ok=True)

    segmenter = AnomalySegmenter(args.checkpoint, args.device, args.short_side, args.half)
    print("device: {} | frames: {} | short side: {}".format(
        segmenter.device, len(paths), args.short_side))

    elapsed = []
    for index, path in enumerate(paths, 1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print("  skipped (unreadable): {}".format(path.name))
            continue
        started = time.perf_counter()
        score = segmenter(image)
        elapsed.append(time.perf_counter() - started)

        np.save(args.output / "{}_score.npy".format(path.stem), score.astype(np.float32))
        if not args.no_images:
            heat, overlay = render(image, score)
            cv2.imwrite(str(args.output / "{}_heat.png".format(path.stem)), heat)
            cv2.imwrite(str(args.output / "{}_overlay.png".format(path.stem)), overlay)
        print("  [{}/{}] {}  max {:.3f}  {:.0f} ms".format(
            index, len(paths), path.name, float(score.max()), elapsed[-1] * 1000))

    if elapsed:
        # The first frame carries lazy CUDA/MPS init; it is not representative.
        steady = elapsed[1:] or elapsed
        print("\nmean {:.0f} ms/frame ({:.1f} FPS) over {} frames".format(
            1000 * np.mean(steady), 1 / np.mean(steady), len(steady)))
    print("results: {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
