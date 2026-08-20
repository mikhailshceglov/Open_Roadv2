"""Stage 2 — train the student on the teacher's stored predictions.

The teacher is never loaded here: everything it knows is already on disk from
the label stage.  That is what makes the run cheap, and it is also why the
stage survives being killed -- it checkpoints every ``--checkpoint-every``
iterations and resumes from the latest one.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import data_root, out_root  # noqa: E402
from student.dataset import DistillationDataset  # noqa: E402
from student.losses import distillation_loss  # noqa: E402
from student.model import Student  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=data_root() / "frames")
    parser.add_argument("--targets", type=Path, default=out_root() / "targets")
    parser.add_argument("--checkpoints", type=Path, default=out_root() / "checkpoints")
    parser.add_argument("--iters", type=int, default=int(os.environ.get("ITERS", "10000")))
    parser.add_argument("--crop", type=int, default=int(os.environ.get("CROP", "768")))
    parser.add_argument("--batch", type=int, default=int(os.environ.get("BATCH", "16")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("LR", "6e-5")))
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "8")))
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def latest_checkpoint(directory: Path):
    if not directory.is_dir():
        return None
    # Only the numbered checkpoints carry optimizer/scheduler state.
    # student_final.pt holds weights alone and sorts after them alphabetically,
    # so globbing "student_*.pt" would resume from a file with no optimizer.
    checkpoints = sorted(directory.glob("student_[0-9]*.pt"))
    return checkpoints[-1] if checkpoints else None


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 needs no loss scaling and A100 runs it natively; fp16 elsewhere.
    autocast_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    dataset = DistillationDataset(args.corpus, args.targets, crop=args.crop)
    print("training on {} labelled frames".format(len(dataset)), flush=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.workers > 0,
    )

    model = Student().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (1.0 - step / args.iters) ** 0.9
    )
    # torch 2.1 (the version detectron2 0.6 is happy with) still spells this
    # torch.cuda.amp.GradScaler; 2.4+ moved it to torch.amp.
    needs_scaling = autocast_dtype == torch.float16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=needs_scaling)

    args.checkpoints.mkdir(parents=True, exist_ok=True)
    start_iteration = 0
    resume = latest_checkpoint(args.checkpoints)
    if resume is not None:
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_iteration = state["iteration"]
        print("resumed from {} at iteration {}".format(resume.name, start_iteration), flush=True)

    model.train()
    stream = infinite(loader)
    started = time.time()
    running = {}

    for iteration in range(start_iteration, args.iters):
        batch = next(stream)
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        anomaly = batch["anomaly"].to(device, non_blocking=True)
        semantic = batch["semantic"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
            semantic_logits, anomaly_logits = model(pixel_values)
            terms = distillation_loss(semantic_logits, anomaly_logits, semantic, anomaly)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(terms["total"]).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        for key, value in terms.items():
            running[key] = running.get(key, 0.0) + float(value.detach())

        step = iteration + 1
        if step % args.log_every == 0:
            elapsed = time.time() - started
            done = step - start_iteration
            message = "  ".join(
                "{}={:.4f}".format(key, value / args.log_every) for key, value in sorted(running.items())
            )
            print(
                "[{}/{}] {}  lr={:.2e}  {:.2f}s/it".format(
                    step, args.iters, message, scheduler.get_last_lr()[0], elapsed / done
                ),
                flush=True,
            )
            running = {}

        if step % args.checkpoint_every == 0 or step == args.iters:
            path = args.checkpoints / "student_{:06d}.pt".format(step)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "iteration": step,
                    "args": vars(args) | {"corpus": str(args.corpus), "targets": str(args.targets),
                                          "checkpoints": str(args.checkpoints)},
                },
                path,
            )
            torch.save({"model": model.state_dict()}, args.checkpoints / "student_final.pt")
            print("checkpoint: {}".format(path.name), flush=True)

    stats = {
        "iterations": args.iters,
        "seconds_total": time.time() - started,
        "frames": len(dataset),
        "crop": args.crop,
        "batch": args.batch,
    }
    (args.checkpoints / "train_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print("train stage done: {}".format(json.dumps(stats)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
