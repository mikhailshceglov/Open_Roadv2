# RAAS-Distill

A real-time student distilled from **RAAS** (Mask2Former Swin-L + CLIP) for road
anomaly segmentation on the [SegmentMeIfYouCan](https://segmentmeifyoucan.com)
benchmark.

The teacher runs at **0.2–0.4 FPS**. The student is **3.72M parameters, 14 MB**,
and matches or beats it on both tracks.

| model | Anomaly AUPR / FPR@95 | Obstacle AUPR / FPR@95 | size |
|---|---|---|---|
| teacher `maskomaly` | 94.09 / 3.40 | 88.62 / 2.81 | ~215M params |
| teacher `maskomaly_id` (distilled from) | 93.18 / 3.42 | 91.21 / 0.42 | ~215M params |
| **student, 1024px** | 94.02 / 7.14 | **97.52 / 0.07** | **3.72M params** |
| **student, 544px** | **95.83 / 4.76** | 92.05 / 0.79 | **3.72M params** |

The student beats its own teacher on ObstacleTrack by 6.3 AUPR. This is not
unusual for distillation: the teacher's decision is assembled from hard-coded
thresholds over Mask2Former queries, while the student learns a dense map over
572 frames and smooths away the individual failures of those rules — most
visibly on the small obstacles the teacher handled worst.

> **On comparability.** These numbers come from this project's own
> `get_scores` (pooled pixel metrics). They are *not* the official SMIYC
> evaluator, which treats void pixels differently and also reports component
> metrics (sIoU / PPV / F1). Treat the table as internally consistent, not as
> an official benchmark submission.

## Quick start

```bash
pip install -r requirements.txt
python3 infer.py --input frame.jpg --output out/
```

Per frame you get a `.npy` probability map at full resolution, plus a heat map
and an overlay to look at. Point `--input` at a folder to batch it.

```bash
python3 infer.py --input frames/ --output out/ --short-side 736 --half
```

**`transformers==4.44.2` is a hard pin.** SegFormer's internal module names
changed in 5.x (`encoder.block` → `stages.blocks`), and the released weights
will not load on the newer layout.

**No network needed.** The architecture is built from a vendored config
(`student/segformer_b0_cityscapes.json`), so nothing is downloaded at run time —
the pretrained backbone would only be fetched to be overwritten by the
checkpoint. Verified with `HF_HUB_OFFLINE=1` against an empty cache.

## Choosing `--short-side`

Resolution is the one knob that matters, and the two tracks pull in opposite
directions. Anomalies on AnomalyTrack are large, so downscaling removes noise
and *helps*. Obstacles are tens of pixels across, so downscaling destroys them.

| short side | Anomaly AUPR | Obstacle AUPR | A100 fp16, full pipeline |
|---|---|---|---|
| 1024 | 94.02 | **97.52** | 19.4 FPS |
| 736 | 95.18 | 95.67 | **36.6 FPS** (p99 32.6) |
| 544 | **95.83** | 92.05 | 54.3 FPS |
| 384 | 95.02 | 83.56 | — |

Pick 1024 if small obstacles matter, 736 for real time, and do not go below 544.

## Latency

Measured end to end — resize, normalise, host→device, forward, upsample,
device→host — not forward alone. On an A100-SXM4-80GB the forward pass at 1080p
fp16 takes 20.7 ms but the surrounding work takes **36.8 ms more**, so the honest
figure is 17.4 FPS rather than the 48 FPS a forward-only benchmark reports.

```bash
python3 profile_student.py --checkpoint weights/student_final.pt --per-module
```

The profiler reports p50/p90/p99 rather than the mean, because a real-time
consumer drops frames on the tail; it synchronises the device around every step,
without which the timer measures how fast Python enqueues work rather than how
fast anything computes.

Most of the overhead is `cv2.resize` and the numpy normalise on the CPU, which
fp16 does nothing for. Moving both onto the device is the obvious next win and
would put 1080p comfortably past 30 FPS.

## Architecture

`student/model.py` — SegFormer-B0 with the classifier widened from 19 to 20
channels. Channels 0–18 keep their pretrained Cityscapes weights; channel 19 is
the anomaly logit, initialised with bias −4.0 so the model starts by predicting
"not anomalous" everywhere. Anomalous pixels are well under 1% of the corpus and
a zero-init head wastes its early capacity unlearning a 50/50 prior.

Nothing else is bolted on: the anomaly score the teacher produces is a function
of the same semantics the first 19 channels predict, so the encoder is shared.

`student/losses.py` — soft BCE against the teacher's map plus Dice (mandatory at
this class imbalance), and KL on the 19 semantic channels as a regulariser.

## Training pipeline

Distillation is offline: the teacher's post-processing is numpy and per-component
CLIP, so it is not differentiable and cannot sit in the graph. The teacher labels
a corpus once, then the student trains on those maps.

| stage | what it does |
|---|---|
| `stages/label.py` | teacher → float16 anomaly maps + quarter-res semantic logits |
| `stages/train.py` | student on those maps |
| `stages/evaluate.py` | SMIYC metrics + teacher fidelity on unlabelled frames |
| `stages/export.py` | ONNX + latency |

`entrypoint.sh` runs them in order, driven by `STAGES`, `ITERS`, `CROP`, `BATCH`.
Each stage is idempotent: it writes a marker and is skipped on re-run, so a job
that dies in training does not re-pay for labelling.

Reproducing the labelling stage needs the [RAAS](https://github.com/jan-ackermann/Maskomaly)
teacher and its Mask2Former Swin-L checkpoint; apply `patches/teacher.patch`
first. It contains two fixes:

* the all-pairs query overlap loop in `maskomaly/model_*.py` is O(N²) over
  full-resolution masks. A pixel is zeroed exactly when two or more masks exceed
  0.1 there, so the whole double loop collapses to one sum — bit-identical
  output, ~33× faster, and the difference between a labelling run that is
  GPU-bound and one that is not;
* `MSDeformAttn` raises unconditionally when its CUDA kernel is absent, which
  makes the model unimportable on any machine without it. It becomes a warning,
  falling back to the PyTorch implementation.

### Running it on Yandex DataSphere

`config.yaml` is a DataSphere job spec; fill in `<REGISTRY_ID>` and
`<YOUR_BUCKET>`, build and push the image, then:

```bash
datasphere project job execute -p <PROJECT_ID> -c config.yaml
```

The container fetches its own inputs from Object Storage rather than using a
DataSphere S3 mount: mounts need a connector that can only be created in the
project UI, and there is no API for it. Credentials come from
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in the job vars and are never
written to this repository.

Build for `linux/amd64`; the image will not build on an Apple-silicon laptop.

## What is measured, and what is not

* Metrics: 40 held-out SMIYC validation frames (10 Anomaly, 30 Obstacle). Small,
  so treat differences under a point as noise.
* Latency: measured on A100-SXM4-80GB and Apple M5. **Not** measured on
  automotive hardware; an Orin figure would need an Orin.
* The corpus is 572 frames. Adding unlabelled city driving data and cut-paste
  outlier supervision are the two obvious levers left untouched.

## Layout

```
infer.py               use the released weights
profile_student.py     end-to-end latency, per step and per module
config.yaml            DataSphere job spec
Dockerfile             linux/amd64, CUDA 12.1 runtime
entrypoint.sh          stage orchestrator
common.py  s3sync.py   shared plumbing
stages/                label, train, evaluate, export
student/               model, dataset, losses
weights/               student_final.pt (14 MB)
results/               measured metrics and latency reports
patches/teacher.patch  fixes required on the RAAS side
```
