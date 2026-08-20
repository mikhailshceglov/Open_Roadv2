# Training: how the student was distilled

> **This does not run out of the box, and that is not a bug to be fixed here.**
> The `label` and `evaluate` stages need the RAAS monorepo — detectron2,
> Mask2Former, Maskomaly and a Swin-L checkpoint — which is a separate checkout
> that is not vendored in this repository. Point `$RAAS_ROOT` at one. The
> `train` and `export` stages need only the labelled targets and run anywhere.
>
> Inference on the released weights imports nothing from this directory.

## Why the teacher is not in the loop

Distillation is offline. The teacher's post-processing is numpy and per-component
CLIP, so it is not differentiable and cannot sit in the graph. The teacher labels
a corpus once, then the student trains on those stored maps.

| stage | what it does | needs `$RAAS_ROOT` |
|---|---|---|
| `stages/label.py` | teacher → float16 anomaly maps + quarter-res semantic logits | yes |
| `stages/train.py` | student on those maps | no |
| `stages/evaluate.py` | SMIYC metrics + teacher fidelity on unlabelled frames | yes |
| `stages/export.py` | ONNX + latency | no |

`entrypoint.sh` runs them in order, driven by `STAGES`, `ITERS`, `CROP`, `BATCH`.
Each stage writes `$OUT_ROOT/<stage>.done` on success and is skipped on re-run,
so a job that dies in training does not re-pay for labelling.

```bash
export RAAS_ROOT=/path/to/raas
export DATA_ROOT=data/raas_distill
export OUT_ROOT=runs/raas_distill/train
methods/raas_distill/train/entrypoint.sh
```

Stages are modules, not loose scripts, so they can also be run one at a time:

```bash
python -m methods.raas_distill.train.stages.train --iters 10000 --crop 768 --batch 16
```

## Data layout

```
$DATA_ROOT/
  frames/<any>/<any>.{png,jpg,jpeg,webp}   corpus to distil on
  smiyc/dataset_AnomalyTrack/{images,labels_masks}
  smiyc/dataset_ObstacleTrack/{images,labels_masks}
  ckpt/model_final_17c1ee.pkl              teacher weights
```

The labelling stage walks `$DATA_ROOT/frames` recursively. Any frame whose name
starts with `validation` is held out as benchmark and never enters training —
and the stage **aborts** if the held-out count is not exactly 40. That guard is
deliberate: a silent drift there either leaks the benchmark into training or
quietly shrinks the corpus, and both are worse than stopping.

## Patching the teacher

`patches/teacher.patch` applies to the RAAS tree, not to this one. It carries
two fixes:

* the all-pairs query overlap loop in `maskomaly/model_*.py` is O(N²) over
  full-resolution masks. A pixel is zeroed exactly when two or more masks exceed
  0.1 there, so the whole double loop collapses to one sum — bit-identical
  output, ~33× faster, and the difference between a labelling run that is
  GPU-bound and one that is not;
* `MSDeformAttn` raises unconditionally when its CUDA kernel is absent, which
  makes the model unimportable on any machine without one. It becomes a warning,
  falling back to the PyTorch implementation.

```bash
cd "$RAAS_ROOT" && git apply /path/to/Open_Road/methods/raas_distill/train/patches/teacher.patch
```

## Running it on Yandex DataSphere

`../config.yaml` is the job spec. Fill in `<REGISTRY_ID>` and `<YOUR_BUCKET>`,
build and push the image, then:

```bash
datasphere project job execute -p <PROJECT_ID> -c methods/raas_distill/config.yaml
```

The container fetches its own inputs from Object Storage rather than using a
DataSphere S3 mount: mounts need a connector that can only be created in the
project UI, and there is no API for it. Credentials come from
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in the job vars and are never
written to this repository.

The Docker build context is a directory holding both this repository and the
RAAS checkout side by side — see the header of `Dockerfile`. Build for
`linux/amd64`; the image will not build on an Apple-silicon laptop.

## What was measured

The corpus is 572 frames; training was 10 000 iterations at crop 768, batch 16,
taking 2h35m (`../results/train_stats.json`). Adding unlabelled city driving
data and cut-paste outlier supervision are the two obvious levers left
untouched.
