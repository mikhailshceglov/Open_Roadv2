# Open Road

A skeleton for road-anomaly segmentation methods, and the methods themselves.

**`main` carries the skeleton and no methods.** Each method lives on its own
branch, in its own directory, with its own dependencies:

```
main                 the skeleton: CLI, run layout, render, metrics, datasets
└── raas-distill     methods/raas_distill/   SegFormer-B0 student, 3.72M params
```

One branch, one method. Branches are written so they merge into each other
without conflicts, which is only possible because a method branch touches no
file that `main` owns — see [Adding a method](#adding-a-method).

## The contract

The methods here disagree about nearly everything. One emits a sigmoid in
`[0, 1]`; another an unbounded `-sem_seg.tanh().sum(0)` living around
`[-19, 19]`. One is inference-only; another carries a four-stage training
pipeline. Their torch pins are not mutually satisfiable. They agree on exactly
one thing:

> one float per pixel, higher meaning more anomalous.

That primitive is the entire seam. A method provides
`score(image_bgr) -> HxW float32` and declares the few things that differ
between methods as data; rendering, metrics, the run layout and the CLI are
shared and written once.

```python
# methods/<name>/method.py
METHOD = MethodSpec(
    name="raas_distill",
    description="SegFormer-B0 student distilled from Mask2Former Swin-L + CLIP",
    build=_build,                 # config mapping -> Scorer, called lazily
    score_range=(0.0, 1.0),       # only for the 8-bit preview PNG
    default_threshold=0.5,
    default_min_area=200,
)
```

## Quick start

```bash
pip install -e ".[dev]"

# 1. get a dataset
python scripts/prepare_road_anomaly.py --src <RoadAnomaly_jpg> --out data/road_anomaly
open-road datasets

# 2. pick a method
git checkout raas-distill
pip install -r methods/raas_distill/requirements.txt
open-road methods

# 3. run it
open-road run --method raas_distill --dataset road_anomaly
```

`run` is `infer`, `render` and `eval` in order; each is also a command of its
own, so a swept threshold is one `render` away and costs no inference.

```bash
open-road infer  --method raas_distill --dataset road_anomaly
open-road render --method raas_distill --dataset road_anomaly --threshold -0.05 --draw both
open-road eval   --method raas_distill --dataset road_anomaly
open-road report --runs runs/raas_distill/road_anomaly --out reports/
```

Only `infer` imports the method. `render`, `eval` and `report` read the score
maps from disk, so they run in a plain environment even when the method's own
pins are not installed there — which is how two methods with incompatible torch
requirements still get compared with the same evaluator.

## Where things go

```
src/open_road/          the skeleton (this is all main owns of the code)
  method.py             MethodSpec, Scorer — the contract
  registry.py           finds methods by walking methods/
  dataset.py            DatasetSpec — describes a layout instead of assuming one
  io.py                 RunLayout — the run directory contract
  stages/               infer, render, evaluate, report
  cli.py
configs/datasets/*.yaml one file per dataset
configs/methods/*.yaml  one file per method, added by that method's branch
methods/<name>/         the method, added by that method's branch
scripts/                dataset preparation
data/                   downloaded and prepared datasets (git-ignored)
runs/                   run artefacts (git-ignored)
```

A run always looks the same, whatever produced it:

```
runs/<method>/<dataset>/
  score_raw/<stem>.npy            float32 HxW — the method's own output
  soft_mask/<stem>_soft_mask.png  8-bit preview, normalised by score_range
  render/mask/<stem>.png          the score map, thresholded
  render/overlay/<stem>.jpg
  render/regions.json             every component, with the knobs that made it
  metrics.json
  run.json                        what ran, with what settings, and when
```

`score_raw` is the only artefact that is not derived. Everything under it is one
threshold and one component filter away, and both are recorded in
`regions.json`, so a picture can always be traced back to the numbers.

## Datasets

A dataset is described, not renamed. Every benchmark in this area lays itself
out differently and encodes anomaly differently, and those differences are the
most common source of a silently zero-metric run: RoadAnomaly ships anomaly as
**2**, so code assuming `== 1` scores against an all-zero ground truth and
reports it without complaint.

```yaml
# configs/datasets/road_anomaly.yaml
name: road_anomaly
root: data/road_anomaly     # $VARS are expanded, so paths stay out of git
images_dir: original
labels_dir: labels
label_suffix: .png
anomaly_value: 1
void_value: null            # SMIYC uses 255; those pixels leave every metric
```

Unknown keys are an error rather than being ignored — the prototype this
replaces silently dropped a whole config block and then died on an
`AttributeError` far from the cause.

## Metrics

Pixel AP, AUROC, FPR95 and best-F1, pooled over every valid pixel; plus the
SegmentMeIfYouCan component metrics (sIoU_gt, PPV, F1\* averaged over
τ ∈ [0.25, 0.75]) at the best-F1 operating point. Void pixels are dropped from
all of them.

These are this repository's own metrics, not the official SMIYC evaluator. They
are internally consistent and fine for comparing methods here; they are not a
benchmark submission, and the component numbers apply no minimum component size,
so they are not paper-comparable.

## Adding a method

```bash
git checkout -b <method-name> main
mkdir -p methods/<method_name>
```

Then add, and add **only**:

| path | what it is |
|---|---|
| `methods/<method_name>/__init__.py` | make it a package |
| `methods/<method_name>/method.py` | `METHOD = MethodSpec(...)` |
| `methods/<method_name>/requirements.txt` | its own pins |
| `methods/<method_name>/README.md` | the method card: what it is, what it scores |
| `configs/methods/<method_name>.yaml` | its settings |

There is no registry to edit. A method registers by existing: `registry.py`
walks `methods/`, imports `methods.<name>.method`, and reads `METHOD`. A central
`REGISTRY = {...}` would be edited by every branch and would therefore conflict
on every merge, so there isn't one.

Use relative imports inside a method (`from .student.model import Student`) so
two methods that both ship a `student` package cannot shadow each other.

The check that the topology still holds:

```bash
git diff --stat main <method-name> -- src/ configs/datasets/ scripts/ pyproject.toml
# must be empty
```

If that diff is not empty, the branch has reached into shared code and the next
method will conflict with it.
