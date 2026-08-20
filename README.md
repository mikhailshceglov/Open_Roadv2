# Open Road

Five road-anomaly segmentation methods, one pipeline, one set of metrics.

<!-- ФОТО: обзорный кадр — оверлей на дороге с найденным препятствием.
     Взять из runs/raas_distill/road_anomaly/render/overlay/obstacles09_boulder3.jpg -->

The methods here agree about almost nothing. One is a 3.7M-parameter student
that runs in a third of a second; another chains three networks and takes a
minute a frame. One emits a calibrated sigmoid, another a set of instances.
They pin incompatible versions of the same library and cannot share a Python
environment.

They agree on exactly one thing, and the whole repository is built on it:

```python
score(image_bgr: np.ndarray) -> np.ndarray   # HxW float32, higher = more anomalous
```

Everything downstream of that — thresholding, components, overlays, video,
pixel and component metrics, comparison tables — is written once and shared.

## Branch layout

**`main` carries the skeleton and no methods at all.** Each method lives on its
own branch:

```
main                 the skeleton: CLI, run layout, render, metrics, datasets
├── raas-distill     methods/raas_distill/       SegFormer-B0 student, 3.7M params
├── ooddino          methods/ooddino/            GroundingDINO + SegFormer + SAM 2.1
├── raas             methods/raas/               Maskomaly over Mask2Former queries
├── raas-objectomaly methods/raas_objectomaly/   + SAM-guided boundary refinement
└── raas-sky-road    methods/raas_sky_road/      + semantic class attenuation
```

A method branch **touches no file that `main` owns**. All five merge into `main`
and into each other without conflicts — there is a check for it in
[Verifying](#verifying).

That is only possible because registration is by filesystem, not by a central
table. `open_road/registry.py` walks `methods/*/method.py`, imports each and
reads a module-level `METHOD` constant. Adding a method is adding a directory.
A central `REGISTRY = {...}` in `main` would conflict on every single merge.

## The five methods

| method | branch | what it is | on RoadAnomaly |
|---|---|---|---|
| **RAAS-Distill** | `raas-distill` | SegFormer-B0 distilled from Mask2Former Swin-L + CLIP | AP 75.6, 0.36 s/frame |
| **OoDDINO** | `ooddino` | Detector proposals gate a dense score, SAM sharpens | AP 31.9, 3.73 s/frame |
| **RAAS** | `raas` | Anomaly by *elimination* over Mask2Former queries | see method README |
| **RAAS + Objectomaly** | `raas-objectomaly` | RAAS with SAM lending it boundaries | see method README |
| **RAAS sky-road** | `raas-sky-road` | + attenuate by predicted class, restore exceptions | see method README |

Each has a README with its architecture, its measured numbers, its known
defects, and what was verified versus what is quoted from elsewhere.

<!-- ФОТО: сравнение методов на одном кадре — 2х2 или ряд.
     Слева raas_distill, справа ooddino, из runs/*/road_anomaly/render/overlay/ -->

### Reading the pictures

Every overlay uses the same four colours:

| | |
|---|---|
| 🔴 red fill | the predicted mask, 50% opacity |
| ⚪ white outline | the contour of that same mask |
| 🟡 yellow box | one connected component, labelled with its mean score |
| 🟢 green outline | ground truth |

Red and white are the model's answer; green is the correct one. Green with no
red inside is a miss; red outside green is a false alarm. Unlabelled datasets
(TAD) have no green — there is nothing to compare against.

## Quick start

```bash
pip install -e .
open-road methods                 # what this branch carries
open-road datasets                # what data is present

open-road run --method raas_distill --dataset road_anomaly
```

`run` is infer → eval → render. Eval goes before render on purpose: it sweeps
the threshold and reports the best-F1 operating point, and render then draws
*that* point. Rendering at a fixed default instead would draw one mask and score
a different one.

Add `--video-fps 10` on a clip dataset and it encodes the overlays into an mp4.

## Where a run puts things

```
runs/<method>/<dataset>/
  score_raw/<stem>.npy            float32 HxW — the method's own output
  soft_mask/<stem>_soft_mask.png  8-bit preview of the same map
  render/mask/<stem>.png          thresholded binary mask
  render/overlay/<stem>.jpg       that mask drawn over the frame
  render/regions.json             every component, with the knobs that made it
  intermediate/<stem>/            the method's internals, when it exposes any
  <dataset>_overlay.mp4           clip datasets only
  metrics.json                    labelled datasets only
  run.json                        what ran, with what, and when
```

**`score_raw` is the only artefact that is not derived.** Everything below it is
one threshold and one component filter away, and both are recorded in
`regions.json`, so any picture can be traced back to the numbers that produced
it. Re-render at a different threshold without re-running the model:

```bash
open-road render --method raas_distill --dataset road_anomaly --threshold 0.6
```

<!-- ФОТО: содержимое intermediate/ у ooddino — сетка из восьми карт стадий.
     runs/ooddino/road_anomaly/intermediate/<кадр>/01..08 -->

## Metrics

Pixel metrics are pooled over every valid pixel of every frame: AP, AUROC,
FPR@95, and precision/recall/F1 at the best-F1 threshold. Component metrics
follow SegmentMeIfYouCan: sIoU over ground-truth components, PPV over predicted
ones, and F1\* averaged over eleven thresholds from 0.25 to 0.75.

Three things the evaluator does that are easy to get wrong:

- **Void pixels are excluded**, not counted as negatives. A dataset marking
  ignore regions as 255 would otherwise be scored against them.
- **`min_area` reaches the component metrics.** Render drops small components;
  scoring the unfiltered mask would measure a mask nobody ships. On RoadAnomaly
  that was 774 predicted components against 298 real ones.
- **An empty ground truth stops the run** instead of reporting metrics against
  nothing — the usual cause is a label-encoding mismatch.

Compare runs across methods, including ones whose dependencies conflict:

```bash
open-road report --runs runs/raas_distill/road_anomaly runs/ooddino/road_anomaly
```

Only `infer` ever imports a method. `render`, `eval` and `report` read the
stored `.npy` maps, so they work in a plain environment even where the method's
own pins are not installed. That is what makes the comparison possible at all.

## Repository layout

| directory | what is in it |
|---|---|
| [`src/open_road/`](src/open_road/) | the skeleton: contract, registry, stages, CLI |
| [`methods/`](methods/) | one directory per method — empty on `main` |
| [`configs/`](configs/) | dataset and method descriptions, as YAML |
| [`scripts/`](scripts/) | dataset preparation |
| [`tests/`](tests/) | the skeleton's tests, plus each method's own |
| `data/`, `runs/`, `reports/` | inputs and outputs, all git-ignored |

Each has its own README.

## Adding a method

```
methods/<name>/
  method.py          module-level METHOD = MethodSpec(...)
  requirements.txt   this method's pins, however incompatible
  README.md          architecture, numbers, known defects
configs/methods/<name>.yaml
tests/test_<name>_*.py
```

`method.py` declares what the shared stages cannot guess — score range, default
threshold, default minimum component area — and a `build` callable returning
something with `score(image_bgr)`. Keep heavy imports inside it: constructing the
spec must stay free, because `open-road methods` builds it just to print a line.

A method may also leave a `last_debug` dict on the scorer; `infer` writes it
under `intermediate/`, dispatching on type without interpreting anything.

## Verifying

```bash
python -m pytest -q                       # 42 on main; each method adds its own

# a branch must touch nothing main owns
git diff --stat main <branch> -- src/ configs/datasets/ scripts/ pyproject.toml

# and all five must merge cleanly
for b in raas-distill ooddino raas raas-objectomaly raas-sky-road; do
  git checkout main && git merge --no-commit --no-ff $b && git merge --abort
done
```

## Datasets

| name | frames | labels |
|---|---|---|
| `road_anomaly` | 60 | per-pixel |
| `road_anomaly_sample` | 10 | per-pixel — best and worst frame per category |
| `smiyc_obstacle` | 30 | per-pixel |
| `tad_*` | 41 clips, 7473 frames | clip-level only — pictures, not metrics |

`scripts/prepare_road_anomaly.py` flattens the RoadAnomaly release and remaps its
anomaly label from 2 to 1. Dataset YAML understands `$VARS`, so paths can point
outside the repository.

<!-- ФОТО: кадры TAD — ряд из трёх категорий (RoadSpills, PedestrianOnRoad, Accident).
     runs/raas_distill/tad_*/render/overlay/ -->

## Licence

The skeleton is MIT. Each method carries its own licence and its own credits —
some depend on checkpoints and repositories with terms of their own, and those
are spelled out in the method READMEs rather than here.
