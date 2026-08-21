# RAAS sky-road

One stage longer than `raas_objectomaly`, and the stage goes in the middle:

```
RAAS ──► OASC ──► global fusion ──► MBP ──► refined
                        │
        attenuate by predicted class, then let
        compact SAM regions argue their score back
```

![The coarse map and the semantics fusion keys on](images/coarse-input.jpg)

*Not this method's output — it has never been run here. These are its two inputs,
both from the `raas` branch on one frame: left the anomaly map, right the
Cityscapes semantics that decide every class factor. Every bright pixel on the
left that sits on sky or vegetation on the right is what fusion attenuates.*

## The idea

A road-scene anomaly detector wastes most of its false positives on classes that
are never the answer. Sky is featureless and uncertain, vegetation is
high-frequency noise, building facades are cluttered — all three light up a
residual-energy map, and none of them is an obstacle in the road.

So every pixel's score is multiplied by a factor keyed on its predicted
Cityscapes class:

| class | factor | | class | factor |
|---|---:|---|---|---:|
| road | 1.00 | | vegetation | 0.55 |
| person / vehicles | 0.90 | | building / wall | 0.55 |
| sidewalk | 0.85 | | **sky** | **0.35** |

**Nothing is hard-masked, and that is the whole point.** A mask cannot be undone,
and this branch exists because an object *can* be in the sky — a falling tyre,
cargo off a truck, airborne debris. Six of the twelve OOD prompts name exactly
that. Attenuation leaves the score recoverable; a mask would not.

Recovery is the second half. SAM proposes compact regions, each is measured
against its own surroundings, and a region that looks like an object gets its
**original, un-attenuated score restored**:

```
fused = clip( f[class] · score,                     unprotected
              max(f[class] · score, score),         protected
              max(…, 0.65),                         protected by CLIP
              0, 1 )
```

Three independent tests can protect a region, and any one suffices:

- **contrast** — its 90th-percentile score minus the median of an elliptical
  ring just outside it. Judging a region against its own surroundings rather
  than the whole frame is what makes a bright object on dark tarmac and a bright
  object against bright sky score alike.
- **semantic** — its dominant class is not background. **Currently unreachable**;
  see below.
- **CLIP** — a batched comparison of 13 "normal road scene" prompts against 12
  "airborne debris" ones. Each region is embedded twice, masked and with
  context, and the two are averaged.

## Two live defects, carried over deliberately

**`background_class_ids` lists all 19 classes**, which makes the `semantic_object`
rule unreachable — every dominant class is a background class, so only an empty
mask could pass, and those are filtered earlier. This is in the shipped config
and the source branch's own report confirms it. Kept unchanged because it is what
the published numbers were measured with; `test_raas_sky_road_fusion.py` pins
both the defect and the fact that shortening the list revives the rule.

**`clip_margin_threshold: 0.0`** means any region whose best OOD prompt merely
beats its best normal prompt is floored at 0.65. Combined with
`clip_probe_min_score: 0.05` and `max_candidates: 96`, the source branch's report
blames this for the AnomalyTrack PPV collapse.

Those two are the first and second experiments to try.

## Reference numbers (SMIYC validation: 10 Anomaly + 30 Obstacle frames)

| | AnomalyTrack AUPR / mean F1 | ObstacleTrack AUPR / mean F1 |
|---|---|---|
| RAAS `maskomaly` | 94.09 / 63.60 | 88.62 / 53.21 |
| RAAS `maskomaly_id` | 93.18 / 62.26 | 91.21 / 58.65 |
| **+ fusion**, `maskomaly` | 89.48 / 39.49 | 75.12 / 65.46 |
| **+ fusion**, `_id` / `_ood` | 88.85 / 38.36 | **92.37 / 73.00** |

**The result is genuinely ambiguous, and presenting it as a win would be wrong.**
ObstacleTrack gains 14 points of mean F1 and PPV climbs 53.6 → 89.4. AnomalyTrack
loses 24 points to false object components. A method that clearly wins on small
road obstacles and clearly loses on large anomalies is not better — it is
differently tuned, and the config was tuned on the same 40 frames it is scored
on. Splitting the config per track is the obvious next move.

`_id` and `_ood` produce **bit-identical** metrics here, as they do without
fusion. Nobody established why.

## Cost

**58–64 s/frame on an A100**, of which SAM is ~41 s (65–71%), fusion and CLIP
22–24%, and RAAS itself 4–5%. Not a real-time method, and no tuning makes it one.
This config runs SAM at `points_per_side: 64`, denser than the plain refinement's
32, because fusion needs small regions to exist before it can protect them.

## Weights

Three checkpoints, all fetched by hand except CLIP's.

**Mask2Former Swin-L, Cityscapes semantic**, shared by all three RAAS variants —
they differ only in CLIP prompts and post-processing, not in weights:

```bash
mkdir -p "$RAAS_ROOT/Maskomaly/maskomaly/ckpt"
curl -L https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/\
maskformer2_swin_large_IN21k_384_bs16_90k/model_final_17c1ee.pkl \
  -o "$RAAS_ROOT/Maskomaly/maskomaly/ckpt/model_final_17c1ee.pkl"
```

**SAM ViT-H**, for the refinement stage. `-C -` resumes an interrupted download;
the digest is verified on load, because SAM ships several checkpoints with
similar names and the wrong one degrades results silently:

```bash
curl -L -C - https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
  -o checkpoints/sam_vit_h_4b8939.pth
md5sum checkpoints/sam_vit_h_4b8939.pth   # 4b8939a88964f0f4ff5f5b2642c598a6
```

**OpenAI CLIP ViT-B/32** needs no manual download — `clip.load("ViT-B/32")`
fetches it into `~/.cache/clip/ViT-B-32.pt`. Pin the implementation:

```bash
pip install git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1
```

**Objectomaly ships no weights.** OASC and MBP are algorithms with no
checkpoints; only the code is needed, and it must stay an unmodified external
checkout — the pinned commit has no LICENSE.

## Running it

```bash
export RAAS_ROOT=/path/to/raas
export OBJECTOMALY_ROOT=/path/to/Objectomaly     # cloned at 66d2ad2a, unmodified
export SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth
pip install -r methods/raas_sky_road/requirements.txt
open-road run --method raas_sky_road --dataset road_anomaly
```

`global_fusion.enabled: false` runs the plain refinement, which is the ablation
worth measuring first.

## What is verified, and what is not

This method has **not been run**: none of the three checkpoints is present here.
So verification is on the arithmetic:

- `tests/test_raas_sky_road_fusion.py` — 20 tests over attenuation, the road
  boost taken against the original score, ring contrast separating an object
  from a bright background, candidate ranking, and each protection rule
  independently. Including the two defects above, pinned as tests.
- `tests/test_raas_sky_road_maths.py` — the RAAS formulas.
- `tests/test_raas_sky_road_refine.py` — the Objectomaly bridge.

**No number in this README was reproduced.** They are quoted from the source
branch's `metrics/summary.json` and its profiler output.

## Looking at what it did

```
runs/raas_sky_road/<dataset>/intermediate/<frame>/
  01_coarse.png        RAAS, before anything
  02_calibrated.png    after OASC
  03_road_holes.png    candidate geometry        (_id / _ood variants)
  04_fused.png         after semantic fusion
  05_protected.png     which regions argued their way back
  06_fusion_delta.png  fused − calibrated
  07_refined.png       after MBP
  08_total_delta.png   refined − coarse
  stages.json          every candidate: score, ring, contrast, class, CLIP margin, reasons
```

`05_protected` against `06_fusion_delta` is the pair to read: the first shows
what fusion spared, the second what it cost everywhere else. `stages.json`
records the `reasons` list per candidate, so an unexpected restoration can be
traced to the exact rule that caused it.

## Credit

[Maskomaly](https://github.com/jan-ackermann/Maskomaly) for the elimination
formula, [Mask2Former](https://github.com/facebookresearch/Mask2Former) and
[detectron2](https://github.com/facebookresearch/detectron2) for the backbone,
[Objectomaly](https://github.com/hon121215/Objectomaly) for OASC and MBP,
[Segment Anything](https://github.com/facebookresearch/segment-anything) for the
regions, and [CLIP](https://github.com/openai/CLIP) for the second opinion.
