# RAAS / Maskomaly

Anomaly by **elimination**. Instead of asking whether a pixel looks strange, this
asks whether anybody claimed it: Mask2Former emits 100 query masks with class
distributions, every confident non-void query suppresses its own region, and
whatever is left unclaimed is the anomaly.

That inversion is why the map starts at **one** everywhere and is pulled down,
rather than starting at zero and being pushed up.

```
Mask2Former (Swin-L, Cityscapes)
  ├─► mask_cls  (100, 20)      softmax over 19 classes + void
  └─► mask_pred (100, H, W)    per-pixel sigmoid
        │
        ├─► rejection  = min over confident non-void queries of 1 − mask·conf
        ├─► promotion  = max over queries [49,31,83,32] of mask·conf
        ├─► contested pixels (claimed by ≥2 queries) → 0
        ├─► ground correction from queries [19,24]
        └─► 0.6·rejection + 0.4·promotion
                │
   variants _id / _ood only:
                ├─► road query #20 → fill contours → polygon ∧ ¬road = holes
                └─► CLIP over each hole → write 0.05 (known) or 1.0 (anomaly)
```

## The three variants

| `variant` | what runs after the soft mask |
|---|---|
| `maskomaly` | nothing — the soft mask is the answer |
| `maskomaly_id` | road-hole candidates, CLIP softmax over 19 known prompts, known if > 0.85 |
| `maskomaly_ood` | the same, plus 3 open-ended OOD prompts and a three-way margin rule |

They are one pipeline with a switch, so they live in one method under a config
key rather than as three methods.

**"Road-aware" is a geometric trick, not an appearance model.** The road query
segments drivable surface; anything standing on the road is *not* road, so it
appears as a hole in that mask while the road's outer boundary stays intact.
Filling the road's external contours and subtracting the road leaves exactly the
objects sitting on it. It costs two morphology calls and no network — and its
limits follow directly: an obstacle touching the road's edge encloses no hole,
and a puddle, shadow or repair patch is indistinguishable from an object.

**CLIP overwrites rather than modulates.** Both variants write a hard `0.05` or
`1.0` into each candidate region, discarding whatever the soft mask believed
there. That is the original's behaviour, preserved deliberately.

## Reference numbers (SMIYC validation: 10 Anomaly + 30 Obstacle frames)

| variant | AnomalyTrack AUPR / mean F1 | ObstacleTrack AUPR / mean F1 |
|---|---|---|
| `maskomaly` | 94.09 / 63.60 | 88.62 / 53.21 |
| `maskomaly_id` | 93.18 / 62.26 | 91.21 / 58.65 |
| `maskomaly_ood` | 93.18 / 62.26 | 91.21 / 58.65 |

**`_id` and `_ood` are bit-identical on all 40 frames.** The cause was never
established. Do not read the table as evidence that the OOD prompts do nothing —
read it as an unresolved question. Comparing the two runs' stored maps directly
is the obvious next step.

Latency: **2.5–3.4 s/frame on an A100.** The dominant cost was an all-pairs loop
over ~100 full-resolution masks; this port collapses it (see below), which should
move that number, but it has not been measured here.

## This method cannot currently be run

It needs the RAAS monorepo, which is a separate checkout and is not vendored:

```bash
export RAAS_ROOT=/path/to/raas          # must contain Mask2Former/ detectron2/ Maskomaly/
export RAAS_WEIGHTS=$RAAS_ROOT/Maskomaly/maskomaly/ckpt/model_final_17c1ee.pkl
pip install -r methods/raas/requirements.txt
pip install git+https://github.com/openai/CLIP.git   # _id and _ood variants only
open-road run --method raas --dataset road_anomaly
```

detectron2 must be **built from source** inside `$RAAS_ROOT` (`pip install -e .`),
never `pip install detectron2`.

### The checkpoint

`model_final_17c1ee.pkl` is the upstream Mask2Former Cityscapes Swin-L model
(`maskformer2_swin_large_IN21k_384_bs16_90k`, ImageNet-21k pretraining, 90k
iterations). No branch of this project recorded a URL for it, which is what made
these methods unreproducible; it is written down here now:

```bash
mkdir -p "$RAAS_ROOT/Maskomaly/maskomaly/ckpt"
curl -L https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/\
maskformer2_swin_large_IN21k_384_bs16_90k/model_final_17c1ee.pkl \
  -o "$RAAS_ROOT/Maskomaly/maskomaly/ckpt/model_final_17c1ee.pkl"
```

All three variants share it — `maskomaly_id` and `maskomaly_ood` have no weights
of their own, differing only in CLIP prompts and post-processing.

The numbers above are still quoted rather than reproduced: the checkpoint was
not present when this port was written.

Since the method could not be run, correctness rests on `tests/test_raas_maths.py`,
which pins every formula against hand-made query tensors — including a check that
the vectorised border rule is bit-identical to the original all-pairs loop.

## What this port changes

- **The O(N²) border loop is collapsed.** The original zeroed a pixel wherever
  some *pair* of non-void masks both exceeded 0.1. "Some pair exceeds" is exactly
  "at least two exceed", so the double loop becomes one sum over the query axis:
  bit-identical output, linear instead of quadratic, and this was the pipeline's
  dominant cost. A test checks the two against each other on random input.
- **`$RAAS_ROOT` replaces `Path(__file__).parent.parent`.** The original could
  only run from inside the monorepo.
- **Query indices out of range fail with the reason.** They are tuned to one
  checkpoint; against other weights the old failure was a bare `IndexError` deep
  in a loop.
- **`MODEL.DEVICE` is set explicitly.** detectron2 defaults to cuda and is
  otherwise unusable on a laptop.

## Known fragilities, carried over deliberately

- **The magic indices** — anomaly `[49, 31, 83, 32]`, ground `[19, 24]`, road `20`,
  void class `19` — hold only for `model_final_17c1ee.pkl`. They were ranked on
  SMIYC and Cityscapes validation. Nothing verifies them against the loaded model.
- **`maskomaly` computes `self.ranking` from `--analysis-file` and then ignores
  it**; its loop hardcodes the defaults. `_id` and `_ood` do honour it. Preserved,
  because changing it changes the published numbers.
- **No minimum component size** anywhere in the candidate path: components one
  pixel wide are skipped, everything else reaches CLIP.
- **`detectron2_replacements/` must precede `detectron2/` on `sys.path`.** The
  patched predictor returns three values where the stock one returns one; get the
  order wrong and the failure surfaces far away as
  `ValueError: not enough values to unpack (expected 3, got 1)`.

## Looking at what it did

```
runs/raas/<dataset>/intermediate/<frame>/
  01_soft_mask.png    the elimination map, before any CLIP edit
  02_road_holes.png   candidate geometry            (_id / _ood only)
  03_after_clip.png   the map after the hard writes (_id / _ood only)
  semantic.png        Cityscapes argmax from the same forward pass
  stages.json         per-candidate verdicts: bbox, label, probabilities
```

Comparing `01` against `03` is the direct way to see what CLIP actually changed,
which matters given it overwrites rather than modulates.

## Credit

[Maskomaly](https://github.com/jan-ackermann/Maskomaly) (Ackermann et al.) for the
elimination formula; [Mask2Former](https://github.com/facebookresearch/Mask2Former)
(MIT) and [detectron2](https://github.com/facebookresearch/detectron2) (Apache 2.0)
for the backbone; [CLIP](https://github.com/openai/CLIP) for the arbiter. The
road-polygon and CLIP stages follow arXiv:2607.04304 (Yan & Elger).
