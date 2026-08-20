# Methods

**Empty on `main`.** Each method lives on its own branch, in its own directory
here, and a branch touches no file that `main` owns.

```
main                 (this directory is empty)
├── raas-distill     raas_distill/       SegFormer-B0 student, 3.7M params
├── ooddino          ooddino/            GroundingDINO + SegFormer + SAM 2.1
├── raas             raas/               Maskomaly over Mask2Former queries
├── raas-objectomaly raas_objectomaly/   + SAM-guided boundary refinement
└── raas-sky-road    raas_sky_road/      + semantic class attenuation
```

`git checkout <branch>` to get one. Each carries a README with its architecture,
its numbers, and its known defects.

## What a method directory holds

```
methods/<name>/
  method.py          module-level METHOD = MethodSpec(...)   ← the only required file
  requirements.txt   this method's pins, however incompatible with the others
  README.md          architecture, measured numbers, known defects, weights
  <the rest>         whatever the method needs
```

`registry.discover()` walks this directory, imports each `method.py` and reads
`METHOD`. Nothing central is edited, which is why the branches merge.

## Why the RAAS methods duplicate code

`raas`, `raas_objectomaly` and `raas_sky_road` each carry their own copy of
`soft_mask.py`, `road_aware.py`, `clip_filter.py` and `backbone.py`.

That is deliberate. The three are *alternatives* — nobody runs two at once — and
a shared module would have to live in `main`, which owns no method code. Sharing
it would break the property that makes the branches mergeable.

## Dependencies genuinely conflict

`raas_distill` pins `transformers==4.44.2`, because SegFormer's internal module
names changed in 5.x and its released checkpoint will not load on the new layout.
`ooddino` needs `transformers>=4.51` for SAM 2.1. The two cannot share a Python
environment.

This is the case the skeleton was built for: only `infer` imports a method, so
each can be scored in its own environment and all of them compared by the same
evaluator afterwards.

## Keeping the spec cheap

`open-road methods` builds every spec just to print one line, so constructing a
`MethodSpec` must not import torch. Keep heavy imports inside the `build`
callable. A method whose dependencies are absent is then reported as unavailable,
with the pip command that would fix it, instead of taking the CLI down.
