# Tests

```bash
python -m pytest -q
```

42 on `main`; each method branch adds its own on top.

| file | what it pins |
|---|---|
| `test_dataset.py` | unknown keys rejected, label encodings, void handling |
| `test_registry.py` | filesystem discovery, broken methods reported not fatal |
| `test_evaluate.py` | metrics, void exclusion, `min_area` reaching components |
| `test_render_and_infer.py` | thresholding, speck filtering, resumability |
| `test_report.py` | comparison tables across runs |
| `test_video.py` | natural frame order, encoding, frame counts |

## Why the maths is tested and the models are not

Three of the five methods **cannot be run here** without weights that are large,
externally hosted, or licence-encumbered. Their formulas were therefore pulled
out into pure-numpy modules with no model behind them, and pinned against
hand-made tensors.

That is the whole reason the port can be trusted: a reconstruction of unpublished
equations is either tested or guessed at.

Examples of what that buys:

- `test_ooddino.py` — entropy is 0 when certain and 1 when uniform; the ADT
  decision is exactly score-versus-regional-threshold; `delta` softens the
  reported probability without moving the decision.
- `test_raas_*_maths.py` — the vectorised border rule is **bit-identical** to
  the original all-pairs loop it replaces, checked on random input rather than
  argued.
- `test_raas_sky_road_fusion.py` — the road boost is taken against the original
  score so a small class factor cannot cancel it; and two live defects in the
  shipped config are pinned as tests, including that shortening one list revives
  a rule that is currently unreachable.

Tests that describe a defect rather than a requirement say so in a comment. They
exist so nobody has to rediscover it.

## Naming

`test_<subject>_<what_it_pins>`. A test name should state the behaviour, so a
failure reads as a sentence about what broke.
