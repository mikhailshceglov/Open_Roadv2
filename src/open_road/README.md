# The skeleton

Everything here is method-agnostic. Nothing in this directory knows the name of
a single method, and that is enforced by the fact that `main` carries no methods
at all — if the skeleton needed to know about one, `main` would not run.

## The seam

```python
score(image_bgr: np.ndarray) -> np.ndarray   # HxW float32, higher = more anomalous
```

The methods disagree about score range, thresholds, dependencies, whether they
train, and whether they output pixels or instances. They agree on one primitive.
So the primitive is the seam, and **everything they disagree about becomes data
the method declares** rather than a branch in shared code.

| file | what it holds |
|---|---|
| [`method.py`](method.py) | `Scorer` protocol and `MethodSpec` — the contract |
| [`registry.py`](registry.py) | discovery by walking `methods/`, not a central table |
| [`dataset.py`](dataset.py) | `DatasetSpec`: where frames are, how labels are encoded |
| [`io.py`](io.py) | `RunLayout` — every run writes the same shape |
| [`paths.py`](paths.py) | repository root, run directories, `$VAR` expansion |
| [`device.py`](device.py) | `auto` → cuda, then mps, then cpu |
| [`cli.py`](cli.py) | the seven verbs |
| [`stages/`](stages/) | infer, render, evaluate, report, video |

## Why registration walks the filesystem

`registry.discover()` walks `methods/*/method.py`, imports each through
`importlib`, and reads a module-level `METHOD`. A method is registered by
existing.

The obvious alternative — a `REGISTRY = {...}` dict in `main` — would be edited
by every method branch and would therefore conflict on **every single merge**.
The filesystem walk is what makes "one branch, one method, no conflicts"
possible at all.

A method whose dependencies are missing is reported as broken, with the import
error and the pip command that would fix it, rather than taking the CLI down.

## What `MethodSpec` declares

- `score_range` — for scaling the 8-bit preview. A sigmoid needs none; RbA's
  unbounded `-tanh().sum()` needs a lot.
- `default_threshold`, `default_min_area` — swept per method and meaningless
  across them. RbA's −0.0161 says nothing to a method emitting probabilities.
- `build` — a callable returning something with `score()`. Heavy imports live
  inside it, so building the spec stays free.

## Which stages import a method

Only `infer`. `render`, `evaluate` and `report` read `score_raw/*.npy`, so they
run in a plain environment even where the method's own pins are not installed —
which is what lets two methods with incompatible `transformers` versions be
compared with the same evaluator.

## Two things the stages get right that are easy to get wrong

**Frame order is natural, not lexicographic.** Clip datasets number frames
without zero padding, so sorting as text runs 0, 1, 10, 100, 11. Nothing about
that is visible in any single frame — the clip just plays as nonsense.

**`min_area` reaches the metrics.** Render drops small components and evaluate
applies the same filter, so the mask that is scored is the mask that ships.
