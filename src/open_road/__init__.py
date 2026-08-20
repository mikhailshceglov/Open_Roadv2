"""Open Road: one skeleton, many road-anomaly methods.

``main`` carries this package and nothing else; each method lives on its own
branch under ``methods/<name>/``. The seam between them is deliberately narrow:
a method provides ``score(image_bgr) -> HxW float32``, and everything else --
rendering, metrics, run layout, the CLI -- is shared and written once.
"""

from open_road.method import MethodSpec, Scorer

__all__ = ["MethodSpec", "Scorer"]
__version__ = "0.1.0"
