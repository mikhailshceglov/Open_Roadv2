"""Shared plumbing for the distillation stages.

The teacher lives in the RAAS repository, which is only importable after
``run_smiyc_eval.configure_import_paths`` has fixed ``sys.path`` (the patched
DefaultPredictor must shadow the upstream one).  Every stage goes through here
so that ordering is applied exactly once and in one place.
"""

import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

DISTILL_DIR = Path(__file__).resolve().parent
RAAS_DIR = DISTILL_DIR.parent
SCRIPTS_DIR = RAAS_DIR / "Maskomaly" / "scripts"

# Frames whose stem starts with this are the 40 labelled SMIYC validation
# frames.  They are the benchmark and must never enter the training corpus.
VAL_PREFIX = "validation"
EXPECTED_VAL_FRAMES = 40

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def data_root() -> Path:
    return env_path("DATA_ROOT", "/data")


def out_root() -> Path:
    return env_path("OUT_ROOT", "/out")


def _scripts_module(name: str):
    """Import a module from Maskomaly/scripts without installing the package."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    return importlib.import_module(name)


def teacher_defaults() -> SimpleNamespace:
    smiyc = _scripts_module("run_smiyc_eval")
    return SimpleNamespace(
        config_file=Path(os.environ.get("TEACHER_CONFIG", smiyc.DEFAULT_CONFIG)),
        weights=Path(os.environ.get("TEACHER_WEIGHTS", smiyc.DEFAULT_WEIGHTS)),
        masks=int(os.environ.get("TEACHER_MASKS", "4")),
        analysis_file=os.environ.get("TEACHER_ANALYSIS_FILE") or None,
    )


def load_teacher(name: str):
    """Build the RAAS teacher and expose its semantic output.

    ``get_soft_mask`` drops the segmentation dict that ``get_probs_and_seg``
    produced, but the student distils the 19-class semantics too.  Rather than
    editing the model files (and running the Swin-L forward twice), wrap the
    method so each call stashes the segmentation on the instance.
    """
    smiyc = _scripts_module("run_smiyc_eval")
    smiyc.configure_import_paths()
    if name not in smiyc.MODEL_MODULES:
        raise ValueError(
            "Unknown teacher {!r}; expected one of {}".format(
                name, sorted(smiyc.MODEL_MODULES)
            )
        )
    defaults = teacher_defaults()
    for label, path in (("config", defaults.config_file), ("weights", defaults.weights)):
        if not Path(path).is_file():
            raise FileNotFoundError("Teacher {} not found: {}".format(label, path))

    # Built here rather than through smiyc.load_model so that MODEL.DEVICE can be
    # overridden: detectron2 defaults to cuda, which makes the image impossible
    # to smoke-test anywhere without a GPU.
    module = importlib.import_module(smiyc.MODEL_MODULES[name])
    opts = ["MODEL.WEIGHTS", str(Path(defaults.weights).resolve())]
    device = os.environ.get("TEACHER_DEVICE")
    if device:
        opts += ["MODEL.DEVICE", device]
    model = module.Maskomaly(
        SimpleNamespace(
            config_file=str(Path(defaults.config_file).resolve()),
            opts=opts,
            masks=defaults.masks,
            analysis_file=defaults.analysis_file,
        )
    )

    original = model.get_probs_and_seg

    def capturing(image):
        result = original(image)
        model.last_segmentation = result[2]
        return result

    model.get_probs_and_seg = capturing
    model.last_segmentation = None
    return model


def soft_mask_of(model, image):
    """Call ``get_soft_mask`` across the three teacher variants.

    ``model_ori`` returns ``(soft_mask, mask_pred_result)``; ``model_id`` and
    ``model_ood`` return the map alone.
    """
    result = model.get_soft_mask(image)
    if isinstance(result, tuple):
        return result[0]
    return result


def is_validation_frame(path: Path) -> bool:
    return path.stem.startswith(VAL_PREFIX)


def list_corpus(root: Path):
    """Every training frame under ``root``, with the 40 val frames removed.

    Raises if the count of excluded frames is not exactly 40 — a silent drift
    here would either leak the benchmark into training or quietly shrink the
    corpus, and both are worse than stopping.
    """
    if not root.is_dir():
        raise FileNotFoundError("Corpus root does not exist: {}".format(root))

    frames, held_out = [], []
    # os.walk with followlinks, not Path.rglob: rglob refuses to descend into
    # symlinked directories, and datasets are very often mounted or linked in.
    for directory, _subdirs, filenames in os.walk(root, followlinks=True):
        current = Path(directory)
        if any(part in ("labels", "labels_masks") for part in current.parts):
            continue
        for filename in sorted(filenames):
            # Skip dotfiles, and in particular macOS AppleDouble twins (._name):
            # they carry image extensions, so an extension check alone lets them
            # through and cv2.imread then returns None on every one.
            if filename.startswith("."):
                continue
            path = current / filename
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            (held_out if is_validation_frame(path) else frames).append(path)
    frames.sort()
    held_out.sort()

    if len(held_out) != EXPECTED_VAL_FRAMES:
        raise RuntimeError(
            "Expected {} held-out validation frames under {}, found {}. "
            "Refusing to continue: the benchmark split is not what the corpus "
            "layout implies.".format(EXPECTED_VAL_FRAMES, root, len(held_out))
        )
    if not frames:
        raise RuntimeError("No training frames found under {}".format(root))
    return frames, held_out


def target_path(frame: Path, corpus_root: Path, targets_dir: Path) -> Path:
    """Mirror the corpus layout so frames from different datasets cannot collide."""
    relative = frame.relative_to(corpus_root).with_suffix(".npz")
    return targets_dir / relative
