"""CLIP as a second opinion on SAM regions, batched.

Distinct from ``clip_filter.py``, which arbitrates road-hole candidates with a
softmax over Cityscapes class names. This one compares two *free-text* prompt
sets — "normal road scene" against "airborne debris" — and reports the raw
similarity margin between the best of each.

Two details are load-bearing:

* each region is embedded **twice**, once with its background greyed out and
  once with context left in, and the two embeddings are averaged. A masked crop
  alone loses the context that tells sky from tarmac; an unmasked crop alone
  lets the background dominate a small object.
* the decision uses a raw cosine-similarity **margin**, not a softmax. There is
  no logit scale and no normalisation across prompts, so the number is not a
  probability and the `clip_ood_probability` recorded beside it is a monotone
  restatement, never a decision input.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .fusion import sigmoid

PINNED_COMMIT = "d50d76daa670286dd6cacf3bcd80b5e4823fc8e1"


class CLIPRegionValidator:
    """Scores SAM regions as normal-versus-OOD, in batches."""

    def __init__(self, config: Mapping[str, Any], device: str = "cpu") -> None:
        try:
            import clip
            import torch
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "Global fusion CLIP validation is enabled but OpenAI CLIP is not "
                f"installed. pip install git+https://github.com/openai/CLIP.git@{PINNED_COMMIT} "
                "(the unrelated PyPI package named `clip` will shadow it -- remove that first). "
                "Set clip.enabled false to run fusion on geometry alone."
            ) from error

        self.torch = torch
        self.device = device
        self.model, self.preprocess = clip.load(
            str(config.get("model", "ViT-B/32")), device=device, jit=False
        )
        self.model.eval()

        self.normal_prompts = list(config.get("normal_prompts", []))
        self.ood_prompts = list(config.get("ood_prompts", []))
        if not self.normal_prompts or not self.ood_prompts:
            raise ValueError("CLIP fusion needs both normal_prompts and ood_prompts")

        tokens = clip.tokenize(self.normal_prompts + self.ood_prompts).to(device)
        with torch.no_grad():
            text = self.model.encode_text(tokens).float()
            self.text_features = text / text.norm(dim=-1, keepdim=True)

        self.batch_size = int(config.get("batch_size", 16))
        self.context_fraction = float(config.get("context_fraction", 0.15))
        self.mask_background = bool(config.get("mask_background", True))
        self.include_context_view = bool(config.get("include_context_view", True))

    def _crop(self, image_bgr: np.ndarray, mask: np.ndarray, bbox: Sequence[int],
              mask_background: bool):
        import cv2
        from PIL import Image

        height, width = image_bgr.shape[:2]
        x0, y0, x1, y1 = (int(value) for value in bbox)
        padding = int(round(max(x1 - x0, y1 - y0) * self.context_fraction))
        x0, y0 = max(0, x0 - padding), max(0, y0 - padding)
        x1, y1 = min(width, x1 + padding), min(height, y1 + padding)

        crop = image_bgr[y0:y1, x0:x1].copy()
        if crop.size == 0:
            raise ValueError("Empty CLIP candidate crop")
        if mask_background:
            # Mid-grey rather than black: black reads as shadow to CLIP.
            crop[~mask[y0:y1, x0:x1]] = 127
        return self.preprocess(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

    def score_regions(self, image_bgr: np.ndarray, bundle,
                      indices: Iterable[int]) -> dict[int, dict[str, Any]]:
        indices = [int(index) for index in indices]
        results: dict[int, dict[str, Any]] = {}

        views = [self.mask_background]
        if self.include_context_view and self.mask_background:
            views.append(False)

        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            tensors = [
                self._crop(image_bgr, bundle.masks[index], bundle.bbox[index], flag)
                for index in batch
                for flag in views
            ]
            images = self.torch.stack(tensors).to(self.device)

            with self.torch.no_grad():
                features = self.model.encode_image(images).float()
                features = features / features.norm(dim=-1, keepdim=True)
                # Average the masked and context views per region, then renormalise.
                features = features.reshape(len(batch), len(views), -1).mean(dim=1)
                features = features / features.norm(dim=-1, keepdim=True)
                similarities = (features @ self.text_features.T).cpu().numpy()

            count = len(self.normal_prompts)
            for index, row in zip(batch, similarities):
                normal, ood = row[:count], row[count:]
                best_normal, best_ood = int(np.argmax(normal)), int(np.argmax(ood))
                margin = float(ood[best_ood] - normal[best_normal])
                if not np.isfinite(margin):
                    raise RuntimeError("CLIP produced a non-finite similarity margin")
                results[index] = {
                    "clip_margin": margin,
                    # Recorded for reading, never used as a decision input.
                    "clip_ood_probability": sigmoid(20.0 * margin),
                    "normal_prompt": self.normal_prompts[best_normal],
                    "ood_prompt": self.ood_prompts[best_ood],
                    "normal_similarity": float(normal[best_normal]),
                    "ood_similarity": float(ood[best_ood]),
                }
        return results


def build_validator(config: Mapping[str, Any], device: str) -> CLIPRegionValidator | None:
    clip_config = config.get("clip", {})
    if not isinstance(clip_config, dict) or not clip_config.get("enabled", False):
        return None
    return CLIPRegionValidator(clip_config, device=device)
