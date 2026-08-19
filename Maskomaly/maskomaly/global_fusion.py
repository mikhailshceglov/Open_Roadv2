"""Soft spatial/semantic fusion for global road-scene anomaly candidates.

Normal background classes are attenuated, never hard-masked. Compact SAM
regions can restore the global RAAS score anywhere in the frame (including
the sky) when geometry/contrast or CLIP indicates an OOD object.
"""

from typing import Dict, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np


CITYSCAPES_CLASS_NAMES = (
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle",
)


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-value)))


def _ring_median(score: np.ndarray, mask: np.ndarray, radius: int) -> float:
    kernel_size = 2 * max(1, int(radius)) + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = np.logical_and(dilated, ~mask)
    if not ring.any():
        return float(np.median(score))
    return float(np.median(score[ring]))


def _dominant_class(semantic: np.ndarray, mask: np.ndarray) -> int:
    values = semantic[mask]
    if not values.size:
        return -1
    return int(np.bincount(values.astype(np.int64)).argmax())


class CLIPRegionValidator:
    """Batched OpenAI CLIP comparison of normal and airborne/OOD prompts."""

    def __init__(self, config: Mapping[str, object], device: str = "cuda"):
        try:
            import clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Global fusion CLIP validation is enabled, but OpenAI CLIP is "
                "not installed in the Objectomaly environment. Install with "
                "`python -m pip install "
                "git+https://github.com/openai/CLIP.git@"
                "d50d76daa670286dd6cacf3bcd80b5e4823fc8e1`."
            ) from exc
        self.torch = torch
        self.device = device
        self.preprocess_model, self.preprocess = clip.load(
            str(config.get("model", "ViT-B/32")), device=device, jit=False
        )
        self.preprocess_model.eval()
        self.normal_prompts = list(config.get("normal_prompts", []))
        self.ood_prompts = list(config.get("ood_prompts", []))
        if not self.normal_prompts or not self.ood_prompts:
            raise ValueError("CLIP fusion requires normal_prompts and ood_prompts")
        tokens = clip.tokenize(self.normal_prompts + self.ood_prompts).to(device)
        with torch.no_grad():
            text = self.preprocess_model.encode_text(tokens).float()
            self.text_features = text / text.norm(dim=-1, keepdim=True)
        self.batch_size = int(config.get("batch_size", 16))
        self.context_fraction = float(config.get("context_fraction", 0.15))
        self.mask_background = bool(config.get("mask_background", True))
        self.include_context_view = bool(config.get("include_context_view", True))

    def _crop(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        bbox: Sequence[int],
        mask_background: bool,
    ):
        from PIL import Image

        height, width = image_bgr.shape[:2]
        x0, y0, x1, y1 = [int(value) for value in bbox]
        padding = int(round(max(x1 - x0, y1 - y0) * self.context_fraction))
        x0, y0 = max(0, x0 - padding), max(0, y0 - padding)
        x1, y1 = min(width, x1 + padding), min(height, y1 + padding)
        crop = image_bgr[y0:y1, x0:x1].copy()
        if crop.size == 0:
            raise ValueError("Empty CLIP candidate crop")
        if mask_background:
            local_mask = mask[y0:y1, x0:x1]
            crop[~local_mask] = 127
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return self.preprocess(Image.fromarray(rgb))

    def score_regions(self, image_bgr, bundle, indices: Iterable[int]):
        indices = [int(index) for index in indices]
        results: Dict[int, Mapping[str, object]] = {}
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            view_flags = [self.mask_background]
            if self.include_context_view and self.mask_background:
                view_flags.append(False)
            tensors = []
            for index in batch_indices:
                tensors.extend(
                    self._crop(
                        image_bgr,
                        bundle.masks[index],
                        bundle.bbox[index],
                        mask_background=flag,
                    )
                    for flag in view_flags
                )
            images = self.torch.stack(tensors).to(self.device)
            with self.torch.no_grad():
                features = self.preprocess_model.encode_image(images).float()
                features = features / features.norm(dim=-1, keepdim=True)
                features = features.reshape(
                    len(batch_indices), len(view_flags), -1
                ).mean(dim=1)
                features = features / features.norm(dim=-1, keepdim=True)
                similarities = features @ self.text_features.T
            similarities = similarities.detach().cpu().numpy()
            normal_count = len(self.normal_prompts)
            for index, row in zip(batch_indices, similarities):
                normal = row[:normal_count]
                ood = row[normal_count:]
                normal_idx = int(np.argmax(normal))
                ood_idx = int(np.argmax(ood))
                margin = float(ood[ood_idx] - normal[normal_idx])
                if not np.isfinite(margin):
                    raise RuntimeError("CLIP produced a non-finite similarity margin")
                results[index] = {
                    "clip_margin": margin,
                    "clip_ood_probability": _sigmoid(20.0 * margin),
                    "normal_prompt": self.normal_prompts[normal_idx],
                    "ood_prompt": self.ood_prompts[ood_idx],
                    "normal_similarity": float(normal[normal_idx]),
                    "ood_similarity": float(ood[ood_idx]),
                }
        return results


def build_clip_validator(config: Mapping[str, object], device: str):
    clip_config = config.get("clip", {})
    if not isinstance(clip_config, dict) or not clip_config.get("enabled", False):
        return None
    return CLIPRegionValidator(clip_config, device=device)


def apply_global_fusion(
    calibrated: np.ndarray,
    semantic: np.ndarray,
    image_bgr: np.ndarray,
    bundle,
    config: Mapping[str, object],
    clip_validator: Optional[CLIPRegionValidator] = None,
):
    """Return fused map, protected-candidate mask and JSON-safe diagnostics."""
    score = np.asarray(calibrated, dtype=np.float32)
    semantic = np.asarray(semantic, dtype=np.int16)
    if score.ndim != 2 or semantic.shape != score.shape:
        raise ValueError(
            "Global fusion shape mismatch: score={}, semantic={}".format(
                score.shape, semantic.shape
            )
        )

    class_factors = {
        int(class_id): float(factor)
        for class_id, factor in config.get("class_factors", {}).items()
    }
    factors = np.ones(score.shape, dtype=np.float32)
    for class_id, factor in class_factors.items():
        factors[semantic == class_id] = np.float32(factor)
    fused = score * factors

    road_ids = [int(value) for value in config.get("road_class_ids", [0])]
    road_boost = float(config.get("road_boost", 1.0))
    if road_boost > 1.0:
        road = np.isin(semantic, road_ids)
        fused[road] = np.maximum(fused[road], score[road] * road_boost)

    candidate_config = config.get("candidates", {})
    background_ids = {
        int(value) for value in config.get("background_class_ids", [])
    }
    image_area = float(score.size)
    min_area = float(candidate_config.get("min_area_frac", 0.0))
    max_area = float(candidate_config.get("max_area_frac", 0.03))
    quantile = float(candidate_config.get("score_quantile", 0.9))
    min_score = float(candidate_config.get("min_score", 0.45))
    probe_min_score = float(candidate_config.get("clip_probe_min_score", 0.2))
    min_contrast = float(candidate_config.get("min_contrast", 0.12))
    ring_radius = int(candidate_config.get("ring_radius", 9))
    max_candidates = int(candidate_config.get("max_candidates", 64))
    clip_margin_threshold = float(candidate_config.get("clip_margin_threshold", 0.0))
    clip_candidate_floor = float(candidate_config.get("clip_candidate_floor", 0.65))

    records = []
    for index in range(bundle.n):
        mask = bundle.masks[index]
        area_fraction = float(bundle.area[index]) / image_area if image_area else 0.0
        if not min_area <= area_fraction <= max_area or not mask.any():
            continue
        region_score = float(np.quantile(score[mask], quantile))
        ring_score = _ring_median(score, mask, ring_radius)
        contrast = region_score - ring_score
        dominant_class = _dominant_class(semantic, mask)
        dominant_name = (
            CITYSCAPES_CLASS_NAMES[dominant_class]
            if 0 <= dominant_class < len(CITYSCAPES_CLASS_NAMES)
            else "unknown"
        )
        records.append(
            {
                "index": int(index),
                "bbox": [int(value) for value in bundle.bbox[index]],
                "area_fraction": area_fraction,
                "quality": float(bundle.quality[index]),
                "dominant_class": dominant_class,
                "dominant_class_name": dominant_name,
                "score": region_score,
                "ring_score": ring_score,
                "contrast": contrast,
            }
        )

    records.sort(
        key=lambda item: (item["score"] + max(0.0, item["contrast"]))
        * item["quality"],
        reverse=True,
    )
    records = records[:max_candidates]
    probe_indices = [
        item["index"] for item in records if item["score"] >= probe_min_score
    ]
    clip_scores = (
        clip_validator.score_regions(image_bgr, bundle, probe_indices)
        if clip_validator is not None and probe_indices
        else {}
    )

    protected_mask = np.zeros(score.shape, dtype=bool)
    protected_count = 0
    for item in records:
        item.update(clip_scores.get(item["index"], {}))
        geometric = item["score"] >= min_score and item["contrast"] >= min_contrast
        semantic_object = (
            item["score"] >= min_score
            and item["dominant_class"] not in background_ids
        )
        clip_ood = item.get("clip_margin", float("-inf")) >= clip_margin_threshold
        protected = bool(geometric or semantic_object or clip_ood)
        item["protected"] = protected
        item["reasons"] = [
            name
            for name, enabled in (
                ("contrast", geometric),
                ("semantic_object", semantic_object),
                ("clip_ood", clip_ood),
            )
            if enabled
        ]
        if not protected:
            continue
        mask = bundle.masks[item["index"]]
        protected_mask |= mask
        fused[mask] = np.maximum(fused[mask], score[mask])
        if clip_ood:
            fused[mask] = np.maximum(fused[mask], clip_candidate_floor)
        protected_count += 1

    diagnostics = {
        "candidate_count": len(records),
        "clip_probed_count": len(probe_indices) if clip_validator is not None else 0,
        "protected_count": protected_count,
        "candidates": records,
    }
    return (
        np.ascontiguousarray(np.clip(fused, 0.0, 1.0), dtype=np.float32),
        np.ascontiguousarray(protected_mask),
        diagnostics,
    )
