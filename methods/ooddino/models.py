"""The three networks, each loaded on first use.

Nothing here is imported at module scope by ``method.py``: constructing the
MethodSpec must stay free, because ``open-road methods`` builds it just to print
one line. All three take RGB, so callers convert from the BGR the skeleton
hands them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .boxes import Box, Detection


def configure_cache(weights_dir: Optional[Path]) -> Optional[Path]:
    """Point the HuggingFace cache at a directory, without overriding the user's.

    Set ``$OODDINO_WEIGHTS`` or the ``weights_dir`` config key to reuse an
    existing cache; the three checkpoints total about 2 GB and are worth not
    downloading twice.
    """
    if weights_dir is None:
        return None
    resolved = Path(weights_dir).expanduser()
    os.environ.setdefault("HF_HOME", str(resolved))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(resolved / "hub"))
    return resolved


class GroundingDINODetector:
    """Open-vocabulary box proposals from a text prompt."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cpu",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
        self._model.to(self.device).eval()
        self._torch = torch

    def detect(self, image_rgb: np.ndarray, prompt: str) -> list[Detection]:
        from PIL import Image

        self._load()
        height, width = image_rgb.shape[:2]
        inputs = self._processor(
            images=Image.fromarray(image_rgb), text=prompt, return_tensors="pt"
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)

        results = self._post_process(outputs, inputs, height, width)
        if not results:
            return []
        payload = results[0]
        labels = payload.get("text_labels") or payload.get("labels") or []

        detections: list[Detection] = []
        for box, score, label in zip(payload.get("boxes", []), payload.get("scores", []), labels):
            coords = box.tolist() if hasattr(box, "tolist") else list(box)
            detections.append(
                Detection(
                    box=Box(*(int(round(value)) for value in coords)).clip(width, height),
                    label=str(label).strip().rstrip("."),
                    confidence=float(score.item() if hasattr(score, "item") else score),
                )
            )
        return detections

    def _post_process(self, outputs: Any, inputs: dict, height: int, width: int) -> list[dict]:
        kwargs = {
            "threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
            "target_sizes": [(height, width)],
        }
        try:
            return self._processor.post_process_grounded_object_detection(
                outputs, inputs.get("input_ids"), **kwargs
            )
        except TypeError:
            # Older transformers called it box_threshold.
            kwargs["box_threshold"] = kwargs.pop("threshold")
            return self._processor.post_process_grounded_object_detection(
                outputs, inputs.get("input_ids"), **kwargs
            )


class SegFormerLogits:
    """Cityscapes SegFormer returning dense class logits at image resolution."""

    def __init__(
        self,
        model_id: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        device: str = "cpu",
        pad_multiple: int = 32,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.pad_multiple = pad_multiple
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = SegformerForSemanticSegmentation.from_pretrained(self.model_id)
        self._model.to(self.device).eval()
        self._torch = torch

    def logits(self, image_rgb: np.ndarray) -> np.ndarray:
        """``(C, H, W)`` float32 at the frame's own resolution.

        One pass at native scale with reflect padding to the encoder's multiple,
        rather than resizing: the decoder predicts at 1/4 and is upsampled back,
        so resizing the input first would cost detail twice.
        """
        from PIL import Image

        self._load()
        height, width = image_rgb.shape[:2]
        pad_h = (self.pad_multiple - height % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - width % self.pad_multiple) % self.pad_multiple
        padded = (
            np.pad(image_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            if pad_h or pad_w
            else image_rgb
        )

        inputs = self._processor(
            images=Image.fromarray(padded), return_tensors="pt", do_resize=False
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
            resized = self._torch.nn.functional.interpolate(
                outputs.logits,
                size=(padded.shape[0], padded.shape[1]),
                mode="bilinear",
                align_corners=False,
            )
        return resized[0].detach().float().cpu().numpy()[:, :height, :width]


class SAM2Segmenter:
    """Box-prompted SAM 2.1, one mask per box, batched into a single call."""

    def __init__(self, model_id: str = "facebook/sam2.1-hiera-large", device: str = "cpu") -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Sam2Model, Sam2Processor

        self._processor = Sam2Processor.from_pretrained(self.model_id)
        self._model = Sam2Model.from_pretrained(self.model_id)
        self._model.to(self.device).eval()
        self._torch = torch

    def segment(self, image_rgb: np.ndarray, boxes: Sequence[Box]) -> list[np.ndarray]:
        from PIL import Image

        height, width = image_rgb.shape[:2]
        if not boxes:
            return []
        self._load()

        inputs = self._processor(
            images=Image.fromarray(image_rgb),
            input_boxes=[[box.as_list() for box in boxes]],
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs, multimask_output=False)

        masks = self._extract(outputs, inputs, height, width)
        empty = np.zeros((height, width), dtype=bool)
        masks.extend(empty.copy() for _ in range(max(0, len(boxes) - len(masks))))
        return masks[: len(boxes)]

    def _extract(self, outputs: Any, inputs: dict, height: int, width: int) -> list[np.ndarray]:
        sizes = inputs.get("original_sizes", [(height, width)])
        predicted = outputs.pred_masks
        if hasattr(predicted, "cpu"):
            predicted = predicted.cpu()
        try:
            processed = self._processor.post_process_masks(predicted, sizes)
        except TypeError:
            processed = self._processor.post_process_masks(
                predicted, sizes, inputs.get("reshaped_input_sizes")
            )

        array = processed[0]
        array = array.cpu().numpy() if hasattr(array, "cpu") else np.asarray(array)
        if array.ndim == 4:  # (boxes, masks, H, W)
            return [(item[0] if item.ndim == 3 else item) > 0 for item in array]
        if array.ndim == 3:
            return [item > 0 for item in array]
        return [array > 0]
