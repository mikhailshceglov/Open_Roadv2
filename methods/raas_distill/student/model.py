"""The student: SegFormer-B0 with one extra output channel for anomaly.

Rather than bolting a second decoder onto the encoder, the anomaly logit is
simply channel 19 of a 20-channel classifier.  Channels 0..18 keep their
pretrained Cityscapes weights, channel 19 starts fresh.  Everything upstream is
shared, which is the point: the anomaly score the teacher produces is a
function of the same semantics the first 19 channels predict.
"""

import os
from pathlib import Path

import torch
from torch import nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation

DEFAULT_BACKBONE = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
# Vendored so that loading a trained checkpoint needs no network at all: the
# pretrained weights would be downloaded only to be overwritten a moment later.
LOCAL_CONFIG = Path(__file__).resolve().parent / "segformer_b0_cityscapes.json"
NUM_SEMANTIC_CLASSES = 19
ANOMALY_CHANNEL = NUM_SEMANTIC_CLASSES
NUM_OUTPUT_CHANNELS = NUM_SEMANTIC_CLASSES + 1


class Student(nn.Module):
    def __init__(self, backbone: str = None, pretrained: bool = True):
        """``pretrained=False`` builds the same architecture from the vendored
        config without touching the network -- the right choice when a trained
        checkpoint is about to replace every weight anyway."""
        super().__init__()
        backbone = backbone or os.environ.get("STUDENT_BACKBONE", DEFAULT_BACKBONE)
        if pretrained:
            model = SegformerForSemanticSegmentation.from_pretrained(backbone)
        else:
            model = SegformerForSemanticSegmentation(SegformerConfig.from_json_file(LOCAL_CONFIG))

        classifier = model.decode_head.classifier
        if classifier.out_channels != NUM_SEMANTIC_CLASSES:
            raise ValueError(
                "Expected a {}-class Cityscapes checkpoint, got {} classes".format(
                    NUM_SEMANTIC_CLASSES, classifier.out_channels
                )
            )

        widened = nn.Conv2d(
            classifier.in_channels,
            NUM_OUTPUT_CHANNELS,
            kernel_size=classifier.kernel_size,
            stride=classifier.stride,
            padding=classifier.padding,
        )
        with torch.no_grad():
            widened.weight[:NUM_SEMANTIC_CLASSES] = classifier.weight
            widened.bias[:NUM_SEMANTIC_CLASSES] = classifier.bias
            # Start the anomaly channel at a strong negative bias so the first
            # steps predict "not anomalous" everywhere; anomalous pixels are
            # well under 1% of the corpus and a zero-init head spends its early
            # capacity unlearning a 50/50 prior.
            widened.weight[ANOMALY_CHANNEL].normal_(0.0, 0.01)
            widened.bias[ANOMALY_CHANNEL] = -4.0
        model.decode_head.classifier = widened
        model.config.num_labels = NUM_OUTPUT_CHANNELS

        self.model = model

    def forward(self, pixel_values):
        """-> (semantic logits [B,19,h,w], anomaly logit [B,1,h,w]) at 1/4 scale."""
        logits = self.model(pixel_values=pixel_values).logits
        return logits[:, :NUM_SEMANTIC_CLASSES], logits[:, ANOMALY_CHANNEL : ANOMALY_CHANNEL + 1]


class AnomalyOnly(nn.Module):
    """Export wrapper: full-resolution anomaly probability, nothing else.

    ONNX consumers want one map at input resolution, not a 20-channel tensor at
    1/4 scale, and the upsample belongs inside the graph so the runtime does it
    rather than Python.
    """

    def __init__(self, student: Student):
        super().__init__()
        self.student = student

    def forward(self, pixel_values):
        _, anomaly = self.student(pixel_values)
        anomaly = torch.nn.functional.interpolate(
            anomaly, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.sigmoid(anomaly)
