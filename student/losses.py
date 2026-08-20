"""Distillation losses.

Follows the usual offline-KD form -- a soft term against the teacher plus a
hard term where exact labels exist -- with two task-specific choices:

* the anomaly term is soft BCE (equivalently, binary KL) rather than MSE,
  because AUPR and FPR@95 score the *ranking* of pixels, not their calibration;
* Dice sits alongside it, because anomalous pixels are 0.1-1% of the corpus and
  BCE alone converges happily to "nothing is anomalous".
"""

import torch
import torch.nn.functional as F

EPS = 1e-6


def soft_bce(logits, targets):
    """Cross-entropy against soft targets in [0, 1]."""
    return F.binary_cross_entropy_with_logits(logits, targets)


def dice_loss(logits, targets, eps: float = 1.0):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    cardinality = probs.sum(dims) + targets.sum(dims)
    return (1.0 - (2.0 * intersection + eps) / (cardinality + eps)).mean()


def semantic_kl(student_logits, teacher_scores, temperature: float = 2.0):
    """KL(teacher || student) over the 19 Cityscapes classes.

    Mask2Former's ``sem_seg`` is a product of class probabilities and mask
    sigmoids, so it is non-negative but does not sum to one.  Normalising it
    per pixel turns it into the distribution the temperature is meant to
    soften; taking its log first would put the temperature on the wrong side.
    """
    teacher = teacher_scores.clamp_min(0)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(EPS)
    if temperature != 1.0:
        teacher = teacher.clamp_min(EPS).log() / temperature
        teacher = teacher.softmax(dim=1)

    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)

    # Every spatial position is its own distribution, so flatten B,H,W into the
    # batch dimension before reducing. Calling kl_div with reduction="batchmean"
    # on a [B,C,H,W] tensor divides by B alone and inflates the loss by H*W --
    # for a 192x192 target grid that is a factor of ~37000, which drowns out
    # every other term.
    classes = student_logits.shape[1]
    student_log_probs = student_log_probs.permute(0, 2, 3, 1).reshape(-1, classes)
    teacher = teacher.permute(0, 2, 3, 1).reshape(-1, classes)

    loss = F.kl_div(student_log_probs, teacher, reduction="batchmean")
    # Standard KD scaling: gradients through log_softmax shrink as 1/T^2.
    return loss * (temperature ** 2)


def distillation_loss(
    semantic_logits,
    anomaly_logits,
    semantic_target,
    anomaly_target,
    weight_bce: float = 1.0,
    weight_dice: float = 1.0,
    weight_semantic: float = 0.5,
    temperature: float = 2.0,
):
    if anomaly_logits.shape[-2:] != anomaly_target.shape[-2:]:
        anomaly_logits = F.interpolate(
            anomaly_logits, size=anomaly_target.shape[-2:], mode="bilinear", align_corners=False
        )
    if semantic_logits.shape[-2:] != semantic_target.shape[-2:]:
        semantic_logits = F.interpolate(
            semantic_logits, size=semantic_target.shape[-2:], mode="bilinear", align_corners=False
        )

    terms = {
        "bce": weight_bce * soft_bce(anomaly_logits, anomaly_target),
        "dice": weight_dice * dice_loss(anomaly_logits, anomaly_target),
        "semantic": weight_semantic * semantic_kl(semantic_logits, semantic_target, temperature),
    }
    terms["total"] = terms["bce"] + terms["dice"] + terms["semantic"]
    return terms
