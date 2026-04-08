"""
Evaluation metrics for all three tasks.

  compute_f1_macro     — macro F1 for classification (Task 1)
  compute_iou          — per-sample IoU for detection (Task 2)
  compute_map          — mean Average Precision for detection (Task 2)
  compute_dice         — mean Dice coefficient for segmentation (Task 3)
  compute_pixel_acc    — pixel accuracy for segmentation (Task 3)
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score


# ---------------------------------------------------------------------------
# Task 1: Macro F1
# ---------------------------------------------------------------------------
def compute_f1_macro(all_preds: list, all_labels: list) -> float:
    """
    Compute macro-averaged F1 score.

    Args:
        all_preds  : list of predicted class indices (int)
        all_labels : list of ground-truth class indices (int)
    Returns:
        float F1 score in [0, 1]
    """
    return f1_score(all_labels, all_preds, average="macro", zero_division=0)


# ---------------------------------------------------------------------------
# Task 2: IoU and mAP
# ---------------------------------------------------------------------------
def compute_iou_batch(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU for a batch of predicted and ground-truth boxes.

    Args:
        pred_boxes (Tensor): (N, 4) in [cx, cy, w, h] normalised
        gt_boxes   (Tensor): (N, 4) in [cx, cy, w, h] normalised

    Returns:
        iou (Tensor): (N,)
    """
    # Convert to x1y1x2y2
    def to_corners(b):
        x1 = b[:, 0] - b[:, 2] / 2
        y1 = b[:, 1] - b[:, 3] / 2
        x2 = b[:, 0] + b[:, 2] / 2
        y2 = b[:, 1] + b[:, 3] / 2
        return x1, y1, x2, y2

    px1, py1, px2, py2 = to_corners(pred_boxes)
    gx1, gy1, gx2, gy2 = to_corners(gt_boxes)

    ix1 = torch.max(px1, gx1)
    iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2)
    iy2 = torch.min(py2, gy2)

    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih

    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_g = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    union  = area_p + area_g - inter + 1e-6

    return inter / union


def compute_map(pred_boxes: list, gt_boxes: list,
                iou_thresholds: list = None) -> float:
    """
    Simplified mAP: fraction of predictions with IoU >= threshold,
    averaged over thresholds [0.5, 0.55, ..., 0.95].

    Args:
        pred_boxes : list of (N,4) tensors
        gt_boxes   : list of (N,4) tensors
        iou_thresholds : list of float thresholds (default COCO-style)
    Returns:
        float mAP
    """
    if iou_thresholds is None:
        iou_thresholds = [round(t, 2) for t in np.arange(0.5, 1.0, 0.05)]

    all_preds = torch.cat(pred_boxes, dim=0)
    all_gts   = torch.cat(gt_boxes,   dim=0)
    ious      = compute_iou_batch(all_preds, all_gts)

    aps = []
    for thresh in iou_thresholds:
        ap = (ious >= thresh).float().mean().item()
        aps.append(ap)
    return float(np.mean(aps))


# ---------------------------------------------------------------------------
# Task 3: Dice score and pixel accuracy
# ---------------------------------------------------------------------------
def compute_dice(pred_logits: torch.Tensor, targets: torch.Tensor,
                 num_classes: int = 3, eps: float = 1e-6) -> float:
    """
    Mean Dice coefficient across all classes.

    Args:
        pred_logits (Tensor): (B, C, H, W) — raw model output
        targets     (Tensor): (B, H, W)    — integer class indices
        num_classes (int)
        eps         (float): smoothing constant

    Returns:
        float mean Dice in [0, 1]
    """
    preds = pred_logits.argmax(dim=1)              # (B, H, W)

    dice_scores = []
    for c in range(num_classes):
        pred_c = (preds   == c).float()
        true_c = (targets == c).float()
        intersection = (pred_c * true_c).sum()
        denom        = pred_c.sum() + true_c.sum() + eps
        dice_scores.append((2.0 * intersection / denom).item())

    return float(np.mean(dice_scores))


def compute_pixel_acc(pred_logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Pixel-wise accuracy.

    Args:
        pred_logits (Tensor): (B, C, H, W)
        targets     (Tensor): (B, H, W)

    Returns:
        float accuracy in [0, 1]
    """
    preds   = pred_logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total   = targets.numel()
    return correct / total
