"""
Custom layers: CustomDropout and IoULoss.
Both inherit from nn.Module as required by the autograder.
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """
    Custom Dropout implementation using inverted dropout scaling.

    During training:
      - A binary mask is sampled where each element is 1 with prob (1-p).
      - The output is scaled by 1/(1-p) so the expected magnitude is
        preserved and no scaling is needed at inference time.

    During eval (self.training == False):
      - The input is returned unchanged.

    Args:
        p (float): probability of an element being zeroed. Default: 0.5
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        # Binary mask: Bernoulli(1 - p) for each element
        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(torch.full_like(x, keep_prob))

        # Inverted dropout: scale up to keep expected value constant
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"p={self.p}"


class IoULoss(nn.Module):
    """
    Intersection over Union loss for bounding box regression.

    Boxes are expected in [x_center, y_center, width, height] format,
    normalised to [0, 1] relative to image dimensions.

    Loss = 1 - IoU  (so minimising this maximises overlap)

    A small epsilon is added to the denominator for numerical stability
    and to keep gradients viable when boxes do not overlap at all.

    Args:
        eps (float): small value for numerical stability. Default: 1e-6
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   (Tensor): shape (B, 4) — predicted [cx, cy, w, h]
            target (Tensor): shape (B, 4) — ground truth [cx, cy, w, h]

        Returns:
            Scalar loss averaged over the batch.
        """
        # ---- Convert cx/cy/w/h → x1/y1/x2/y2 ----
        pred_x1 = pred[:, 0] - pred[:, 2] / 2
        pred_y1 = pred[:, 1] - pred[:, 3] / 2
        pred_x2 = pred[:, 0] + pred[:, 2] / 2
        pred_y2 = pred[:, 1] + pred[:, 3] / 2

        tgt_x1 = target[:, 0] - target[:, 2] / 2
        tgt_y1 = target[:, 1] - target[:, 3] / 2
        tgt_x2 = target[:, 0] + target[:, 2] / 2
        tgt_y2 = target[:, 1] + target[:, 3] / 2

        # ---- Intersection ----
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        intersection = inter_w * inter_h

        # ---- Union ----
        area_pred = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        area_tgt  = (tgt_x2  - tgt_x1).clamp(min=0)  * (tgt_y2  - tgt_y1).clamp(min=0)
        union = area_pred + area_tgt - intersection + self.eps

        iou = intersection / union

        # Loss = 1 - IoU, averaged over the batch
        return (1.0 - iou).mean()

    def extra_repr(self) -> str:
        return f"eps={self.eps}"
