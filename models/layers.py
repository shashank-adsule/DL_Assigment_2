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

        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(torch.full_like(x, keep_prob))
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"p={self.p}"


class IoULoss(nn.Module):
    """
    Intersection over Union loss for bounding box regression.

    Args:
        reduction (str): 'mean' | 'sum' | 'none'. Default: 'mean'
        eps (float): numerical stability constant.
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-6):
        super().__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_x1 = pred[:, 0] - pred[:, 2] / 2
        pred_y1 = pred[:, 1] - pred[:, 3] / 2
        pred_x2 = pred[:, 0] + pred[:, 2] / 2
        pred_y2 = pred[:, 1] + pred[:, 3] / 2

        tgt_x1 = target[:, 0] - target[:, 2] / 2
        tgt_y1 = target[:, 1] - target[:, 3] / 2
        tgt_x2 = target[:, 0] + target[:, 2] / 2
        tgt_y2 = target[:, 1] + target[:, 3] / 2

        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        intersection = inter_w * inter_h

        area_pred = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        area_tgt  = (tgt_x2  - tgt_x1).clamp(min=0)  * (tgt_y2  - tgt_y1).clamp(min=0)
        union = area_pred + area_tgt - intersection + self.eps

        iou  = intersection / union
        loss = 1.0 - iou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}', eps={self.eps}"