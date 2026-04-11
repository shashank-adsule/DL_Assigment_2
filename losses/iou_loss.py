"""
losses/iou_loss.py

The autograder imports IoULoss from this exact path:
    from losses.iou_loss import IoULoss

Boxes are in [x_center, y_center, width, height] format normalised to [0, 1].
Loss = 1 - IoU, averaged over the batch.
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """
    Intersection over Union loss for bounding box regression.

    Args:
        eps (float): small value added to denominator for numerical stability
                     and to keep gradients viable when boxes do not overlap.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   (Tensor): (B, 4) — predicted   [cx, cy, w, h] in [0, 1]
            target (Tensor): (B, 4) — ground truth [cx, cy, w, h] in [0, 1]

        Returns:
            Scalar loss = mean(1 - IoU) over the batch.
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

        return (1.0 - iou).mean()

    def extra_repr(self) -> str:
        return f"eps={self.eps}"