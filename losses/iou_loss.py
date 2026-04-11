import torch
import torch.nn as nn


class IoULoss(nn.Module):

    def __init__(self, reduction: str = 'mean', eps: float = 1e-6):
        super().__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # ---- Convert cx/cy/w/h  →  x1/y1/x2/y2 ----
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

        iou  = intersection / union
        loss = 1.0 - iou           # per-sample loss, shape (B,)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:   # 'none'
            return loss

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}', eps={self.eps}"  