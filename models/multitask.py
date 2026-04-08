"""
Task 4: Unified Multi-Task Learning Pipeline.

A single model with:
  - Shared VGG11 convolutional backbone
  - Three task-specific heads branching from the backbone:
      1. Classification head  → 37-class logits
      2. Localization head    → [cx, cy, w, h] bounding box
      3. Segmentation decoder → per-pixel class map (trimap)

Single forward pass yields all three outputs simultaneously.

Multi-task loss
---------------
Total loss = λ_cls * CE_loss + λ_loc * IoU_loss + λ_seg * DiceCE_loss

The λ weights default to equal (1/3 each). They can be tuned as
hyperparameters if one task dominates during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers    import CustomDropout, IoULoss
from .segmentation import DoubleConv, DiceCELoss


class MultiTaskVGG11(nn.Module):
    """
    Unified multi-task model built on a shared VGG11 encoder.

    Args:
        num_classes   (int): number of breed classes (default 37)
        num_seg_classes (int): segmentation classes (default 3 for trimap)
        dropout_p     (float): dropout prob in classification head
    """

    def __init__(
        self,
        num_classes: int = 37,
        num_seg_classes: int = 3,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        # ============================================================
        # Shared encoder — split into 5 addressable blocks so the
        # segmentation decoder can access skip-connection feature maps.
        # ============================================================
        self.enc1 = nn.Sequential(
            nn.Conv2d(3,   64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool2d(2, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # ============================================================
        # Head 1: Classification
        # ============================================================
        self.cls_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

        # ============================================================
        # Head 2: Localization (bounding-box regression)
        # ============================================================
        self.loc_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

        # ============================================================
        # Head 3: Segmentation decoder (U-Net style)
        # ============================================================
        self.bottleneck = DoubleConv(512, 512)

        self.up5  = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec5 = DoubleConv(512 + 512, 512)

        self.up4  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(256 + 512, 256)

        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(128 + 256, 128)

        self.up2  = nn.ConvTranspose2d(128, 64,  2, stride=2)
        self.dec2 = DoubleConv(64  + 128, 64)

        self.up1  = nn.ConvTranspose2d(64,  32,  2, stride=2)
        self.dec1 = DoubleConv(32  + 64,  32)

        self.seg_out = nn.Conv2d(32, num_seg_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        Single forward pass returning all three task outputs.

        Args:
            x (Tensor): (B, 3, H, W)

        Returns:
            cls_logits (Tensor): (B, 37)          — breed classification
            bbox       (Tensor): (B, 4)           — [cx, cy, w, h] in [0,1]
            seg_logits (Tensor): (B, C, H, W)     — segmentation map
        """
        # ---- Shared encoder ----
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))
        s5 = self.enc5(self.pool4(s4))
        pooled = self.pool5(s5)         # (B, 512, H/32, W/32)

        # ---- Classification and localization use global-pooled features ----
        global_feat = torch.flatten(self.avgpool(s5), 1)   # (B, 25088)

        cls_logits = self.cls_head(global_feat)             # (B, 37)
        bbox       = self.loc_head(global_feat)             # (B, 4)

        # ---- Segmentation decoder ----
        d = self.bottleneck(pooled)
        d = self.dec5(torch.cat([self.up5(d), s5], dim=1))
        d = self.dec4(torch.cat([self.up4(d), s4], dim=1))
        d = self.dec3(torch.cat([self.up3(d), s3], dim=1))
        d = self.dec2(torch.cat([self.up2(d), s2], dim=1))
        d = self.dec1(torch.cat([self.up1(d), s1], dim=1))
        seg_logits = self.seg_out(d)                        # (B, C, H, W)

        return cls_logits, bbox, seg_logits


# ---------------------------------------------------------------------------
# Multi-task loss
# ---------------------------------------------------------------------------
class MultiTaskLoss(nn.Module):
    """
    Weighted sum of the three task losses.

    Args:
        lambda_cls (float): weight for classification CE loss
        lambda_loc (float): weight for IoU localization loss
        lambda_seg (float): weight for Dice+CE segmentation loss
    """

    def __init__(
        self,
        num_seg_classes: int = 3,
        lambda_cls: float = 1.0,
        lambda_loc: float = 1.0,
        lambda_seg: float = 1.0,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_loc = lambda_loc
        self.lambda_seg = lambda_seg

        self.ce_loss    = nn.CrossEntropyLoss()
        self.iou_loss   = IoULoss()
        self.dicece     = DiceCELoss(num_classes=num_seg_classes)

    def forward(
        self,
        cls_logits: torch.Tensor,
        bbox_pred:  torch.Tensor,
        seg_logits: torch.Tensor,
        cls_target: torch.Tensor,
        bbox_target:torch.Tensor,
        seg_target: torch.Tensor,
    ):
        loss_cls = self.ce_loss(cls_logits, cls_target)
        loss_loc = self.iou_loss(bbox_pred, bbox_target)
        loss_seg = self.dicece(seg_logits, seg_target)

        total = (
            self.lambda_cls * loss_cls
            + self.lambda_loc * loss_loc
            + self.lambda_seg * loss_seg
        )
        return total, {"cls": loss_cls.item(), "loc": loss_loc.item(), "seg": loss_seg.item()}
