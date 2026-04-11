import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers    import CustomDropout, IoULoss
from .segmentation import DoubleConv, DiceCELoss


class MultiTaskVGG11(nn.Module):

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

    def forward(self, x: torch.Tensor) -> dict:
        # ---- Shared encoder ----
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        s4 = self.enc4(self.pool3(s3))
        s5 = self.enc5(self.pool4(s4))
        pooled = self.pool5(s5)         # (B, 512, H/32, W/32)

        # ---- Classification and localization use global-pooled features ----
        global_feat = torch.flatten(self.avgpool(s5), 1)   # (B, 25088)

        cls_logits = self.cls_head(global_feat)             # (B, num_classes)
        bbox       = self.loc_head(global_feat)             # (B, 4)

        # ---- Segmentation decoder ----
        d = self.bottleneck(pooled)
        d = self.dec5(torch.cat([self.up5(d), s5], dim=1))
        d = self.dec4(torch.cat([self.up4(d), s4], dim=1))
        d = self.dec3(torch.cat([self.up3(d), s3], dim=1))
        d = self.dec2(torch.cat([self.up2(d), s2], dim=1))
        d = self.dec1(torch.cat([self.up1(d), s1], dim=1))
        seg_logits = self.seg_out(d)                        # (B, num_seg_classes, H, W)

        # Return as dict — autograder expects these exact keys
        return {
            'classification': cls_logits,
            'localization':   bbox,
            'segmentation':   seg_logits,
        }


# ---------------------------------------------------------------------------
# Multi-task loss
# ---------------------------------------------------------------------------
class MultiTaskLoss(nn.Module):

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


# Autograder imports this exact name:
#   from models.multitask import MultiTaskPerceptionModel
MultiTaskPerceptionModel = MultiTaskVGG11