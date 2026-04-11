import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder, VGG11


class LocalizationModel(nn.Module):

    def __init__(self, vgg11: VGG11, freeze_backbone: bool = True):
        super().__init__()

        self.backbone = vgg11.get_backbone()          # nn.Sequential of conv blocks
        self.avgpool  = nn.AdaptiveAvgPool2d((7, 7))

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ---- Regression head ----
        # Input: 512 * 7 * 7 = 25088
        self.reg_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),   # outputs in (0, 1) — matches normalised bbox coords
        )

        self._init_head()

    def _init_head(self):
        for m in self.reg_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.backbone(x)       # (B, 512, 7, 7)
        x = self.avgpool(x)        # (B, 512, 7, 7)
        x = torch.flatten(x, 1)   # (B, 25088)
        x = self.reg_head(x)       # (B, 4)
        return x