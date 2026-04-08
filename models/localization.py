"""
Task 2: Encoder-Decoder for Object Localization.

Architecture
------------
Encoder : VGG11 convolutional backbone (pretrained weights from Task 1).
Decoder : A lightweight regression head that outputs 4 values:
          [x_center, y_center, width, height] normalised to [0, 1].

Backbone freezing strategy
---------------------------
By default `freeze_backbone=True`.

Justification for freezing:
  The VGG11 backbone was trained on the same pet images and has already
  learned rich spatial features (edges, textures, semantic shapes). Since
  the localisation task operates on the same domain, the frozen features
  are directly useful. Freezing also:
    - Prevents catastrophic forgetting of the classification features
      (which we will reuse in Task 4).
    - Reduces the number of trainable parameters, speeding up convergence
      and lowering the risk of overfitting on the bounding-box labels
      (which are only head bounding boxes — less data than class labels).

  Set `freeze_backbone=False` to fine-tune the whole network end-to-end;
  this is evaluated empirically in the W&B transfer-learning experiment.

Output activation
-----------------
A Sigmoid on the final linear layer maps all four outputs to (0, 1),
which matches the normalised [cx, cy, w, h] coordinate space.
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11


class LocalizationModel(nn.Module):
    """
    Bounding-box regressor built on top of the VGG11 backbone.

    Args:
        vgg11        : A trained VGG11 instance (Task 1 weights).
        freeze_backbone (bool): If True, backbone weights are frozen.
    """

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
        """
        Args:
            x (Tensor): (B, 3, H, W)
        Returns:
            bbox (Tensor): (B, 4)  —  [cx, cy, w, h] in [0, 1]
        """
        x = self.backbone(x)       # (B, 512, 7, 7)
        x = self.avgpool(x)        # (B, 512, 7, 7)
        x = torch.flatten(x, 1)   # (B, 25088)
        x = self.reg_head(x)       # (B, 4)
        return x
