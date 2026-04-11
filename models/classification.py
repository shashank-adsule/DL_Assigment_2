"""
models/classification.py
-------------------------
Classification head and full VGG-11 classifier.
"""

import torch
import torch.nn as nn

from .vgg11  import VGG11Encoder
from .layers import CustomDropout

_BOTTLENECK_DIM = 512 * 7 * 7   # 25088


class FCHead(nn.Module):
    """
    Fully-connected classification head.

    Layout: Flatten → [Linear → BN1d → ReLU → Dropout] × 2 → Linear
    """

    def __init__(self, num_classes: int = 37, drop_rate: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_BOTTLENECK_DIM, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=drop_rate),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=drop_rate),
            nn.Linear(4096, num_classes),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PetClassifier(nn.Module):
    """Full classifier: VGG11Encoder + FCHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3,
                 drop_rate: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.head    = FCHead(num_classes=num_classes, drop_rate=drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        neck = self.encoder(x, return_features=False)
        return self.head(neck)


# Aliases
ClassificationHead = FCHead
VGG11Classifier    = PetClassifier
