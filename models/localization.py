import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder, IMAGE_SIZE
from .layers import CustomDropout

_BOTTLENECK_DIM = 512 * 7 * 7   # 25088


class BBoxHead(nn.Module):
    def __init__(self, dropout_p: float = 0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_BOTTLENECK_DIM, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.layers(x)
        # Sigmoid → (0,1), scale to pixel space (0, IMAGE_SIZE)
        return torch.sigmoid(raw) * IMAGE_SIZE


class LocalizationModel(nn.Module):
    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5,
                 freeze_backbone: bool = False):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.head    = BBoxHead(dropout_p=dropout_p)

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, 4] — (cx, cy, w, h) in pixel space."""
        bottleneck = self.encoder(x, return_features=False)
        return self.head(bottleneck)


# Keep old class name as alias so train_task2.py still imports it
RegressionHead = BBoxHead
VGG11Localizer = LocalizationModel