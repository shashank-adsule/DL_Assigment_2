"""
VGG11 backbone with BatchNorm and CustomDropout.

Architectural decisions
-----------------------
BatchNorm placement  : after Conv2d, before ReLU.
  Reason: normalising pre-activation values prevents the ReLU from receiving
  inputs with arbitrarily large scale, which would cause dead neurons or
  saturated activations and slow convergence.

Dropout placement    : only in the fully-connected classifier head, NOT in
  convolutional blocks.
  Reason 1 – spatial correlation: adjacent pixels in a feature map are highly
  correlated, so dropping individual values gives weak regularisation compared
  to dropping in FC layers where each unit carries independent information.
  Reason 2 – reuse: the convolutional backbone is reused as an encoder in
  Tasks 2, 3, and 4. Keeping it dropout-free produces more stable feature
  representations for downstream heads.

Dropout after ReLU in FC layers: we mask neurons that are already active
  (positive), which is a stronger signal than masking pre-activation values.
"""

import torch
import torch.nn as nn
from .layers import CustomDropout


class VGG11Encoder(nn.Module):
    """
    VGG11 with BatchNorm, from scratch.

    The convolutional feature extractor follows the standard VGG11 topology:
      Block 1: 1 conv,  64 channels
      Block 2: 1 conv, 128 channels
      Block 3: 2 convs, 256 channels
      Block 4: 2 convs, 512 channels
      Block 5: 2 convs, 512 channels
    Each block ends with MaxPool2d(2, 2).

    Input  : (B, 3, 224, 224)
    Output : (B, num_classes) logits
    """

    def __init__(self, num_classes: int = 37, dropout_p: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # ---- Block 1: 3 → 64, 224×224 → 112×112 ----
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 112×112

            # ---- Block 2: 64 → 128, 112×112 → 56×56 ----
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 56×56

            # ---- Block 3: 128 → 256 → 256, 56×56 → 28×28 ----
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 28×28

            # ---- Block 4: 256 → 512 → 512, 28×28 → 14×14 ----
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 14×14

            # ---- Block 5: 512 → 512 → 512, 14×14 → 7×7 ----
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 7×7
        )

        # AdaptiveAvgPool makes the model robust to inputs != 224×224
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # ---- Classifier ----
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming Normal for conv layers; zero-bias throughout."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)       # (B, 512, 7, 7)
        x = self.avgpool(x)        # (B, 512, 7, 7)  — no-op if input was 224
        x = torch.flatten(x, 1)   # (B, 25088)
        x = self.classifier(x)    # (B, num_classes)
        return x

    def get_backbone(self) -> nn.Sequential:
        """Return only the convolutional feature extractor (for Task 2 & 3)."""
        return self.features


# Backward-compatibility alias — autograder imports VGG11Encoder,
# other internal code may still use VGG11.
VGG11 = VGG11Encoder