import torch
import torch.nn as nn
from .layers import CustomDropout

class VGG11Encoder(nn.Module):
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
        return self.features


# Backward-compatibility alias — autograder imports VGG11Encoder,
# other internal code may still use VGG11.
VGG11 = VGG11Encoder