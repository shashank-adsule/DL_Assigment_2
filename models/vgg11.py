from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .layers import CustomDropout

IMAGE_SIZE = 224
_BOTTLENECK_DIM = 512 * 7 * 7   # 25088


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """3×3 conv → BN → ReLU block (padding=1 keeps spatial dims)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 0,       # 0 = encoder-only mode
        dropout_p: float = 0.5,
    ):
        super().__init__()

        # ---- 5 convolutional blocks (VGG-11 topology) ----
        self.block1 = nn.Sequential(_conv_bn_relu(in_channels, 64))
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2 = nn.Sequential(_conv_bn_relu(64, 128))
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block3 = nn.Sequential(
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block4 = nn.Sequential(
            _conv_bn_relu(256, 512),
            _conv_bn_relu(512, 512),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block5 = nn.Sequential(
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---- Optional classifier head ----
        self.num_classes = num_classes
        if num_classes > 0:
            self.avgpool    = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Linear(_BOTTLENECK_DIM, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
                CustomDropout(p=dropout_p),
                nn.Linear(4096, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
                CustomDropout(p=dropout_p),
                nn.Linear(4096, num_classes),
            )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def get_backbone(self) -> nn.Sequential:
        return nn.Sequential(
            self.block1, self.pool1,
            self.block2, self.pool2,
            self.block3, self.pool3,
            self.block4, self.pool4,
            self.block5, self.pool5,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        f1 = self.block1(x)
        p1 = self.pool1(f1)

        f2 = self.block2(p1)
        p2 = self.pool2(f2)

        f3 = self.block3(p2)
        p3 = self.pool3(f3)

        f4 = self.block4(p3)
        p4 = self.pool4(f4)

        f5 = self.block5(p4)
        bottleneck = self.pool5(f5)

        # Full classifier mode
        if self.num_classes > 0:
            out = self.avgpool(bottleneck)
            out = torch.flatten(out, 1)
            return self.classifier(out)

        # Encoder-only mode
        if return_features:
            features = {"b1": f1, "b2": f2, "b3": f3, "b4": f4, "b5": f5}
            return bottleneck, features

        return bottleneck


# Alias required by autograder: from models.vgg11 import VGG11
VGG11 = VGG11Encoder