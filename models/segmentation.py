"""
Task 3: U-Net Style Semantic Segmentation.

Architecture
------------
Encoder  : VGG11 convolutional backbone split into 5 blocks.
           Skip connections are extracted after each block's ReLU,
           before MaxPool, so spatial resolution is retained.

Decoder  : Symmetric expansive path using ConvTranspose2d for learnable
           upsampling (bilinear interpolation is NOT used per spec).
           At each decoder stage the upsampled feature map is concatenated
           with the spatially-aligned encoder skip map (U-Net fusion).

Output   : Per-pixel logits over 3 classes (foreground / background / border)
           matching the Oxford Pet trimap annotation format.

Loss function choice
--------------------
Combined Dice + CrossEntropy loss.

Justification:
  - CrossEntropy provides strong per-pixel gradient signal everywhere and
    is well-calibrated during early training.
  - Dice loss optimises directly for the overlap metric used in evaluation
    and is robust to class imbalance (the background class dominates the
    trimap pixel count).
  - Their combination stabilises training: CE warms up quickly, Dice
    refines the segmentation boundary.

Upsampling justification
------------------------
ConvTranspose2d (transposed/fractional convolution) learns its own
upsampling kernel and is back-propagable end-to-end. This allows the
decoder to learn task-specific upsampling — e.g. preserving sharp
object edges — rather than using a fixed interpolation algorithm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: a double-conv block used in the decoder
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# Combined Dice + CrossEntropy loss
# ---------------------------------------------------------------------------
class DiceCELoss(nn.Module):
    """
    Combined Dice + Cross-Entropy loss for multi-class segmentation.

    Args:
        num_classes (int): number of output classes
        dice_weight (float): weight applied to the Dice component
        ce_weight   (float): weight applied to the CE component
        eps         (float): Dice smoothing constant
    """

    def __init__(self, num_classes=3, dice_weight=0.5, ce_weight=0.5, eps=1e-6):
        super().__init__()
        self.num_classes  = num_classes
        self.dice_weight  = dice_weight
        self.ce_weight    = ce_weight
        self.eps          = eps
        self.ce           = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  (Tensor): (B, C, H, W) — raw unnormalised scores
            targets (Tensor): (B, H, W)    — integer class indices
        """
        ce_loss = self.ce(logits, targets)

        # Dice over softmax probabilities
        probs   = F.softmax(logits, dim=1)                  # (B, C, H, W)
        targets_oh = F.one_hot(targets, self.num_classes)   # (B, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float() # (B, C, H, W)

        dims = (0, 2, 3)  # average over batch and spatial dims
        intersection = (probs * targets_oh).sum(dim=dims)
        cardinality  = probs.sum(dim=dims) + targets_oh.sum(dim=dims)
        dice_per_class = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        dice_loss = 1.0 - dice_per_class.mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


# ---------------------------------------------------------------------------
# U-Net segmentation model
# ---------------------------------------------------------------------------
class UNetVGG11(nn.Module):
    """
    U-Net with VGG11 encoder.

    The VGG11 features Sequential is split manually into 5 encoder blocks
    so we can extract skip-connection tensors at each stage.

    Encoder channel progression : 64 → 128 → 256 → 512 → 512
    Decoder channel progression : mirrors encoder in reverse.

    Args:
        num_classes     (int):  number of segmentation classes (default 3)
        freeze_encoder  (bool): if True, encoder weights are not updated
    """

    def __init__(self, num_classes: int = 3, freeze_encoder: bool = False):
        super().__init__()

        # ---- Encoder blocks (mirrors VGG11.features) ----
        # Each enc_blockN outputs a skip feature map BEFORE pooling.
        self.enc1 = nn.Sequential(          # 3 → 64,  224→224
            nn.Conv2d(3,   64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2, 2)    # → 112

        self.enc2 = nn.Sequential(          # 64 → 128, 112→112
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2, 2)    # → 56

        self.enc3 = nn.Sequential(          # 128 → 256, 56→56
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2, 2)    # → 28

        self.enc4 = nn.Sequential(          # 256 → 512, 28→28
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(2, 2)    # → 14

        self.enc5 = nn.Sequential(          # 512 → 512, 14→14
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool2d(2, 2)    # → 7

        # Bottleneck
        self.bottleneck = DoubleConv(512, 512)

        # ---- Decoder: ConvTranspose2d for learnable upsampling ----
        # Up-block: TransposedConv doubles spatial res, then DoubleConv refines.
        # Skip concat doubles channels before DoubleConv.

        self.up5   = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)  # 7 → 14
        self.dec5  = DoubleConv(512 + 512, 512)

        self.up4   = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 14 → 28
        self.dec4  = DoubleConv(256 + 512, 256)

        self.up3   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 28 → 56
        self.dec3  = DoubleConv(128 + 256, 128)

        self.up2   = nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2)  # 56 → 112
        self.dec2  = DoubleConv(64  + 128, 64)

        self.up1   = nn.ConvTranspose2d(64,  32,  kernel_size=2, stride=2)  # 112 → 224
        self.dec1  = DoubleConv(32  + 64,  32)

        # 1×1 output head
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        if freeze_encoder:
            for block in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]:
                for param in block.parameters():
                    param.requires_grad = False

    def load_encoder_from_vgg11(self, vgg11_state_dict: dict):
        """
        Initialise encoder weights from a trained VGG11 state dict.
        Keys are remapped from vgg11.features.* to enc*.* naming.
        """
        # Map: features index → encoder block attribute
        # VGG11.features indices: see vgg11.py
        block_map = {
            # (start_idx, end_idx) in features → attribute name
            (0,  3):  "enc1",   # conv1 + bn + relu
            (4,  7):  "enc2",   # conv2 + bn + relu
            (8,  14): "enc3",   # 2× conv+bn+relu
            (15, 21): "enc4",
            (22, 28): "enc5",
        }
        features_state = {
            k.replace("features.", ""): v
            for k, v in vgg11_state_dict.items()
            if k.startswith("features.")
        }

        for (start, end), attr in block_map.items():
            block: nn.Sequential = getattr(self, attr)
            sub_keys = {
                str(i - start): v
                for i, v in features_state.items()
                if isinstance(i, int) and start <= i < end
            }
            block.load_state_dict(sub_keys, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): (B, 3, H, W)
        Returns:
            logits (Tensor): (B, num_classes, H, W)
        """
        # ---- Encoder + skip connections ----
        s1 = self.enc1(x)       # (B, 64,  H,   W)
        s2 = self.enc2(self.pool1(s1))   # (B, 128, H/2, W/2)
        s3 = self.enc3(self.pool2(s2))   # (B, 256, H/4, W/4)
        s4 = self.enc4(self.pool3(s3))   # (B, 512, H/8, W/8)
        s5 = self.enc5(self.pool4(s4))   # (B, 512, H/16,W/16)

        x  = self.bottleneck(self.pool5(s5))  # (B, 512, H/32, W/32)

        # ---- Decoder with skip fusion ----
        x = self.dec5(torch.cat([self.up5(x), s5], dim=1))
        x = self.dec4(torch.cat([self.up4(x), s4], dim=1))
        x = self.dec3(torch.cat([self.up3(x), s3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))

        return self.out_conv(x)  # (B, num_classes, H, W)
