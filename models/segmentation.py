import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg11 import VGG11Encoder
from .layers import CustomDropout

# ---------------------------------------------------------------------------
# Helper: double conv block used in the decoder
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# Combined Dice + CrossEntropy loss
# ---------------------------------------------------------------------------
class DiceCELoss(nn.Module):

    def __init__(self, num_classes=3, dice_weight=0.5, ce_weight=0.5,
                 eps=1e-6, ignore_index=-1):
        super().__init__()
        self.num_classes  = num_classes
        self.dice_weight  = dice_weight
        self.ce_weight    = ce_weight
        self.eps          = eps
        self.ce           = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)

        # Soft Dice over softmax probabilities
        probs      = F.softmax(logits, dim=1)
        valid      = (targets != -1)
        tgt_clamped = targets.clone()
        tgt_clamped[~valid] = 0
        targets_oh  = F.one_hot(tgt_clamped, self.num_classes).permute(0, 3, 1, 2).float()
        mask        = valid.unsqueeze(1).float()
        probs       = probs * mask
        targets_oh  = targets_oh * mask

        dims         = (0, 2, 3)
        intersection = (probs * targets_oh).sum(dim=dims)
        cardinality  = probs.sum(dim=dims) + targets_oh.sum(dim=dims)
        dice_per_cls = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        dice_loss    = 1.0 - dice_per_cls.mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


# ---------------------------------------------------------------------------
# U-Net decoder block
# ---------------------------------------------------------------------------
class UpBlock(nn.Module):

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.refine   = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Guard ±1 pixel mismatch from odd input dims
        if x.shape[2:] != skip.shape[2:]:
            skip = skip[:, :, : x.shape[2], : x.shape[3]]
        x = torch.cat([x, skip], dim=1)
        return self.refine(x)


# ---------------------------------------------------------------------------
# U-Net style segmentation model
# ---------------------------------------------------------------------------
class UNetVGG11(nn.Module):

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        # ---- Encoder (contracting path) ----
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # ---- Decoder (expansive path) ----
        self.up5 = UpBlock(512, 512, 512)   #   7 → 14
        self.up4 = UpBlock(512, 512, 256)   #  14 → 28
        self.up3 = UpBlock(256, 256, 128)   #  28 → 56
        self.up2 = UpBlock(128, 128,  64)   #  56 → 112
        self.up1 = UpBlock( 64,  64,  32)   # 112 → 224

        self.pre_output  = CustomDropout(p=dropout_p)
        self.output_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        self._init_decoder()

    def _init_decoder(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def load_encoder_from_checkpoint(self, ckpt_path: str) -> None:
        raw = torch.load(ckpt_path, map_location="cpu")
        sd  = raw.get("state_dict", raw)
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items()
                  if k.startswith("encoder.")}
        miss, unexp = self.encoder.load_state_dict(enc_sd, strict=False)
        print(f"  [UNetVGG11] encoder loaded from '{ckpt_path}'  "
              f"missing={len(miss)} unexpected={len(unexp)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        neck, skips = self.encoder(x, return_features=True)

        d = self.up5(neck,  skips["b5"])
        d = self.up4(d,     skips["b4"])
        d = self.up3(d,     skips["b3"])
        d = self.up2(d,     skips["b2"])
        d = self.up1(d,     skips["b1"])

        d = self.pre_output(d)
        return self.output_conv(d)   # [B, num_classes, H, W]


# Aliases
DecoderBlock = UpBlock
VGG11UNet    = UNetVGG11