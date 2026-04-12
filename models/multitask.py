import os

import torch
import torch.nn as nn

from .vgg11          import VGG11Encoder
from .classification import FCHead
from .localization   import BBoxHead
from .segmentation   import UpBlock, DiceCELoss
from .layers         import CustomDropout

_CKPT_DIR = "checkpoints"


def _read_ckpt(path: str) -> dict:
    raw = torch.load(path, map_location="cpu")
    return raw.get("state_dict", raw)


def _strip_prefix(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


class MultiTaskPerceptionModel(nn.Module):
    """
    Single forward-pass model returning classification, localisation,
    and segmentation outputs simultaneously.
    """

    def __init__(
        self,
        num_breeds:  int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        cls_ckpt: str = os.path.join(_CKPT_DIR, "classifier.pth"),
        loc_ckpt: str = os.path.join(_CKPT_DIR, "localizer.pth"),
        seg_ckpt: str = os.path.join(_CKPT_DIR, "unet.pth"),
    ):
        # ----------------------------------------------------------------
        # Step 3 from submission guidelines:
        # Replace the placeholder IDs below with your actual Google Drive
        # file IDs after uploading your checkpoints.
        #
        # How to get the ID:
        #   Drive link → https://drive.google.com/file/d/<ID>/view?usp=sharing
        #   Copy only the <ID> part and paste it below.
        # ----------------------------------------------------------------
        import gdown
        os.makedirs(os.path.dirname(cls_ckpt) or ".", exist_ok=True)
        gdown.download(id="REPLACE_WITH_CLASSIFIER_DRIVE_ID",
                       output=cls_ckpt, quiet=False)
        gdown.download(id="REPLACE_WITH_LOCALIZER_DRIVE_ID",
                       output=loc_ckpt, quiet=False)
        gdown.download(id="REPLACE_WITH_UNET_DRIVE_ID",
                       output=seg_ckpt, quiet=False)

        super().__init__()

        # ---- Three task-specific encoders ----
        self.enc_cls = VGG11Encoder(in_channels=in_channels)
        self.enc_loc = VGG11Encoder(in_channels=in_channels)
        self.enc_seg = VGG11Encoder(in_channels=in_channels)

        # ---- Task heads ----
        self.cls_head = FCHead(num_classes=num_breeds, drop_rate=0.5)
        self.loc_head = BBoxHead(dropout_p=0.5)

        # ---- Segmentation decoder ----
        self.up5 = UpBlock(512, 512, 512)
        self.up4 = UpBlock(512, 512, 256)
        self.up3 = UpBlock(256, 256, 128)
        self.up2 = UpBlock(128, 128,  64)
        self.up1 = UpBlock( 64,  64,  32)
        self.seg_drop = CustomDropout(p=0.5)
        self.seg_proj = nn.Conv2d(32, seg_classes, kernel_size=1)

        self._load_weights(cls_ckpt, loc_ckpt, seg_ckpt)

    # ------------------------------------------------------------------
    def _load_encoder(self, enc: nn.Module, sd: dict, tag: str) -> None:
        enc_sd = _strip_prefix(sd, "encoder.")
        if enc_sd:
            miss, unexp = enc.load_state_dict(enc_sd, strict=False)
            print(f"  [{tag}] encoder: missing={len(miss)} unexpected={len(unexp)}")
        else:
            print(f"  [{tag}] WARNING: no 'encoder.*' keys found.")

    def _load_weights(self, cls_path, loc_path, seg_path) -> None:
        # Classification
        if os.path.isfile(cls_path):
            sd = _read_ckpt(cls_path)
            self._load_encoder(self.enc_cls, sd, "cls")
            head_sd = _strip_prefix(sd, "head.")
            if head_sd:
                self.cls_head.load_state_dict(head_sd, strict=False)
        else:
            print(f"  WARNING: '{cls_path}' not found.")

        # Localisation
        if os.path.isfile(loc_path):
            sd = _read_ckpt(loc_path)
            self._load_encoder(self.enc_loc, sd, "loc")
            head_sd = _strip_prefix(sd, "head.")
            if head_sd:
                self.loc_head.load_state_dict(head_sd, strict=False)
        else:
            print(f"  WARNING: '{loc_path}' not found.")

        # Segmentation
        if os.path.isfile(seg_path):
            sd = _read_ckpt(seg_path)
            self._load_encoder(self.enc_seg, sd, "seg")
            for name in ["up5", "up4", "up3", "up2", "up1"]:
                blk_sd = _strip_prefix(sd, f"{name}.")
                if blk_sd:
                    getattr(self, name).load_state_dict(blk_sd, strict=False)
            proj_sd = _strip_prefix(sd, "output_conv.")
            if proj_sd:
                self.seg_proj.load_state_dict(proj_sd, strict=False)
            print(f"  [seg] decoder loaded from '{seg_path}'.")
        else:
            print(f"  WARNING: '{seg_path}' not found.")

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict:
        """
        Single forward pass.

        Returns:
            {
              'classification': [B, num_breeds]          class logits
              'localization'  : [B, 4]                   (cx,cy,w,h) pixels
              'segmentation'  : [B, seg_classes, H, W]
            }
        """
        # Classification branch
        cls_out = self.cls_head(self.enc_cls(x, return_features=False))

        # Localisation branch
        loc_out = self.loc_head(self.enc_loc(x, return_features=False))

        # Segmentation branch
        neck, skips = self.enc_seg(x, return_features=True)
        d = self.up5(neck,  skips["b5"])
        d = self.up4(d,     skips["b4"])
        d = self.up3(d,     skips["b3"])
        d = self.up2(d,     skips["b2"])
        d = self.up1(d,     skips["b1"])
        d = self.seg_drop(d)
        seg_out = self.seg_proj(d)

        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }


# Alias kept for compatibility
MultiTaskVGG11 = MultiTaskPerceptionModel