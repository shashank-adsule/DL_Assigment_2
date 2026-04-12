"""
report_2_4_feature_maps.py  —  Section 2.4: Inside the Black Box — Feature Maps

Loads classifier.pth, picks a probe image from the val set, and logs:
  - Feature map grid from encoder.block1 (first conv  → low-level edges)
  - Feature map grid from encoder.block5 (last conv → semantic shapes)
  - Side-by-side comparison figure

Requires:
  checkpoints/classifier.pth   (produced by train.py --task classification)

Run from project root:
    python wandb_reports/report_2_4_feature_maps.py \
        --data_root /path/to/pets \
        --cls_ckpt  checkpoints/classifier.pth
"""

import os, sys, warnings
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import wandb
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from models.classification import PetClassifier
from data.pets_dataset import OxfordPetDataset, collate_fn

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denorm(t):
    return (t * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


def hook_block(block, image):
    """Register a hook on the first Conv2d in a block, return activations."""
    buf = {}
    # block is an nn.Sequential; its first child is _conv_bn_relu → Conv2d
    first_conv = list(block.children())[0][0]  # _conv_bn_relu[0] = Conv2d
    h = first_conv.register_forward_hook(
        lambda m, i, o: buf.update({"feat": o.detach().cpu()}))
    with torch.no_grad():
        block  # model is called below
    return buf, h


def get_feature_maps(model, image, block_name):
    """Run a forward pass and capture the first conv output of block_name."""
    buf = {}
    block = getattr(model.encoder, block_name)
    first_conv = list(block.children())[0][0]
    h = first_conv.register_forward_hook(
        lambda m, i, o: buf.update({"feat": o.detach().cpu()}))
    model.eval()
    with torch.no_grad():
        model(image)
    h.remove()
    return buf["feat"][0]  # (C, H, W)


def feat_to_grid(feat, n=32, nrow=8):
    maps = feat[:n].unsqueeze(1)
    mn   = maps.flatten(2).min(2).values[..., None, None]
    mx   = maps.flatten(2).max(2).values[..., None, None]
    maps = (maps - mn) / (mx - mn + 1e-6)
    grid = make_grid(maps, nrow=nrow, padding=2, pad_value=0.5)
    return grid.permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",     default=r"D:\code\repo\DL_Assigment_2\temp")
    ap.add_argument("--cls_ckpt",      default="checkpoints/classifier.pth")
    ap.add_argument("--n_channels",    type=int, default=32)
    ap.add_argument("--num_workers",   type=int, default=4)
    ap.add_argument("--wandb_project", default="da6401-assignment2")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project=args.wandb_project, name="2.4_feature_maps",
               config=vars(args))

    # Load model
    model = PetClassifier(num_classes=37)
    raw   = torch.load(args.cls_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(raw.get("state_dict", raw))
    model = model.to(device)
    model.eval()
    print(f"Loaded {args.cls_ckpt}")

    # Grab one val image
    va_dl = DataLoader(
        OxfordPetDataset(args.data_root, partition="val", mode="cls"),
        batch_size=16, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn)
    probe = next(iter(va_dl))["image"][:1].to(device)

    # Extract feature maps from block1 (low-level) and block5 (high-level)
    feat1 = get_feature_maps(model, probe, "block1")  # 64 channels, 224×224
    feat5 = get_feature_maps(model, probe, "block5")  # 512 channels, 14×14

    n      = min(args.n_channels, feat1.shape[0], feat5.shape[0])
    grid1  = feat_to_grid(feat1, n)
    grid5  = feat_to_grid(feat5, n)
    orig   = denorm(probe.cpu().squeeze())

    print(f"block1 feat: {tuple(feat1.shape)}   block5 feat: {tuple(feat5.shape)}")

    # Comparison figure
    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 3, wspace=0.04)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(orig); ax0.set_title("Input image", fontsize=13); ax0.axis("off")

    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(grid1[..., 0], cmap="viridis")
    ax1.set_title(f"Block 1 — {n} channels\nedges · colours · gradients", fontsize=11)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(grid5[..., 0], cmap="magma")
    ax2.set_title(f"Block 5 — {n} channels\nsnouts · ears · fur patterns", fontsize=11)
    ax2.axis("off")

    fig.suptitle("Feature maps: low-level (Block 1) vs high-level (Block 5)", fontsize=14)
    plt.tight_layout()

    wandb.log({
        "input_image":               wandb.Image(orig),
        "feature_maps/block1":       wandb.Image(grid1[..., 0],
                                         caption="Block 1 — edges & colours"),
        "feature_maps/block5":       wandb.Image(grid5[..., 0],
                                         caption="Block 5 — semantic shapes"),
        "feature_maps/comparison":   wandb.Image(fig),
    })
    plt.close(fig)

    print("Section 2.4 done.")
    wandb.finish()


if __name__ == "__main__":
    main()