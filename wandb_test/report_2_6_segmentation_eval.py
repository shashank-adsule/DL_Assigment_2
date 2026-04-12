"""
report_2_6_segmentation_eval.py  —  Section 2.6: Dice vs Pixel Accuracy

Loads UNetVGG11 from unet.pth and:
  1. Logs 5 triplet panels: [Original | GT trimap | Predicted trimap]
  2. Computes model Dice and Pixel Accuracy on the val set
  3. Shows a naive "predict all background" baseline to prove Pixel
     Accuracy is misleading for imbalanced segmentation
  4. Logs a bar-chart comparison

Requires:
  checkpoints/unet.pth   (produced by train.py --task segmentation)

Run from project root:
    python wandb_reports/report_2_6_segmentation_eval.py \
        --data_root /path/to/pets \
        --unet_ckpt checkpoints/unet.pth
"""

import os, sys, warnings
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from models.segmentation import UNetVGG11
from data.dataset import OxfordPetDataset, collate_fn

PALETTE = np.array([[255,128,0],[70,130,180],[255,255,0]], dtype=np.uint8)
MEAN    = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
STD     = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)


def denorm(t):
    return (t * STD + MEAN).clamp(0,1).permute(1,2,0).numpy()


def mask_to_rgb(mask_tensor):
    return PALETTE[mask_tensor.cpu().numpy().clip(0,2)]


def compute_macro_dice(logits, masks, ignore=-1):
    preds = logits.argmax(1); valid = masks != ignore
    total, count = 0.0, 0
    for c in range(logits.size(1)):
        tp = ((preds==c)&(masks==c)&valid).sum().item()
        fp = ((preds==c)&(masks!=c)&valid).sum().item()
        fn = ((preds!=c)&(masks==c)&valid).sum().item()
        d  = 2*tp+fp+fn
        if d > 0:
            total += 2*tp/d; count += 1
    return total / max(count, 1)


def compute_pixel_acc(logits, masks, ignore=-1):
    preds = logits.argmax(1); valid = masks != ignore
    return ((preds == masks) & valid).sum().item() / max(valid.sum().item(), 1)


def log_triplets(model, loader, device, n=5):
    model.eval(); samples = []
    with torch.no_grad():
        for batch in loader:
            imgs  = batch["image"].to(device)
            masks = batch["mask"]
            logits = model(imgs).cpu()
            preds  = logits.argmax(1)
            for i in range(len(imgs)):
                if len(samples) >= n: break
                fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
                ax[0].imshow(denorm(imgs[i].cpu())); ax[0].set_title("Original"); ax[0].axis("off")
                ax[1].imshow(mask_to_rgb(masks[i]));  ax[1].set_title("Ground truth"); ax[1].axis("off")
                ax[2].imshow(mask_to_rgb(preds[i]));  ax[2].set_title("Predicted"); ax[2].axis("off")
                plt.tight_layout()
                samples.append(wandb.Image(fig, caption=f"Sample {len(samples)+1}"))
                plt.close(fig)
            if len(samples) >= n: break
    wandb.log({"segmentation_triplets": samples})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",     default=r"D:\code\repo\DL_Assigment_2\temp")
    ap.add_argument("--unet_ckpt",     default="checkpoints/unet.pth")
    ap.add_argument("--batch_size",    type=int, default=8)
    ap.add_argument("--num_workers",   type=int, default=4)
    ap.add_argument("--wandb_project", default="da6401-assignment2")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project=args.wandb_project, name="2.6_dice_vs_pixel_acc",
               config=vars(args))

    model = UNetVGG11(num_classes=3)
    raw   = torch.load(args.unet_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(raw.get("state_dict", raw))
    model = model.to(device)
    print(f"Loaded {args.unet_ckpt}")

    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
              collate_fn=collate_fn)
    va_dl = DataLoader(OxfordPetDataset(args.data_root, partition="val", mode="seg"),
                       shuffle=False, **kw)
    te_dl = DataLoader(OxfordPetDataset(args.data_root, partition="test", mode="seg"),
                       shuffle=False, **kw)

    # 1. Triplet panels from test set
    log_triplets(model, te_dl, device, n=5)

    # 2. Real model metrics on val set
    model.eval()
    all_logits, all_masks = [], []
    with torch.no_grad():
        for batch in va_dl:
            all_logits.append(model(batch["image"].to(device)).cpu())
            all_masks.append(batch["mask"])
    lc = torch.cat(all_logits); mc = torch.cat(all_masks)

    dice   = compute_macro_dice(lc, mc)
    pxacc  = compute_pixel_acc(lc, mc)
    wandb.log({"val/dice": dice, "val/pixel_acc": pxacc})
    print(f"Model   — Dice={dice:.4f}  PixelAcc={pxacc:.4f}")

    # 3. Naive "all background" baseline
    naive = torch.zeros(mc.shape[0], 3, mc.shape[1], mc.shape[2])
    naive[:, 1, :, :] = 10.0
    n_dice  = compute_macro_dice(naive, mc)
    n_pxacc = compute_pixel_acc(naive, mc)
    wandb.log({"baseline/dice": n_dice, "baseline/pixel_acc": n_pxacc})
    print(f"Baseline— Dice={n_dice:.4f}  PixelAcc={n_pxacc:.4f}")

    # 4. Class balance
    counts = torch.tensor([(mc==c).float().sum().item() for c in range(3)])
    total  = counts.sum()
    wandb.log({f"class_{c}_pct": (counts[c]/total*100).item() for c in range(3)})

    # 5. Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Pixel Accuracy", "Dice Score"]
    x, w   = np.arange(2), 0.32
    ax.bar(x-w/2, [pxacc, dice],   w, label="Trained UNetVGG11", color="#378ADD")
    ax.bar(x+w/2, [n_pxacc, n_dice], w, label="All-background baseline", color="#D85A30")
    for i, (mv, bv) in enumerate(zip([pxacc, dice], [n_pxacc, n_dice])):
        ax.text(i-w/2, mv+0.01, f"{mv:.3f}", ha="center", fontsize=10)
        ax.text(i+w/2, bv+0.01, f"{bv:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.1); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Score")
    ax.set_title("Dice vs Pixel Accuracy — why Dice is the better metric\n"
                 "(naive baseline gets high pixel acc but near-zero Dice)")
    ax.legend(); plt.tight_layout()
    wandb.log({"dice_vs_pixel_acc": wandb.Image(fig)})
    plt.close(fig)

    print("Section 2.6 done.")
    wandb.finish()


if __name__ == "__main__":
    main()