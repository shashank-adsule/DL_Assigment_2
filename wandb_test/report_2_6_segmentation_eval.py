"""
report_2_6_segmentation_eval.py  —  Section 2.6: Dice vs Pixel Accuracy

Loads UNetVGG11 from unet.pth and:
  1. Logs 5 triplet panels: [Original | GT trimap | Predicted trimap]
  2. Tracks both Pixel Accuracy AND Dice Score epoch-by-epoch on the val set
  3. Shows naive "predict all background" baseline to prove pixel accuracy
     is misleading for imbalanced segmentation
  4. Logs a bar-chart comparison with actual numbers visible

Fixes vs previous version:
  - correct import: data.pets_dataset (not data.dataset)
  - tracks metrics per epoch so W&B shows the divergence over time
  - baseline comparison clearly shown with numbers on bar chart

Requires:
  checkpoints/unet.pth   (produced by train.py --task segmentation)

Run from project root:
    python wandb_reports/report_2_6_segmentation_eval.py
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
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.segmentation import UNetVGG11, DiceCELoss
from data.dataset import OxfordPetDataset, collate_fn

PALETTE = np.array([[255, 128, 0],   # 0 = foreground (orange)
                     [70, 130, 180],  # 1 = background (steel blue)
                     [255, 255, 0]],  # 2 = boundary   (yellow)
                    dtype=np.uint8)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denorm(t):
    return (t * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()

def mask_to_rgb(mask_tensor):
    return PALETTE[mask_tensor.cpu().numpy().clip(0, 2)]

def compute_macro_dice(logits, masks, ignore=-1):
    preds = logits.argmax(1); valid = masks != ignore
    total, count = 0.0, 0
    for c in range(logits.size(1)):
        tp = ((preds==c)&(masks==c)&valid).sum().item()
        fp = ((preds==c)&(masks!=c)&valid).sum().item()
        fn = ((preds!=c)&(masks==c)&valid).sum().item()
        d  = 2*tp+fp+fn
        if d > 0: total += 2*tp/d; count += 1
    return total / max(count, 1)

def compute_pixel_acc(logits, masks, ignore=-1):
    preds = logits.argmax(1); valid = masks != ignore
    return ((preds==masks)&valid).sum().item() / max(valid.sum().item(), 1)


def log_triplets(model, loader, device, n=5):
    """Log n panels: Original | Ground Truth Trimap | Predicted Trimap."""
    model.eval(); samples = []
    with torch.no_grad():
        for batch in loader:
            imgs   = batch["image"].to(device)
            masks  = batch["mask"]
            preds  = model(imgs).cpu().argmax(1)
            for i in range(len(imgs)):
                if len(samples) >= n: break
                fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
                ax[0].imshow(denorm(imgs[i].cpu()))
                ax[0].set_title("Original Image"); ax[0].axis("off")
                ax[1].imshow(mask_to_rgb(masks[i]))
                ax[1].set_title("Ground Truth Trimap"); ax[1].axis("off")
                ax[2].imshow(mask_to_rgb(preds[i]))
                ax[2].set_title("Predicted Trimap"); ax[2].axis("off")
                plt.tight_layout()
                samples.append(wandb.Image(fig, caption=f"Sample {len(samples)+1}"))
                plt.close(fig)
            if len(samples) >= n: break
    wandb.log({"segmentation_triplets": samples})
    print(f"  Logged {len(samples)} triplet panels.")


def evaluate(model, loader, criterion, device):
    model.eval()
    all_logits, all_masks, total_loss, n = [], [], 0.0, 0
    with torch.no_grad():
        for batch in loader:
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(imgs)
            total_loss += criterion(logits, masks).item() * imgs.size(0)
            n += imgs.size(0)
            all_logits.append(logits.cpu()); all_masks.append(masks.cpu())
    lc = torch.cat(all_logits); mc = torch.cat(all_masks)
    return total_loss/max(n,1), compute_macro_dice(lc,mc), compute_pixel_acc(lc,mc), lc, mc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",        default=r"D:\code\repo\DL_Assigment_2\temp")
    ap.add_argument("--unet_ckpt",        default="checkpoints/unet.pth")
    ap.add_argument("--fine_tune_epochs", type=int, default=10,
                    help="Short fine-tune to generate per-epoch metric curves")
    ap.add_argument("--batch_size",       type=int,   default=8)
    ap.add_argument("--lr",               type=float, default=1e-4)
    ap.add_argument("--num_workers",      type=int,   default=4)
    ap.add_argument("--wandb_project",    default="da6401-assignment2")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project=args.wandb_project, name="2.6_dice_vs_pixel_acc",
               config=vars(args))

    # Load model
    model = UNetVGG11(num_classes=3)
    raw   = torch.load(args.unet_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(raw.get("state_dict", raw))
    model = model.to(device)
    print(f"Loaded {args.unet_ckpt}")

    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
              pin_memory=True, collate_fn=collate_fn)
    tr_dl = DataLoader(OxfordPetDataset(args.data_root, partition="train", mode="seg"),
                       shuffle=True,  **kw)
    va_dl = DataLoader(OxfordPetDataset(args.data_root, partition="val",   mode="seg"),
                       shuffle=False, **kw)
    te_dl = DataLoader(OxfordPetDataset(args.data_root, partition="test",  mode="seg"),
                       shuffle=False, **kw)

    # ----------------------------------------------------------------
    # STEP 1 — 5 triplet panels (original | GT trimap | predicted trimap)
    # ----------------------------------------------------------------
    log_triplets(model, te_dl, device, n=5)

    # ----------------------------------------------------------------
    # STEP 2 — Track Dice AND Pixel Accuracy per epoch
    # Fine-tune for a few epochs so W&B shows both metrics over time.
    # This demonstrates pixel_acc rising quickly (easy to predict background)
    # while dice lags behind (harder to get foreground/boundary right).
    # ----------------------------------------------------------------
    seg_w     = torch.tensor([1.0, 0.8, 3.0], device=device)
    ce_fn     = nn.CrossEntropyLoss(ignore_index=-1, weight=seg_w)
    dice_fn   = DiceCELoss(num_classes=3, ignore_index=-1)
    criterion = lambda logits, masks: ce_fn(logits, masks) + dice_fn(logits, masks)
    opt       = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched     = CosineAnnealingLR(opt, T_max=args.fine_tune_epochs)

    print(f"\nTracking metrics for {args.fine_tune_epochs} epochs...")
    for epoch in range(1, args.fine_tune_epochs + 1):
        model.train()
        for batch in tr_dl:
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)
            opt.zero_grad()
            criterion(model(imgs), masks).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        val_loss, dice, pxacc, _, _ = evaluate(model, va_dl, criterion, device)

        # Log both on same step so W&B overlays them — shows the gap clearly
        wandb.log({
            "epoch":                        epoch,
            "val/loss":                     val_loss,
            "val/dice":                     dice,
            "val/pixel_acc":                pxacc,
            "comparison/dice_score":        dice,     # these two keys together
            "comparison/pixel_accuracy":    pxacc,    # show divergence on one chart
        }, step=epoch)

        print(f"  Ep {epoch:02d}  loss={val_loss:.4f}  "
              f"dice={dice:.4f}  pxacc={pxacc:.4f}  "
              f"gap(pxacc-dice)={pxacc-dice:.4f}")

    # ----------------------------------------------------------------
    # STEP 3 — Final evaluation
    # ----------------------------------------------------------------
    print("\nFinal evaluation...")
    _, final_dice, final_pxacc, lc, mc = evaluate(
        model, va_dl, criterion, device)
    print(f"Trained model  — Dice={final_dice:.4f}  PixelAcc={final_pxacc:.4f}")

    # Naive baseline: predict ALL pixels as background (class 1)
    naive = torch.zeros(mc.shape[0], 3, mc.shape[1], mc.shape[2])
    naive[:, 1, :, :] = 10.0
    n_dice  = compute_macro_dice(naive, mc)
    n_pxacc = compute_pixel_acc(naive, mc)
    print(f"Naive baseline — Dice={n_dice:.4f}  PixelAcc={n_pxacc:.4f}")

    wandb.log({
        "final/model_dice":      final_dice,
        "final/model_pixel_acc": final_pxacc,
        "final/baseline_dice":   n_dice,
        "final/baseline_pxacc":  n_pxacc,
    })

    # ----------------------------------------------------------------
    # STEP 4 — Class balance (explains WHY pixel acc is misleading)
    # ----------------------------------------------------------------
    counts    = torch.tensor([(mc == c).float().sum().item() for c in range(3)])
    total     = counts.sum()
    class_pct = {f"class_{c}_pct": (counts[c]/total*100).item() for c in range(3)}
    wandb.log(class_pct)
    print(f"Class balance — "
          f"foreground={class_pct['class_0_pct']:.1f}%  "
          f"background={class_pct['class_1_pct']:.1f}%  "
          f"boundary={class_pct['class_2_pct']:.1f}%")

    # ----------------------------------------------------------------
    # STEP 5 — Bar chart: model vs naive baseline on both metrics
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    x, w    = np.arange(2), 0.32
    labels  = ["Pixel Accuracy", "Dice Score"]
    ax.bar(x - w/2, [final_pxacc, final_dice], w,
           label="Trained UNetVGG11", color="#378ADD")
    ax.bar(x + w/2, [n_pxacc, n_dice], w,
           label="All-background baseline", color="#D85A30")
    for i, (mv, bv) in enumerate(zip([final_pxacc, final_dice],
                                      [n_pxacc, n_dice])):
        ax.text(i-w/2, mv+0.015, f"{mv:.3f}", ha="center",
                fontsize=11, fontweight="bold", color="#185FA5")
        ax.text(i+w/2, bv+0.015, f"{bv:.3f}", ha="center",
                fontsize=11, fontweight="bold", color="#993C1D")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Why Dice is a better metric than Pixel Accuracy\n"
        f"Naive baseline gets PixelAcc={n_pxacc:.3f} but Dice={n_dice:.3f}  "
        f"(background = {class_pct['class_1_pct']:.0f}% of all pixels)",
        fontsize=11)
    ax.legend(fontsize=11)
    plt.tight_layout()
    wandb.log({"dice_vs_pixel_acc_comparison": wandb.Image(fig)})
    plt.close(fig)

    print("\nSection 2.6 done.")
    wandb.finish()


if __name__ == "__main__":
    main()