"""
train_task3.py — U-Net Segmentation (Task 3)

Runs three encoder-freezing strategies for W&B report section 2.3.

Usage:
    python train_tasks/train_task3.py \
        --data_root /path/to/oxford_pets \
        --cls_ckpt  checkpoints/classifier.pth \
        --strategy all \
        --epochs 50 --batch_size 16
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from sklearn.metrics import f1_score

from models.segmentation import UNetVGG11, DiceCELoss
from data.dataset        import OxfordPetDataset, collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
def apply_strategy(model: UNetVGG11, strategy: str):
    enc = model.encoder
    for p in enc.parameters():
        p.requires_grad = True

    if strategy == "frozen":
        for p in enc.parameters():
            p.requires_grad = False
    elif strategy == "partial":
        for blk in [enc.block1, enc.block2, enc.block3]:
            for p in blk.parameters():
                p.requires_grad = False

    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tt = sum(p.numel() for p in model.parameters())
    print(f"  '{strategy}': {tr:,}/{tt:,} params trainable ({100*tr/tt:.1f}%)")


def save_checkpoint(model, tag, ckpt_dir, epoch, metric):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{tag}.pth")
    torch.save({"state_dict": model.state_dict(),
                "epoch": epoch, "metric": metric}, path)
    return path


# ---------------------------------------------------------------------------
def train_strategy(args, strategy: str):
    run_name = f"seg_{strategy}"
    wandb.init(project=args.wandb_project, name=run_name,
               config={**vars(args), "strategy": strategy}, reinit=True)
    print(f"\nDevice: {DEVICE}  |  Strategy: {strategy}")

    tr_ds = OxfordPetDataset(args.data_root, partition="train", mode="seg")
    va_ds = OxfordPetDataset(args.data_root, partition="val",   mode="seg")
    kw    = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                 pin_memory=True, collate_fn=collate_fn)
    tr_dl = DataLoader(tr_ds, shuffle=True,  **kw)
    va_dl = DataLoader(va_ds, shuffle=False, **kw)

    model = UNetVGG11(num_classes=3).to(DEVICE)

    if os.path.isfile(args.cls_ckpt):
        model.load_encoder_from_checkpoint(args.cls_ckpt)

    apply_strategy(model, strategy)

    seg_w   = torch.tensor([1.0, 0.8, 3.0], device=DEVICE)
    ce_fn   = nn.CrossEntropyLoss(ignore_index=-1, weight=seg_w)
    dice_fn = DiceCELoss(num_classes=3, ignore_index=-1)

    params    = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs)

    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        tr_loss = tr_n = 0
        for batch in tr_dl:
            imgs  = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            optimiser.zero_grad()
            logits = model(imgs)
            loss   = ce_fn(logits, masks) + dice_fn(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            tr_loss += loss.item() * imgs.size(0)
            tr_n    += imgs.size(0)
        scheduler.step()
        tr_loss /= max(tr_n, 1)

        # ---- Validate ----
        model.eval()
        va_loss  = va_n = 0.0
        dice_acc = np.zeros(3)
        px_ok    = px_tot = 0
        seg_p, seg_t = [], []

        with torch.no_grad():
            for batch in va_dl:
                imgs  = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)
                logits = model(imgs)
                preds  = logits.argmax(1)
                valid  = masks >= 0
                n      = imgs.size(0)

                va_loss += (ce_fn(logits, masks) + dice_fn(logits, masks)).item() * n

                for c in range(3):
                    tp = ((preds == c) & (masks == c) & valid).sum().item()
                    fp = ((preds == c) & (masks != c) & valid).sum().item()
                    fn = ((preds != c) & (masks == c) & valid).sum().item()
                    d  = 2 * tp + fp + fn
                    dice_acc[c] += (2 * tp / d if d > 0 else 0.0) * n

                px_ok  += ((preds == masks) & valid).sum().item()
                px_tot += valid.sum().item()
                va_n   += n
                seg_p.append(preds[valid].cpu())
                seg_t.append(masks[valid].cpu())

        va_loss    /= max(va_n, 1)
        per_dice    = dice_acc / max(va_n, 1)
        macro_dice  = float(per_dice.mean())
        px_acc      = px_ok / max(px_tot, 1)
        seg_f1      = f1_score(torch.cat(seg_t).numpy(),
                               torch.cat(seg_p).numpy(),
                               average="macro", zero_division=0)

        wandb.log({
            "epoch":          epoch,
            "lr":             scheduler.get_last_lr()[0],
            "train/loss":     tr_loss,
            "val/loss":       va_loss,
            "val/macro_dice": macro_dice,
            "val/pixel_acc":  px_acc,
            "val/seg_f1":     seg_f1,
        })

        print(f"  Ep {epoch:03d}  tr={tr_loss:.4f}  va={va_loss:.4f}  "
              f"dice={macro_dice:.4f}  px={px_acc:.4f}")

        if macro_dice > best_dice:
            best_dice = macro_dice
            save_checkpoint(model, "unet", args.save_dir, epoch, best_dice)

    wandb.finish()
    print(f"Best val macro-Dice: {best_dice:.4f}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",     required=True)
    p.add_argument("--cls_ckpt",      default="checkpoints/classifier.pth")
    p.add_argument("--strategy",      default="all",
                   choices=["frozen", "partial", "full", "all"])
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=5e-4)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--save_dir",      default="checkpoints")
    p.add_argument("--wandb_project", default="da6401_a2")
    args = p.parse_args()

    strategies = (["frozen", "partial", "full"]
                  if args.strategy == "all" else [args.strategy])
    for s in strategies:
        train_strategy(args, s)


if __name__ == "__main__":
    main()