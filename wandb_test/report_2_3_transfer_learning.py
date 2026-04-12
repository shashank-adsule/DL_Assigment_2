"""
report_2_3_transfer_learning.py  —  Section 2.3: Transfer Learning Showdown

Trains UNetVGG11 segmentation under 3 encoder-freezing strategies,
each as its own W&B run, so curves can be overlaid in the dashboard.

  frozen  — entire VGG11 encoder frozen
  partial — block1-block3 frozen, block4+block5 trainable
  full    — all weights trainable end-to-end

Requires:
  checkpoints/classifier.pth   (produced by train.py --task classification)

Run from project root:
    python wandb_reports/report_2_3_transfer_learning.py \
        --data_root /path/to/pets \
        --cls_ckpt  checkpoints/classifier.pth \
        --strategy  all
"""

import os, sys, warnings, time
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from models.segmentation import UNetVGG11, DiceCELoss
from data.dataset import OxfordPetDataset, collate_fn


def make_loaders(root, batch_size, num_workers):
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=True, collate_fn=collate_fn)
    return (DataLoader(OxfordPetDataset(root, partition="train", mode="seg"),
                       shuffle=True, **kw),
            DataLoader(OxfordPetDataset(root, partition="val",   mode="seg"),
                       shuffle=False, **kw))


def apply_strategy(model: UNetVGG11, strategy: str) -> int:
    # Start with all trainable
    for p in model.parameters():
        p.requires_grad = True
    if strategy == "frozen":
        for p in model.encoder.parameters():
            p.requires_grad = False
    elif strategy == "partial":
        # Freeze only early blocks
        for blk in [model.encoder.block1, model.encoder.block2,
                    model.encoder.block3]:
            for p in blk.parameters():
                p.requires_grad = False
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tt = sum(p.numel() for p in model.parameters())
    print(f"  '{strategy}': {tr:,}/{tt:,} params trainable ({100*tr/tt:.1f}%)")
    return tr


def compute_metrics(logits, masks):
    preds = logits.argmax(1)
    valid = masks >= 0
    dice_sum, px_ok, px_tot = np.zeros(3), 0, 0
    for c in range(3):
        tp = ((preds==c)&(masks==c)&valid).sum().item()
        fp = ((preds==c)&(masks!=c)&valid).sum().item()
        fn = ((preds!=c)&(masks==c)&valid).sum().item()
        d  = 2*tp+fp+fn
        dice_sum[c] = 2*tp/d if d>0 else 0.0
    px_ok  = ((preds==masks)&valid).sum().item()
    px_tot = valid.sum().item()
    return dice_sum.mean(), px_ok / max(px_tot, 1)


def train_strategy(args, strategy: str, device):
    run_name = f"2.3_seg_{strategy}"
    wandb.init(project=args.wandb_project, name=run_name,
               config={**vars(args), "strategy": strategy}, reinit=True)

    tr_dl, va_dl = make_loaders(args.data_root, args.batch_size, args.num_workers)
    model = UNetVGG11(num_classes=3).to(device)

    # Load encoder from classifier checkpoint
    if args.cls_ckpt and os.path.isfile(args.cls_ckpt):
        model.load_encoder_from_checkpoint(args.cls_ckpt)

    trainable = apply_strategy(model, strategy)
    wandb.config.update({"trainable_params": trainable})

    seg_w   = torch.tensor([1.0, 0.8, 3.0], device=device)
    ce_fn   = nn.CrossEntropyLoss(ignore_index=-1, weight=seg_w)
    dice_fn = DiceCELoss(num_classes=3, ignore_index=-1)

    params = [p for p in model.parameters() if p.requires_grad]
    opt    = AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched  = CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- train ---
        model.train()
        tr_loss, tr_n = 0.0, 0
        for batch in tr_dl:
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss   = ce_fn(logits, masks) + dice_fn(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * imgs.size(0); tr_n += imgs.size(0)
        sched.step()
        tr_loss /= max(tr_n, 1)

        # --- validate ---
        model.eval()
        va_loss, va_n = 0.0, 0
        all_logits, all_masks = [], []
        with torch.no_grad():
            for batch in va_dl:
                imgs  = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs)
                va_loss += (ce_fn(logits, masks) + dice_fn(logits, masks)).item() * imgs.size(0)
                va_n    += imgs.size(0)
                all_logits.append(logits.cpu()); all_masks.append(masks.cpu())

        va_loss /= max(va_n, 1)
        lc, mc = torch.cat(all_logits), torch.cat(all_masks)
        macro_dice, px_acc = compute_metrics(lc, mc)
        elapsed = time.time() - t0

        wandb.log({"epoch": epoch, "lr": opt.param_groups[0]["lr"],
                   "train_loss": tr_loss, "val_loss": va_loss,
                   "val_dice": macro_dice, "val_pixel_acc": px_acc,
                   "time_per_epoch": elapsed}, step=epoch)

        print(f"  [{strategy} {epoch:02d}]  tr={tr_loss:.4f}  va={va_loss:.4f}  "
              f"dice={macro_dice:.4f}  px={px_acc:.4f}  t={elapsed:.1f}s")

    wandb.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",     default=r"D:\code\repo\DL_Assigment_2\temp")
    ap.add_argument("--cls_ckpt",      default="checkpoints/classifier.pth")
    ap.add_argument("--strategy",      default="all",
                    choices=["frozen", "partial", "full", "all"])
    ap.add_argument("--epochs",        type=int,   default=20)
    ap.add_argument("--batch_size",    type=int,   default=16)
    ap.add_argument("--lr",            type=float, default=2e-4)
    ap.add_argument("--num_workers",   type=int,   default=4)
    ap.add_argument("--wandb_project", default="da6401-assignment2")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    strategies = ["frozen", "partial", "full"] if args.strategy == "all" \
                 else [args.strategy]
    for s in strategies:
        print(f"\n{'='*55}\nStrategy: {s}\n{'='*55}")
        train_strategy(args, s, device)

    print("\nSection 2.3 done.")
    print("Tip: in W&B select all 3 runs → overlay val_dice and val_loss.")


if __name__ == "__main__":
    main()