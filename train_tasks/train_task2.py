"""
train_task2.py — Bounding-Box Localisation (Task 2)

Loads the VGG-11 encoder from the Task-1 classifier checkpoint and
trains a BBoxHead on top of it.

Usage:
    python train_tasks/train_task2.py \
        --data_root /path/to/oxford_pets \
        --cls_ckpt  checkpoints/classifier.pth \
        --epochs 40 --batch_size 32
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

from models.localization import LocalizationModel
from data.dataset        import OxfordPetDataset, collate_fn
from losses.iou_loss     import IoULoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(1)
    return torch.stack(
        [(x1 + x2) * 0.5, (y1 + y2) * 0.5,
         (x2 - x1).clamp(0), (y2 - y1).clamp(0)],
        dim=1,
    )


def iou_per_sample(pred: torch.Tensor, gt: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    def corners(b):
        cx, cy, w, h = b.unbind(1)
        return cx - w/2, cy - h/2, cx + w/2, cy + h/2
    px1, py1, px2, py2 = corners(pred)
    gx1, gy1, gx2, gy2 = corners(gt)
    iw = (torch.min(px2, gx2) - torch.max(px1, gx1)).clamp(0)
    ih = (torch.min(py2, gy2) - torch.max(py1, gy1)).clamp(0)
    inter = iw * ih
    ap = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    ag = (gx2 - gx1).clamp(0) * (gy2 - gy1).clamp(0)
    return inter / (ap + ag - inter + eps)


def hit_rate(pred, gt, thr):
    return (iou_per_sample(pred, gt) >= thr).float().mean().item()


def save_checkpoint(model, tag, ckpt_dir, epoch, metric):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{tag}.pth")
    torch.save({"state_dict": model.state_dict(),
                "epoch": epoch, "metric": metric}, path)
    return path


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      default=r"D:\code\repo\DL_Assigment_2\temp",)
    p.add_argument("--cls_ckpt",       default=r"D:\code\repo\DL_Assigment_2\checkpoints", help="Task-1 classifier checkpoint")
    p.add_argument("--freeze_encoder", action="store_true", default=False)
    p.add_argument("--epochs",         type=int,   default=40)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=5e-4)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--save_dir",       default="checkpoints")
    p.add_argument("--wandb_project",  default="da6401_a2")
    args = p.parse_args()

    wandb.init(project=args.wandb_project, name="localisation",
               config=vars(args))
    print(f"Device: {DEVICE}")

    tr_ds = OxfordPetDataset(args.data_root, partition="train", mode="loc")
    va_ds = OxfordPetDataset(args.data_root, partition="val",   mode="loc")
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
              pin_memory=True, collate_fn=collate_fn)
    tr_dl = DataLoader(tr_ds, shuffle=True,  **kw)
    va_dl = DataLoader(va_ds, shuffle=False, **kw)

    model = LocalizationModel(freeze_backbone=args.freeze_encoder).to(DEVICE)

    # Warm-start encoder from classifier checkpoint
    if os.path.isfile(args.cls_ckpt):
        raw    = torch.load(args.cls_ckpt, map_location="cpu")
        sd     = raw.get("state_dict", raw)
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items()
                  if k.startswith("encoder.")}
        model.encoder.load_state_dict(enc_sd, strict=False)
        print(f"  Encoder warm-started from {args.cls_ckpt}")

    mse_fn    = nn.MSELoss()
    iou_fn    = IoULoss(reduction="mean")
    params    = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs)

    IMG_SZ   = 224.0
    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        tr_loss = tr_n = 0
        for batch in tr_dl:
            imgs       = batch["image"].to(DEVICE)
            bbox_xyxy  = batch["bbox"].to(DEVICE)
            valid_mask = batch["bbox_mask"].to(DEVICE).bool()
            if valid_mask.sum() == 0:
                continue
            bbox_cx = xyxy_to_cxcywh(bbox_xyxy)
            optimiser.zero_grad()
            pred    = model(imgs)
            p_norm  = pred[valid_mask] / IMG_SZ
            t_norm  = bbox_cx[valid_mask] / IMG_SZ
            loss    = 0.7 * mse_fn(p_norm, t_norm) + iou_fn(p_norm, t_norm)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            n_v     = valid_mask.sum().item()
            tr_loss += loss.item() * n_v
            tr_n    += n_v
        scheduler.step()
        tr_loss /= max(tr_n, 1)

        # ---- Validate ----
        model.eval()
        va_loss = va_iou = va_p50 = va_p75 = va_n = 0.0
        with torch.no_grad():
            for batch in va_dl:
                imgs       = batch["image"].to(DEVICE)
                bbox_xyxy  = batch["bbox"].to(DEVICE)
                valid_mask = batch["bbox_mask"].to(DEVICE).bool()
                if valid_mask.sum() == 0:
                    continue
                bbox_cx = xyxy_to_cxcywh(bbox_xyxy)
                pred    = model(imgs).clamp(0, IMG_SZ)
                p_norm  = pred[valid_mask] / IMG_SZ
                t_norm  = bbox_cx[valid_mask] / IMG_SZ
                loss    = 0.7 * mse_fn(p_norm, t_norm) + iou_fn(p_norm, t_norm)
                n_v     = valid_mask.sum().item()
                va_loss += loss.item() * n_v
                va_iou  += iou_per_sample(pred[valid_mask],
                                          bbox_cx[valid_mask]).mean().item() * n_v
                va_p50  += hit_rate(pred[valid_mask], bbox_cx[valid_mask], 0.50) * n_v
                va_p75  += hit_rate(pred[valid_mask], bbox_cx[valid_mask], 0.75) * n_v
                va_n    += n_v

        d = max(va_n, 1)
        va_loss /= d; va_iou /= d; va_p50 /= d; va_p75 /= d

        wandb.log({
            "epoch":        epoch,
            "lr":           scheduler.get_last_lr()[0],
            "train/loss":   tr_loss,
            "val/loss":     va_loss,
            "val/mean_iou": va_iou,
            "val/p@0.50":   va_p50,
            "val/p@0.75":   va_p75,
        })

        print(f"  Epoch {epoch:03d}  tr={tr_loss:.4f}  va={va_loss:.4f}  "
              f"iou={va_iou:.4f}  p50={va_p50:.4f}  p75={va_p75:.4f}")

        if va_iou > best_iou:
            best_iou = va_iou
            save_checkpoint(model, "localizer", args.save_dir, epoch, best_iou)

    wandb.finish()
    print(f"Best val mean-IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()