"""
train_task4.py — Unified Multi-Task Pipeline (Task 4)

Usage:
    python train_tasks/train_task4.py \
        --data_root /path/to/oxford_pets \
        --epochs 30 --batch_size 16
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

from models.multitask    import MultiTaskPerceptionModel
from models.segmentation import DiceCELoss
from losses.iou_loss     import IoULoss
from data.dataset        import OxfordPetDataset, collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def xyxy_to_cxcywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(1)
    return torch.stack([(x1+x2)*0.5,(y1+y2)*0.5,(x2-x1).clamp(0),(y2-y1).clamp(0)],1)


def iou_per_sample(pred, gt, eps=1e-6):
    def corners(b):
        cx,cy,w,h = b.unbind(1); return cx-w/2,cy-h/2,cx+w/2,cy+h/2
    px1,py1,px2,py2 = corners(pred); gx1,gy1,gx2,gy2 = corners(gt)
    iw=(torch.min(px2,gx2)-torch.max(px1,gx1)).clamp(0)
    ih=(torch.min(py2,gy2)-torch.max(py1,gy1)).clamp(0)
    inter=iw*ih; ap=(px2-px1).clamp(0)*(py2-py1).clamp(0)
    ag=(gx2-gx1).clamp(0)*(gy2-gy1).clamp(0)
    return inter/(ap+ag-inter+eps)


def save_checkpoint(model, tag, ckpt_dir, epoch, metric):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{tag}.pth")
    torch.save({"state_dict": model.state_dict(), "epoch": epoch, "metric": metric}, path)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",     required=True)
    p.add_argument("--cls_ckpt",      default="checkpoints/classifier.pth")
    p.add_argument("--loc_ckpt",      default="checkpoints/localizer.pth")
    p.add_argument("--seg_ckpt",      default="checkpoints/unet.pth")
    p.add_argument("--lambda_cls",    type=float, default=1.0)
    p.add_argument("--lambda_loc",    type=float, default=1.0)
    p.add_argument("--lambda_seg",    type=float, default=1.0)
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=5e-5)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--save_dir",      default="checkpoints")
    p.add_argument("--wandb_project", default="da6401_a2")
    args = p.parse_args()

    wandb.init(project=args.wandb_project, name="multitask", config=vars(args))
    print(f"Device: {DEVICE}")

    model = MultiTaskPerceptionModel(
        cls_ckpt=args.cls_ckpt,
        loc_ckpt=args.loc_ckpt,
        seg_ckpt=args.seg_ckpt,
    ).to(DEVICE)

    tr_ds = OxfordPetDataset(args.data_root, partition="train", mode="all")
    va_ds = OxfordPetDataset(args.data_root, partition="val",   mode="all")
    kw    = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                 pin_memory=True, collate_fn=collate_fn)
    tr_dl = DataLoader(tr_ds, shuffle=True,  **kw)
    va_dl = DataLoader(va_ds, shuffle=False, **kw)

    ce_fn   = nn.CrossEntropyLoss(label_smoothing=0.1)
    iou_fn  = IoULoss(reduction="mean")
    dice_fn = DiceCELoss(num_classes=3, ignore_index=-1)
    mse_fn  = nn.MSELoss()

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    IMG_SZ  = 224.0
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = tr_n = 0
        for batch in tr_dl:
            imgs      = batch["image"].to(DEVICE)
            labels    = batch["label"].to(DEVICE)
            bbox_xyxy = batch["bbox"].to(DEVICE)
            masks     = batch["mask"].to(DEVICE)
            valid_box = batch["bbox_mask"].to(DEVICE).bool()

            optimiser.zero_grad()
            out   = model(imgs)
            cls_l = ce_fn(out["classification"], labels)

            bbox_cx = xyxy_to_cxcywh(bbox_xyxy)
            if valid_box.sum() > 0:
                p_n   = out["localization"][valid_box] / IMG_SZ
                t_n   = bbox_cx[valid_box] / IMG_SZ
                loc_l = 0.7 * mse_fn(p_n, t_n) + iou_fn(p_n, t_n)
            else:
                loc_l = torch.tensor(0.0, device=DEVICE)

            seg_l = dice_fn(out["segmentation"], masks)
            loss  = (args.lambda_cls * cls_l +
                     args.lambda_loc * loc_l +
                     args.lambda_seg * seg_l)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            tr_loss += loss.item() * imgs.size(0)
            tr_n    += imgs.size(0)

        scheduler.step()
        tr_loss /= max(tr_n, 1)

        model.eval()
        va_cls_t, va_cls_p = [], []
        va_iou_vals = []
        dice_acc    = np.zeros(3)
        va_n        = 0

        with torch.no_grad():
            for batch in va_dl:
                imgs      = batch["image"].to(DEVICE)
                labels    = batch["label"].to(DEVICE)
                bbox_xyxy = batch["bbox"].to(DEVICE)
                masks     = batch["mask"].to(DEVICE)
                valid_box = batch["bbox_mask"].to(DEVICE).bool()
                n         = imgs.size(0)
                out       = model(imgs)

                va_cls_t.extend(labels.cpu().tolist())
                va_cls_p.extend(out["classification"].argmax(1).cpu().tolist())

                if valid_box.sum() > 0:
                    bbox_cx = xyxy_to_cxcywh(bbox_xyxy)
                    ious = iou_per_sample(
                        out["localization"][valid_box].clamp(0, IMG_SZ),
                        bbox_cx[valid_box])
                    va_iou_vals.extend(ious.cpu().tolist())

                preds = out["segmentation"].argmax(1)
                valid = masks >= 0
                for c in range(3):
                    tp = ((preds==c)&(masks==c)&valid).sum().item()
                    fp = ((preds==c)&(masks!=c)&valid).sum().item()
                    fn = ((preds!=c)&(masks==c)&valid).sum().item()
                    d  = 2*tp+fp+fn
                    dice_acc[c] += (2*tp/d if d>0 else 0.0) * n
                va_n += n

        macro_f1   = f1_score(va_cls_t, va_cls_p, average="macro", zero_division=0)
        mean_iou   = float(np.mean(va_iou_vals)) if va_iou_vals else 0.0
        macro_dice = float((dice_acc / max(va_n, 1)).mean())

        wandb.log({"epoch": epoch, "lr": scheduler.get_last_lr()[0],
                   "train/loss": tr_loss, "val/macro_f1": macro_f1,
                   "val/mean_iou": mean_iou, "val/macro_dice": macro_dice})

        print(f"  Ep {epoch:03d}  tr={tr_loss:.4f}  "
              f"f1={macro_f1:.4f}  iou={mean_iou:.4f}  dice={macro_dice:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            save_checkpoint(model, "multitask", args.save_dir, epoch, best_f1)

    wandb.finish()
    print(f"Best val macro-F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()