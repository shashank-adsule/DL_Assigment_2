"""
inference.py
-------------
Evaluate MultiTaskPerceptionModel on the test split.

Metrics reported:
  Classification : Macro F1-score (37 breeds)
  Detection      : Mean IoU  (cx,cy,w,h pixel predictions vs gt)
  Segmentation   : Macro Dice over valid trimap pixels

Usage:
    python inference.py --data_root /path/to/oxford_pets
"""
import os
import sys
import warnings
# Suppress albumentations offline version-check warning (harmless network timeout)
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from data.pets_dataset import OxfordPetDataset, collate_fn
from models.multitask import MultiTaskPerceptionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
def boxes_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(1)
    return torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dim=1)


def iou_corners(pred: torch.Tensor, gt: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x1 = torch.max(pred[:,0], gt[:,0]); y1 = torch.max(pred[:,1], gt[:,1])
    x2 = torch.min(pred[:,2], gt[:,2]); y2 = torch.min(pred[:,3], gt[:,3])
    inter = (x2-x1).clamp(0) * (y2-y1).clamp(0)
    ap = (pred[:,2]-pred[:,0]).clamp(0) * (pred[:,3]-pred[:,1]).clamp(0)
    ag = (gt[:,2]  -gt[:,0]).clamp(0)   * (gt[:,3]  -gt[:,1]).clamp(0)
    return inter / (ap + ag - inter + eps)


def macro_dice(logits: torch.Tensor, masks: torch.Tensor, ignore=-1) -> float:
    preds = logits.argmax(1); valid = masks != ignore
    total = count = 0.0
    for c in range(logits.size(1)):
        tp    = ((preds==c)&(masks==c)&valid).sum().item()
        fp    = ((preds==c)&(masks!=c)&valid).sum().item()
        fn    = ((preds!=c)&(masks==c)&valid).sum().item()
        denom = 2*tp + fp + fn
        if denom > 0:
            total += 2*tp/denom; count += 1
    return total / max(count, 1)


# ---------------------------------------------------------------------------
def evaluate(args) -> None:
    test_ds = OxfordPetDataset(args.data_root, partition="test", mode="all")
    test_dl = DataLoader(test_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=0, collate_fn=collate_fn)
    print(f"Test samples: {len(test_ds)}")

    model = MultiTaskPerceptionModel(
        cls_ckpt=str(Path(args.ckpt_dir) / "classifier.pth"),
        loc_ckpt=str(Path(args.ckpt_dir) / "localizer.pth"),
        seg_ckpt=str(Path(args.ckpt_dir) / "unet.pth"),
    ).to(DEVICE)
    model.eval()

    all_labels, all_preds = [], []
    iou_scores            = []
    dice_total = n_total  = 0

    with torch.no_grad():
        for batch in test_dl:
            imgs      = batch["image"].to(DEVICE)
            out       = model(imgs)

            # Classification
            all_labels.extend(batch["label"].tolist())
            all_preds.extend(out["classification"].argmax(1).cpu().tolist())

            # Localisation — model outputs (cx,cy,w,h) pixel space
            bbox      = batch["bbox"].to(DEVICE)
            valid_box = batch["bbox_mask"].to(DEVICE).bool()
            if valid_box.sum() > 0:
                pred_c = boxes_to_corners(out["localization"][valid_box])
                gt_c   = boxes_to_corners(bbox[valid_box])
                iou_scores.extend(iou_corners(pred_c, gt_c).cpu().tolist())

            # Segmentation
            masks = batch["mask"].to(DEVICE)
            d     = macro_dice(out["segmentation"], masks)
            n     = imgs.size(0)
            dice_total += d * n
            n_total    += n

    clf_f1    = f1_score(all_labels, all_preds, average="macro")
    mean_iou  = float(np.mean(iou_scores)) if iou_scores else 0.0
    mean_dice = dice_total / max(n_total, 1)

    print("\n" + "=" * 52)
    print(f"  Classification  Macro F1   :  {clf_f1:.4f}")
    print(f"  Detection       Mean IoU   :  {mean_iou:.4f}")
    print(f"  Segmentation    Dice Score :  {mean_dice:.4f}")
    print("=" * 52 + "\n")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   default=r"D:\code\repo\DL_Assigment_2\temp")
    p.add_argument("--ckpt_dir",    default="checkpoints")
    p.add_argument("--batch_size",  type=int, default=32)
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()