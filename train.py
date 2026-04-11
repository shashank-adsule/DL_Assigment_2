"""
train.py  —  DA6401 Assignment-2 Training Entry Point

Trains three tasks sequentially or individually:
  1. train_classification — VGG11Encoder + classifier head
  2. train_localization   — LocalizationModel (frozen/fine-tuned VGG11 encoder)
  3. train_segmentation   — UNetVGG11 (U-Net with VGG11 encoder)

Checkpoints are saved as:
  checkpoints/classifier.pth
  checkpoints/localizer.pth
  checkpoints/unet.pth

Usage:
    python train.py --task classification --data_dir /path/to/oxford_pet
    python train.py --task localization   --data_dir /path/to/oxford_pet
    python train.py --task segmentation   --data_dir /path/to/oxford_pet
    python train.py --task all            --data_dir /path/to/oxford_pet
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from sklearn.metrics import f1_score, precision_score, recall_score

from data.dataset import (
    PetClassificationDataset,
    PetLocalizationDataset,
    PetSegmentationDataset,
)
from models.vgg11        import VGG11Encoder
from models.localization import LocalizationModel
from models.segmentation import UNetVGG11, DiceCELoss
from losses.iou_loss     import IoULoss

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BREEDS = 37
VAL_SPLIT  = 0.1
SEED       = 42

SEG_CLASS_NAMES = ["foreground", "background", "boundary"]


# ---------------------------------------------------------------------------
# Collate helper — handles None bboxes from PetLocalizationDataset
# ---------------------------------------------------------------------------
def loc_collate_fn(batch):
    images, bboxes = zip(*batch)
    images = torch.stack(images)
    fixed  = [b if b is not None else torch.zeros(4) for b in bboxes]
    bboxes = torch.stack(fixed)
    return images, bboxes


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------
def _split_dataset(dataset, val_split=VAL_SPLIT, seed=SEED):
    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    return random_split(dataset, [n_train, n_val],
                        generator=torch.Generator().manual_seed(seed))


def make_cls_loaders(args):
    train_full = PetClassificationDataset(args.data_dir, split="trainval")
    train_ds, val_ds = _split_dataset(train_full)
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    return (DataLoader(train_ds, shuffle=True,  **kw),
            DataLoader(val_ds,   shuffle=False, **kw))


def make_loc_loaders(args):
    train_full = PetLocalizationDataset(args.data_dir, split="trainval")
    train_ds, val_ds = _split_dataset(train_full)
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
              pin_memory=True, collate_fn=loc_collate_fn)
    return (DataLoader(train_ds, shuffle=True,  **kw),
            DataLoader(val_ds,   shuffle=False, **kw))


def make_seg_loaders(args):
    train_full = PetSegmentationDataset(args.data_dir, split="trainval")
    train_ds, val_ds = _split_dataset(train_full)
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    return (DataLoader(train_ds, shuffle=True,  **kw),
            DataLoader(val_ds,   shuffle=False, **kw))


# ---------------------------------------------------------------------------
# LR scheduler: warm-up then cosine annealing
# ---------------------------------------------------------------------------
def make_scheduler(optimizer, warmup_epochs, total_epochs):
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer,
                               T_max=max(total_epochs - warmup_epochs, 1),
                               eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_epochs])


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------
def save_checkpoint(model, name, ckpt_dir, epoch, metric):
    path = Path(ckpt_dir) / f"{name}.pth"
    torch.save({"state_dict": model.state_dict(),
                "epoch": epoch, "best_metric": metric}, path)
    print(f"  Saved {path}  (epoch={epoch}, metric={metric:.4f})")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def clf_metrics(y_true, y_pred):
    arr_t  = np.array(y_true)
    arr_p  = np.array(y_pred)
    labels = list(range(NUM_BREEDS))
    return {
        "macro_f1":  f1_score(arr_t, arr_p, average="macro",  zero_division=0, labels=labels),
        "macro_pre": precision_score(arr_t, arr_p, average="macro", zero_division=0, labels=labels),
        "macro_rec": recall_score(arr_t, arr_p, average="macro",    zero_division=0, labels=labels),
    }


def batch_iou_cxcywh(pred, target, eps=1e-6):
    def to_xyxy(b):
        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=1)
    p, t  = to_xyxy(pred), to_xyxy(target)
    ix1   = torch.max(p[:, 0], t[:, 0]); iy1 = torch.max(p[:, 1], t[:, 1])
    ix2   = torch.min(p[:, 2], t[:, 2]); iy2 = torch.min(p[:, 3], t[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    pa    = (p[:, 2] - p[:, 0]).clamp(0) * (p[:, 3] - p[:, 1]).clamp(0)
    ta    = (t[:, 2] - t[:, 0]).clamp(0) * (t[:, 3] - t[:, 1]).clamp(0)
    return inter / (pa + ta - inter + eps)


def precision_at_iou(pred, target, threshold):
    return (batch_iou_cxcywh(pred, target) >= threshold).float().mean().item()


# ---------------------------------------------------------------------------
# Task 1: Classification
# ---------------------------------------------------------------------------
def train_classification(args):
    print("\n========== Task 1: Classification ==========")
    tr_dl, va_dl = make_cls_loaders(args)

    model     = VGG11Encoder(num_classes=NUM_BREEDS, dropout_p=args.dropout_p).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, warmup_epochs=5, total_epochs=args.epochs)

    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_f1  = 0.0

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        tr_loss = tr_total = 0
        tr_true, tr_pred   = [], []

        for imgs, labels in tr_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss  += loss.item() * imgs.size(0)
            tr_total += imgs.size(0)
            tr_true.extend(labels.cpu().tolist())
            tr_pred.extend(logits.argmax(1).cpu().tolist())

        scheduler.step()
        tr_loss /= tr_total
        tr_m     = clf_metrics(tr_true, tr_pred)

        # ---- Validate ----
        model.eval()
        val_loss = val_total = 0
        val_true, val_pred   = [], []

        with torch.no_grad():
            for imgs, labels in va_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                val_loss  += criterion(logits, labels).item() * imgs.size(0)
                val_total += imgs.size(0)
                val_true.extend(labels.cpu().tolist())
                val_pred.extend(logits.argmax(1).cpu().tolist())

        val_loss /= val_total
        val_m     = clf_metrics(val_true, val_pred)
        val_acc   = float(np.mean(np.array(val_true) == np.array(val_pred)))

        print(f"[Clf {epoch:03d}/{args.epochs}]  "
              f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
              f"acc={val_acc:.4f}  macro_f1={val_m['macro_f1']:.4f}  "
              f"pre={val_m['macro_pre']:.4f}  rec={val_m['macro_rec']:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            save_checkpoint(model, "classifier", args.ckpt_dir, epoch, best_f1)

    print(f"Best val macro-F1: {best_f1:.4f}\n")


# ---------------------------------------------------------------------------
# Task 2: Localization
# ---------------------------------------------------------------------------
def train_localization(args):
    print("\n========== Task 2: Localization ==========")
    tr_dl, va_dl = make_loc_loaders(args)

    vgg = VGG11Encoder(num_classes=NUM_BREEDS, dropout_p=args.dropout_p)
    clf_ckpt = Path(args.ckpt_dir) / "classifier.pth"
    if clf_ckpt.exists():
        ckpt = torch.load(clf_ckpt, map_location="cpu")
        vgg.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        print("Loaded VGG11 weights from classifier.pth")

    model = LocalizationModel(vgg, freeze_backbone=args.freeze_encoder).to(DEVICE)

    mse_loss  = nn.MSELoss()
    iou_loss  = IoULoss(reduction="mean")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, warmup_epochs=3, total_epochs=args.epochs)

    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        tr_loss = tr_total = 0

        for imgs, bboxes in tr_dl:
            imgs, bboxes = imgs.to(DEVICE), bboxes.to(DEVICE)
            valid = bboxes.sum(dim=1) > 0
            if valid.sum() == 0:
                continue
            optimizer.zero_grad()
            pred = model(imgs)
            loss = mse_loss(pred[valid], bboxes[valid]) + \
                   iou_loss(pred[valid], bboxes[valid])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = valid.sum().item()
            tr_loss  += loss.item() * n
            tr_total += n

        scheduler.step()
        tr_loss /= max(tr_total, 1)

        # ---- Validate ----
        model.eval()
        va_loss = va_iou = va_p50 = va_p75 = va_total = 0.0

        with torch.no_grad():
            for imgs, bboxes in va_dl:
                imgs, bboxes = imgs.to(DEVICE), bboxes.to(DEVICE)
                valid = bboxes.sum(dim=1) > 0
                if valid.sum() == 0:
                    continue
                pred     = model(imgs)
                n        = valid.sum().item()
                loss     = mse_loss(pred[valid], bboxes[valid]) + \
                           iou_loss(pred[valid], bboxes[valid])
                iou_vals = batch_iou_cxcywh(pred[valid], bboxes[valid])
                va_loss  += loss.item()            * n
                va_iou   += iou_vals.mean().item() * n
                va_p50   += precision_at_iou(pred[valid], bboxes[valid], 0.50) * n
                va_p75   += precision_at_iou(pred[valid], bboxes[valid], 0.75) * n
                va_total += n

        va_loss /= max(va_total, 1)
        va_iou  /= max(va_total, 1)
        va_p50  /= max(va_total, 1)
        va_p75  /= max(va_total, 1)

        print(f"[Loc {epoch:03d}/{args.epochs}]  "
              f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
              f"iou={va_iou:.4f}  P@50={va_p50:.4f}  P@75={va_p75:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if va_iou > best_iou:
            best_iou = va_iou
            save_checkpoint(model, "localizer", args.ckpt_dir, epoch, best_iou)

    print(f"Best val IoU: {best_iou:.4f}\n")


# ---------------------------------------------------------------------------
# Task 3: Segmentation
# ---------------------------------------------------------------------------
def train_segmentation(args):
    print("\n========== Task 3: Segmentation ==========")
    tr_dl, va_dl = make_seg_loaders(args)

    model = UNetVGG11(num_classes=3, freeze_encoder=args.freeze_encoder).to(DEVICE)

    clf_ckpt = Path(args.ckpt_dir) / "classifier.pth"
    if clf_ckpt.exists():
        ckpt   = torch.load(clf_ckpt, map_location="cpu")
        sd     = ckpt.get("state_dict", ckpt)
        enc_sd = {k.replace("features.", ""): v
                  for k, v in sd.items() if k.startswith("features.")}
        missing, unexpected = model.encoder.load_state_dict(enc_sd, strict=False)
        print(f"Loaded encoder from classifier.pth "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        if args.freeze_encoder:
            for p in model.encoder.parameters():
                p.requires_grad_(False)
            print("Encoder frozen.")

    dicece_loss = DiceCELoss(num_classes=3)
    optimizer   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler   = make_scheduler(optimizer, warmup_epochs=5, total_epochs=args.epochs)

    ckpt_dir  = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        tr_loss = tr_total = 0

        for imgs, masks in tr_dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = dicece_loss(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss  += loss.item() * imgs.size(0)
            tr_total += imgs.size(0)

        scheduler.step()
        tr_loss /= max(tr_total, 1)

        # ---- Validate ----
        model.eval()
        va_loss       = va_total   = 0.0
        va_dice_sum   = np.zeros(3)
        va_px_correct = va_px_total = 0

        with torch.no_grad():
            for imgs, masks in va_dl:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                logits = model(imgs)
                preds  = logits.argmax(1)
                valid  = masks >= 0
                n      = imgs.size(0)
                loss   = dicece_loss(logits, masks)
                va_loss += loss.item() * n

                for c in range(3):
                    tp = ((preds == c) & (masks == c) & valid).sum().item()
                    fp = ((preds == c) & (masks != c) & valid).sum().item()
                    fn = ((preds != c) & (masks == c) & valid).sum().item()
                    d  = 2 * tp + fp + fn
                    va_dice_sum[c] += (2 * tp / d if d > 0 else 0.0) * n

                va_px_correct += ((preds == masks) & valid).sum().item()
                va_px_total   += valid.sum().item()
                va_total      += n

        va_loss      /= va_total
        va_dice_mean  = va_dice_sum / va_total
        macro_dice    = float(va_dice_mean.mean())
        va_px_acc     = va_px_correct / max(va_px_total, 1)

        print(f"[Seg {epoch:03d}/{args.epochs}]  "
              f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
              f"dice={macro_dice:.4f}  px_acc={va_px_acc:.4f}  " +
              "  ".join(f"{SEG_CLASS_NAMES[i]}={va_dice_mean[i]:.3f}"
                        for i in range(3)) +
              f"  lr={scheduler.get_last_lr()[0]:.2e}")

        if macro_dice > best_dice:
            best_dice = macro_dice
            save_checkpoint(model, "unet", args.ckpt_dir, epoch, best_dice)

    print(f"Best val Dice: {best_dice:.4f}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DA6401 A2 Training (local, no W&B)")
    p.add_argument("--task", choices=["classification", "localization",
                   "segmentation", "all"], default="classification",
                   help="Which task to train. 'all' runs 1->2->3 sequentially.")
    # p.add_argument("--data_dir",        default="data/oxford_pet")
    p.add_argument("--data_dir",        default=r"D:\code\repo\DL_Assigment_2\temp")
    p.add_argument("--ckpt_dir",        default="checkpoints")
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=5e-4)
    p.add_argument("--dropout_p",       type=float, default=0.5)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--freeze_encoder",  action="store_true",
                   help="Freeze backbone when training localization/segmentation.")
    p.add_argument("--num_workers",     type=int,   default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Device: {DEVICE}")
    dispatch = {
        "classification": train_classification,
        "localization":   train_localization,
        "segmentation":   train_segmentation,
    }
    if args.task == "all":
        for fn in [train_classification, train_localization, train_segmentation]:
            fn(args)
    else:
        dispatch[args.task](args)