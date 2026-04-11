"""
train.py  —  DA6401 Assignment-2 Training Entry Point

Trains three tasks sequentially or individually:
  1. train_classification — VGG11Encoder + classifier head
  2. train_localization   — LocalizationModel (frozen/fine-tuned VGG11 encoder)
  3. train_segmentation   — UNetVGG11 (U-Net with VGG11 encoder)

After all three tasks are trained, checkpoints are saved as:
  checkpoints/classifier.pth
  checkpoints/localizer.pth
  checkpoints/unet.pth

These are the exact filenames expected by inference.py.

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

import wandb
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    precision_recall_fscore_support, confusion_matrix,
)
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project imports  (all paths relative to the repo root)
# ---------------------------------------------------------------------------
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
IMAGE_SIZE = 224.0
VAL_SPLIT  = 0.1
SEED       = 42

BREED_NAMES = [
    "Abyssinian", "american_bulldog", "american_pit_bull_terrier",
    "basset_hound", "beagle", "Bengal", "Birman", "Bombay", "boxer",
    "British_Shorthair", "chihuahua", "Egyptian_Mau",
    "english_cocker_spaniel", "english_setter", "german_shorthaired",
    "great_pyrenees", "havanese", "japanese_chin", "keeshond",
    "leonberger", "Maine_Coon", "miniature_pinscher", "newfoundland",
    "Persian", "pomeranian", "pug", "Ragdoll", "Russian_Blue",
    "saint_bernard", "samoyed", "scottish_terrier", "shiba_inu",
    "Siamese", "Sphynx", "staffordshire_bull_terrier",
    "wheaten_terrier", "yorkshire_terrier",
]
SEG_CLASS_NAMES = ["foreground", "background", "boundary"]


# ---------------------------------------------------------------------------
# Collate helper — handles None bboxes from PetLocalizationDataset
# ---------------------------------------------------------------------------
def loc_collate_fn(batch):
    """Stack (image, bbox) pairs; replace None bbox with zeros."""
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
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
               pin_memory=True)
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
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
               pin_memory=True)
    return (DataLoader(train_ds, shuffle=True,  **kw),
            DataLoader(val_ds,   shuffle=False, **kw))


# ---------------------------------------------------------------------------
# LR scheduler: warm-up → cosine annealing
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
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(model, name, ckpt_dir, epoch, metric):
    path = Path(ckpt_dir) / f"{name}.pth"
    torch.save({"state_dict": model.state_dict(),
                "epoch": epoch, "best_metric": metric}, path)
    try:
        artifact = wandb.Artifact(name=f"{name}_model", type="model")
        artifact.add_file(str(path))
        wandb.log_artifact(artifact)
    except Exception:
        pass  # W&B optional during local runs


def load_encoder_weights(model_with_encoder, ckpt_path, encoder_attr="encoder"):
    """Load VGG11Encoder weights into a sub-module named `encoder_attr`."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd   = ckpt.get("state_dict", ckpt)
    # The VGG11Encoder stores features + classifier; we only want features
    feat_sd = {k.replace("features.", ""): v
               for k, v in sd.items() if k.startswith("features.")}
    enc = getattr(model_with_encoder, encoder_attr)
    missing, unexpected = enc.load_state_dict(feat_sd, strict=False)
    print(f"  Loaded encoder weights from {ckpt_path}  "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def clf_metrics(y_true, y_pred):
    arr_t  = np.array(y_true)
    arr_p  = np.array(y_pred)
    labels = list(range(NUM_BREEDS))
    macro_f1  = f1_score(arr_t, arr_p, average="macro",  zero_division=0, labels=labels)
    macro_pre = precision_score(arr_t, arr_p, average="macro", zero_division=0, labels=labels)
    macro_rec = recall_score(arr_t, arr_p, average="macro",    zero_division=0, labels=labels)
    per_f1, _, _, _ = precision_recall_fscore_support(
        arr_t, arr_p, labels=labels, average=None, zero_division=0)
    cm_raw  = confusion_matrix(arr_t, arr_p, labels=labels).astype(np.float32)
    cm_norm = cm_raw / (cm_raw.sum(axis=1, keepdims=True) + 1e-6)
    return {"macro_f1": macro_f1, "macro_pre": macro_pre, "macro_rec": macro_rec,
            "per_class_f1": per_f1, "conf_matrix": cm_norm, "y_true": arr_t}


def batch_iou_cxcywh(pred, target, eps=1e-6):
    def to_xyxy(b):
        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dim=1)
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
# W&B log helpers
# ---------------------------------------------------------------------------
def log_confusion_matrix(cm_norm, epoch):
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    short = [n[:10] for n in BREED_NAMES]
    ax.set_xticks(range(NUM_BREEDS)); ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticks(range(NUM_BREEDS)); ax.set_yticklabels(short, fontsize=6)
    ax.set_xlabel("Predicted", fontsize=10); ax.set_ylabel("True", fontsize=10)
    ax.set_title(f"Normalised Confusion Matrix — epoch {epoch}")
    fig.colorbar(im, ax=ax, fraction=0.03); fig.tight_layout()
    wandb.log({"val/confusion_matrix": wandb.Image(fig, caption=f"CM epoch {epoch}"),
               "epoch": epoch})
    plt.close(fig)


def log_per_class_f1(per_class_f1, epoch):
    data  = [[BREED_NAMES[i], float(per_class_f1[i])] for i in range(NUM_BREEDS)]
    table = wandb.Table(data=data, columns=["breed", "f1"])
    wandb.log({"val/per_class_f1": wandb.plot.bar(
                   table, "breed", "f1", title=f"Class F1 (epoch {epoch})"),
               "epoch": epoch})


# ---------------------------------------------------------------------------
# Task 1: Classification  (saves checkpoints/classifier.pth)
# ---------------------------------------------------------------------------
def train_classification(args):
    wandb.init(project=args.wandb_project, name="classification", config=vars(args))
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

        wandb.log({
            "epoch": epoch, "lr": scheduler.get_last_lr()[0],
            "train/loss": tr_loss,           "val/loss": val_loss,
            "val/acc":    val_acc,
            "train/macro_f1":        tr_m["macro_f1"],
            "val/macro_f1":          val_m["macro_f1"],
            "train/macro_precision": tr_m["macro_pre"],
            "val/macro_precision":   val_m["macro_pre"],
            "train/macro_recall":    tr_m["macro_rec"],
            "val/macro_recall":      val_m["macro_rec"],
        })

        if epoch > 1 and (epoch % args.conf_matrix_every == 0 or epoch == args.epochs):
            log_per_class_f1(val_m["per_class_f1"], epoch)
            log_confusion_matrix(val_m["conf_matrix"], epoch)

        print(f"[Clf {epoch:03d}/{args.epochs}]  "
              f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
              f"acc={val_acc:.4f}  macro_f1={val_m['macro_f1']:.4f}")

        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            save_checkpoint(model, "classifier", args.ckpt_dir, epoch, best_f1)

    wandb.finish()
    print(f"Best val macro-F1: {best_f1:.4f}")


# ---------------------------------------------------------------------------
# Task 2: Localization  (saves checkpoints/localizer.pth)
# ---------------------------------------------------------------------------
def train_localization(args):
    wandb.init(project=args.wandb_project, name="localization", config=vars(args))
    tr_dl, va_dl = make_loc_loaders(args)

    # Build VGG11 and wrap in LocalizationModel
    vgg = VGG11Encoder(num_classes=NUM_BREEDS, dropout_p=args.dropout_p)
    clf_ckpt = Path(args.ckpt_dir) / "classifier.pth"
    if clf_ckpt.exists():
        ckpt = torch.load(clf_ckpt, map_location="cpu")
        sd   = ckpt.get("state_dict", ckpt)
        vgg.load_state_dict(sd, strict=False)
        print("Loaded VGG11 weights from classifier.pth")

    model = LocalizationModel(vgg, freeze_backbone=args.freeze_encoder).to(DEVICE)

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")
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
            # Filter samples that have valid (non-zero) boxes
            valid = bboxes.sum(dim=1) > 0
            if valid.sum() == 0:
                continue
            optimizer.zero_grad()
            pred = model(imgs)               # (B, 4) in [0, 1] cxcywh normalised
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
                pred = model(imgs)
                n    = valid.sum().item()
                loss = mse_loss(pred[valid], bboxes[valid]) + \
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

        wandb.log({
            "epoch": epoch, "lr": scheduler.get_last_lr()[0],
            "train/loss": tr_loss,      "val/loss":       va_loss,
            "val/mean_iou": va_iou,     "val/precision@50": va_p50,
            "val/precision@75": va_p75,
        })

        print(f"[Loc {epoch:03d}/{args.epochs}]  "
              f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
              f"iou={va_iou:.4f}  P@50={va_p50:.4f}  P@75={va_p75:.4f}")

        if va_iou > best_iou:
            best_iou = va_iou
            save_checkpoint(model, "localizer", args.ckpt_dir, epoch, best_iou)

    wandb.finish()
    print(f"Best val IoU: {best_iou:.4f}")


# ---------------------------------------------------------------------------
# Task 3: Segmentation  (saves checkpoints/unet.pth)
# ---------------------------------------------------------------------------
def train_segmentation(args):
    wandb.init(project=args.wandb_project, name="segmentation", config=vars(args))
    tr_dl, va_dl = make_seg_loaders(args)

    model = UNetVGG11(num_classes=3, dropout_p=args.dropout_p).to(DEVICE)

    clf_ckpt = Path(args.ckpt_dir) / "classifier.pth"
    if clf_ckpt.exists():
        ckpt   = torch.load(clf_ckpt, map_location="cpu")
        sd     = ckpt.get("state_dict", ckpt)
        # Load only the encoder (features) part
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

        wandb.log({
            "epoch": epoch, "lr": scheduler.get_last_lr()[0],
            "train/loss": tr_loss,       "val/loss":       va_loss,
            "val/dice_macro": macro_dice, "val/pixel_acc":  va_px_acc,
            **{f"val/dice_{SEG_CLASS_NAMES[i]}": float(va_dice_mean[i])
               for i in range(3)},
        })

        print(f"[Seg {epoch:03d}/{args.epochs}]  "
              f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
              f"dice={macro_dice:.4f}  px_acc={va_px_acc:.4f}  " +
              "  ".join(f"{SEG_CLASS_NAMES[i]}={va_dice_mean[i]:.3f}"
                        for i in range(3)))

        if macro_dice > best_dice:
            best_dice = macro_dice
            save_checkpoint(model, "unet", args.ckpt_dir, epoch, best_dice)

    wandb.finish()
    print(f"Best val Dice: {best_dice:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DA6401 A2 Training")
    p.add_argument("--task", choices=["classification", "localization",
                   "segmentation", "all"], default="classification",
                   help="Which task to train. 'all' runs 1→2→3 sequentially.")
    p.add_argument("--data_dir",          default="data/oxford_pet")
    p.add_argument("--ckpt_dir",          default="checkpoints")
    p.add_argument("--epochs",            type=int,   default=60)
    p.add_argument("--batch_size",        type=int,   default=32)
    p.add_argument("--lr",                type=float, default=5e-4)
    p.add_argument("--dropout_p",         type=float, default=0.5)
    p.add_argument("--weight_decay",      type=float, default=1e-4)
    p.add_argument("--label_smoothing",   type=float, default=0.1)
    p.add_argument("--freeze_encoder",    action="store_true",
                   help="Freeze backbone when training localization/segmentation.")
    p.add_argument("--num_workers",       type=int,   default=4)
    p.add_argument("--conf_matrix_every", type=int,   default=5)
    p.add_argument("--wandb_project",     default="da6401-assignment2")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
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