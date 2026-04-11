"""
train.py — DA6401 Assignment-2 Training Entry Point

Mirrors the structure of the reference implementation but uses your
own model classes (PetClassifier, LocalizationModel, UNetVGG11) and
your OxfordPetDataset loader.

Usage
-----
  # Task 1 — classification (run first)
  python train.py --task classification --data_dir /path/to/oxford_pets

  # Task 2 — localisation (needs classifier.pth)
  python train.py --task localization --data_dir /path/to/oxford_pets

  # Task 3 — segmentation (needs classifier.pth)
  python train.py --task segmentation --data_dir /path/to/oxford_pets

  # Ablation variants (for W&B report sections 2.1 and 2.2)
  python train.py --task classification --data_dir /path/to/oxford_pets --ablation

Checkpoints are saved to --ckpt_dir (default: checkpoints/)
  classifier.pth   — best val macro-F1
  localizer.pth    — best val mean-IoU
  unet.pth         — best val macro-Dice
"""

import os
import warnings
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import wandb
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    precision_recall_fscore_support, confusion_matrix,
)
import matplotlib
import matplotlib.pyplot as plt

from data.dataset          import OxfordPetDataset, collate_fn
from models.classification import PetClassifier
from models.localization   import LocalizationModel
from models.segmentation   import UNetVGG11, DiceCELoss
from losses.iou_loss       import IoULoss

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_BREEDS = 37
IMAGE_SIZE = 224.0

BREED_NAMES = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx", "american_bulldog", "american_pit_bull_terrier",
    "basset_hound", "beagle", "boxer", "chihuahua", "english_cocker_spaniel",
    "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
    "japanese_chin", "keeshond", "leonberger", "miniature_pinscher",
    "newfoundland", "pomeranian", "pug", "saint_bernard", "samoyed",
    "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier",
    "wheaten_terrier", "yorkshire_terrier",
]
SEG_CLASS_NAMES = ["foreground", "background", "boundary"]


# ---------------------------------------------------------------------------
# Utility: box conversion
# ---------------------------------------------------------------------------

def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (x1, y1, x2, y2) pixel boxes → (cx, cy, w, h) pixel boxes."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.stack([
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        (x2 - x1).clamp(min=0),
        (y2 - y1).clamp(min=0),
    ], dim=1)


def batch_iou_cxcywh(pred: torch.Tensor, target: torch.Tensor,
                     eps: float = 1e-6) -> torch.Tensor:
    """Per-sample IoU for (cx, cy, w, h) boxes."""
    def to_xyxy(b):
        cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dim=1)
    p, t  = to_xyxy(pred), to_xyxy(target)
    ix1   = torch.max(p[:, 0], t[:, 0]); iy1 = torch.max(p[:, 1], t[:, 1])
    ix2   = torch.min(p[:, 2], t[:, 2]); iy2 = torch.min(p[:, 3], t[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    pa    = (p[:, 2]-p[:, 0]).clamp(0) * (p[:, 3]-p[:, 1]).clamp(0)
    ta    = (t[:, 2]-t[:, 0]).clamp(0) * (t[:, 3]-t[:, 1]).clamp(0)
    return inter / (pa + ta - inter + eps)


def precision_at_iou(pred: torch.Tensor, target: torch.Tensor,
                     threshold: float) -> float:
    return (batch_iou_cxcywh(pred, target) >= threshold).float().mean().item()


# ---------------------------------------------------------------------------
# Mixup augmentation (classification only)
# ---------------------------------------------------------------------------

def mixup_data(imgs: torch.Tensor, labels: torch.Tensor, alpha: float = 0.4):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    return lam * imgs + (1 - lam) * imgs[idx], labels, labels[idx], lam


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average of model weights)
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.shadow = {k: v.clone().detach().float()
                       for k, v in model.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            self.shadow[k] = d * self.shadow[k] + (1 - d) * v.detach().float()

    def apply(self, model: nn.Module) -> None:
        self._backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(
            {k: v.to(self._backup[k].dtype) for k, v in self.shadow.items()})

    def restore(self, model: nn.Module) -> None:
        model.load_state_dict(self._backup)


# ---------------------------------------------------------------------------
# Soft Dice loss (segmentation, ignores -1 pixels)
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        C     = logits.size(1)
        probs = torch.softmax(logits, dim=1)
        valid = targets != self.ignore_index

        tgt_c = targets.clone(); tgt_c[~valid] = 0
        oh    = nn.functional.one_hot(tgt_c, C).permute(0, 3, 1, 2).float()
        mask  = valid.unsqueeze(1).float()
        probs = probs * mask; oh = oh * mask

        inter    = (probs * oh).sum(dim=(0, 2, 3))
        cardinal = (probs + oh).sum(dim=(0, 2, 3))
        return 1.0 - ((2 * inter + self.smooth) / (cardinal + self.smooth)).mean()


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def clf_metrics(y_true: list, y_pred: list) -> dict:
    t, p   = np.array(y_true), np.array(y_pred)
    labels = list(range(NUM_BREEDS))
    per_f1, _, _, _ = precision_recall_fscore_support(
        t, p, labels=labels, average=None, zero_division=0)
    cm_raw  = confusion_matrix(t, p, labels=labels).astype(np.float32)
    cm_norm = cm_raw / (cm_raw.sum(axis=1, keepdims=True) + 1e-6)
    return {
        "macro_f1":     f1_score(t, p, average="macro",  zero_division=0, labels=labels),
        "macro_pre":    precision_score(t, p, average="macro", zero_division=0, labels=labels),
        "macro_rec":    recall_score(t, p, average="macro",    zero_division=0, labels=labels),
        "per_class_f1": per_f1,
        "conf_matrix":  cm_norm,
        "y_true":       t,
    }


# ---------------------------------------------------------------------------
# W&B logging helpers
# ---------------------------------------------------------------------------

def log_confusion_matrix(cm_norm: np.ndarray, epoch: int) -> None:
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    short = [n[:10] for n in BREED_NAMES]
    ax.set_xticks(range(NUM_BREEDS)); ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticks(range(NUM_BREEDS)); ax.set_yticklabels(short, fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Normalised Confusion Matrix — epoch {epoch}")
    fig.colorbar(im, ax=ax, fraction=0.03); fig.tight_layout()
    wandb.log({"val/confusion_matrix": wandb.Image(fig, caption=f"CM epoch {epoch}"),
               "epoch": epoch})
    plt.close(fig)


def log_per_class_f1(per_f1: np.ndarray, epoch: int) -> None:
    data  = [[BREED_NAMES[i], float(per_f1[i])] for i in range(NUM_BREEDS)]
    table = wandb.Table(data=data, columns=["breed", "f1"])
    wandb.log({"val/per_class_f1": wandb.plot.bar(
                   table, "breed", "f1", title=f"Class F1 (epoch {epoch})"),
               "epoch": epoch})


def log_class_distribution(y_true: np.ndarray) -> None:
    wandb.log({"val/class_distribution":
               wandb.Histogram(np.bincount(y_true, minlength=NUM_BREEDS))})


def save_checkpoint(model: nn.Module, name: str, ckpt_dir: str,
                    epoch: int, metric: float) -> None:
    path = Path(ckpt_dir) / f"{name}.pth"
    torch.save({"state_dict": model.state_dict(),
                "epoch": epoch, "best_metric": metric}, path)
    art = wandb.Artifact(name=f"{name}_model", type="model")
    art.add_file(str(path)); wandb.log_artifact(art)
    print(f"  Saved {path}  (metric={metric:.4f})")


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_loaders(args, mode: str):
    tr = OxfordPetDataset(args.data_dir, partition="train", mode=mode)
    va = OxfordPetDataset(args.data_dir, partition="val",   mode=mode)
    kw = dict(num_workers=args.num_workers, pin_memory=True,
              collate_fn=collate_fn)
    return (DataLoader(tr, batch_size=args.batch_size, shuffle=True,  **kw),
            DataLoader(va, batch_size=args.batch_size, shuffle=False, **kw))


def make_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Linear warm-up → cosine annealing to 1e-6."""
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer,
                               T_max=max(total_epochs - warmup_epochs, 1),
                               eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_epochs])


# ---------------------------------------------------------------------------
# Task 1: Classification
# ---------------------------------------------------------------------------

def train_classification(args) -> None:
    wandb.init(project=args.wandb_project, name="classification",
               config=vars(args))

    tr_dl, va_dl = make_loaders(args, mode="cls")

    model     = PetClassifier(num_classes=NUM_BREEDS,
                               drop_rate=args.dropout_p).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, warmup_epochs=5,
                               total_epochs=args.epochs)
    ema       = EMA(model, decay=0.999)

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    best_f1     = 0.0
    WARMUP_DONE = 5

    for epoch in range(1, args.epochs + 1):

        # ---- Train ----
        model.train()
        tr_loss = tr_total = 0
        tr_true, tr_pred = [], []

        for batch in tr_dl:
            imgs   = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            optimizer.zero_grad()

            if epoch > WARMUP_DONE and args.mixup_alpha > 0:
                imgs_m, la, lb, lam = mixup_data(imgs, labels, args.mixup_alpha)
                logits = model(imgs_m)
                loss   = lam * criterion(logits, la) + (1 - lam) * criterion(logits, lb)
            else:
                logits = model(imgs)
                loss   = criterion(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)

            tr_loss  += loss.item() * imgs.size(0)
            tr_total += imgs.size(0)
            tr_true.extend(labels.cpu().tolist())
            tr_pred.extend(logits.argmax(1).cpu().tolist())

        scheduler.step()
        tr_loss /= tr_total
        tr_m     = clf_metrics(tr_true, tr_pred)

        # ---- Validate (EMA weights give better generalisation) ----
        ema.apply(model)
        model.eval()
        val_loss = val_total = 0
        val_true, val_pred = [], []

        with torch.no_grad():
            for batch in va_dl:
                imgs   = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                logits = model(imgs)
                val_loss  += criterion(logits, labels).item() * imgs.size(0)
                val_total += imgs.size(0)
                val_true.extend(labels.cpu().tolist())
                val_pred.extend(logits.argmax(1).cpu().tolist())

        ema.restore(model)

        val_loss /= val_total
        val_m     = clf_metrics(val_true, val_pred)
        val_acc   = float(np.mean(np.array(val_true) == np.array(val_pred)))

        wandb.log({
            "epoch": epoch, "lr": scheduler.get_last_lr()[0],
            "train/loss":            tr_loss,     "val/loss":          val_loss,
            "val/acc":               val_acc,
            "train/macro_f1":        tr_m["macro_f1"],
            "val/macro_f1":          val_m["macro_f1"],
            "train/macro_precision": tr_m["macro_pre"],
            "val/macro_precision":   val_m["macro_pre"],
            "train/macro_recall":    tr_m["macro_rec"],
            "val/macro_recall":      val_m["macro_rec"],
        })

        if epoch > 1 and (epoch % args.conf_matrix_every == 0
                          or epoch == args.epochs):
            log_per_class_f1(val_m["per_class_f1"], epoch)
            log_confusion_matrix(val_m["conf_matrix"], epoch)
            log_class_distribution(val_m["y_true"])

        print(f"[Clf {epoch:03d}/{args.epochs}]  "
              f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
              f"acc={val_acc:.4f}  macro_f1={val_m['macro_f1']:.4f}  "
              f"pre={val_m['macro_pre']:.4f}  rec={val_m['macro_rec']:.4f}")

        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            ema.apply(model)
            save_checkpoint(model, "classifier", args.ckpt_dir, epoch, best_f1)
            ema.restore(model)

    wandb.finish()
    print(f"Best val macro-F1: {best_f1:.4f}")


# ---------------------------------------------------------------------------
# Task 1 ablation variant (no BN / different dropout) for W&B sections 2.1-2.2
# ---------------------------------------------------------------------------

def train_classification_variant(args, use_bn: bool, dropout_p: float,
                                  run_suffix: str) -> None:
    run_name = f"classification_bn{int(use_bn)}_dp{dropout_p}{run_suffix}"
    wandb.init(project=args.wandb_project, name=run_name,
               config={**vars(args), "use_bn": use_bn, "dropout_p": dropout_p},
               reinit=True)

    tr_dl, va_dl = make_loaders(args, mode="cls")
    model        = PetClassifier(num_classes=NUM_BREEDS,
                                  drop_rate=dropout_p).to(DEVICE)

    if not use_bn:
        for name, m in list(model.named_modules()):
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                parent = model
                parts  = name.rsplit(".", 1)
                if len(parts) == 2:
                    for attr in parts[0].split("."):
                        parent = getattr(parent, attr)
                setattr(parent,
                        parts[-1] if len(parts) == 2 else name,
                        nn.Identity())

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, warmup_epochs=5,
                               total_epochs=args.epochs)

    best_f1 = 0.0
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = tr_total = 0
        tr_true, tr_pred = [], []
        for batch in tr_dl:
            imgs   = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
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

        model.eval()
        val_loss = val_total = 0
        val_true, val_pred = [], []
        with torch.no_grad():
            for batch in va_dl:
                imgs   = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
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
            "train/loss":   tr_loss,         "val/loss":     val_loss,
            "val/acc":      val_acc,
            "train/macro_f1": tr_m["macro_f1"],
            "val/macro_f1":   val_m["macro_f1"],
            "generalisation_gap": val_loss - tr_loss,
        })

        print(f"  [{run_name} {epoch:03d}]  "
              f"tr={tr_loss:.4f}  va={val_loss:.4f}  "
              f"f1={val_m['macro_f1']:.4f}")

        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]

    wandb.finish()
    print(f"  Best macro-F1: {best_f1:.4f}")


# ---------------------------------------------------------------------------
# Task 2: Localisation
# ---------------------------------------------------------------------------

def train_localization(args) -> None:
    wandb.init(project=args.wandb_project, name="localization",
               config=vars(args))

    tr_dl, va_dl = make_loaders(args, mode="loc")

    model = LocalizationModel(dropout_p=args.dropout_p).to(DEVICE)

    # Warm-start encoder from classifier checkpoint
    clf_ckpt = Path(args.ckpt_dir) / "classifier.pth"
    if clf_ckpt.exists():
        ckpt   = torch.load(clf_ckpt, map_location="cpu")
        sd     = ckpt.get("state_dict", ckpt)
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items()
                  if k.startswith("encoder.")}
        model.encoder.load_state_dict(enc_sd, strict=False)
        print("Loaded encoder from classifier.pth")

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, warmup_epochs=3,
                               total_epochs=args.epochs)

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):

        # ---- Train ----
        model.train()
        tr_loss = tr_total = 0

        for batch in tr_dl:
            imgs      = batch["image"].to(DEVICE)
            bbox      = batch["bbox"].to(DEVICE)          # xyxy pixel
            bbox_mask = batch["bbox_mask"].to(DEVICE).bool()

            if bbox_mask.sum() == 0:
                continue

            bbox_cx = xyxy_to_cxcywh(bbox)               # cxcywh pixel
            optimizer.zero_grad()
            pred = model(imgs)                             # cxcywh pixel (sigmoid*224)

            pred_n   = pred[bbox_mask]    / IMAGE_SIZE
            target_n = bbox_cx[bbox_mask] / IMAGE_SIZE

            loss = mse_loss(pred_n, target_n) + iou_loss(pred_n, target_n)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n_valid   = bbox_mask.sum().item()
            tr_loss  += loss.item() * n_valid
            tr_total += n_valid

        scheduler.step()
        tr_loss /= max(tr_total, 1)

        # ---- Validate ----
        model.eval()
        va_loss = va_iou = va_p50 = va_p75 = va_total = 0.0

        with torch.no_grad():
            for batch in va_dl:
                imgs      = batch["image"].to(DEVICE)
                bbox      = batch["bbox"].to(DEVICE)
                bbox_mask = batch["bbox_mask"].to(DEVICE).bool()

                if bbox_mask.sum() == 0:
                    continue

                bbox_cx = xyxy_to_cxcywh(bbox)
                pred    = model(imgs)
                n_valid = bbox_mask.sum().item()

                pred_n   = pred[bbox_mask]    / IMAGE_SIZE
                target_n = bbox_cx[bbox_mask] / IMAGE_SIZE

                loss    = mse_loss(pred_n, target_n) + iou_loss(pred_n, target_n)
                iou_val = batch_iou_cxcywh(pred[bbox_mask], bbox_cx[bbox_mask])

                va_loss  += loss.item()               * n_valid
                va_iou   += iou_val.mean().item()     * n_valid
                va_p50   += precision_at_iou(pred[bbox_mask],
                                             bbox_cx[bbox_mask], 0.50) * n_valid
                va_p75   += precision_at_iou(pred[bbox_mask],
                                             bbox_cx[bbox_mask], 0.75) * n_valid
                va_total += n_valid

        va_loss /= max(va_total, 1); va_iou /= max(va_total, 1)
        va_p50  /= max(va_total, 1); va_p75 /= max(va_total, 1)

        wandb.log({
            "epoch": epoch, "lr": scheduler.get_last_lr()[0],
            "train/loss":       tr_loss,  "val/loss":       va_loss,
            "val/mean_iou":     va_iou,
            "val/precision@50": va_p50,   "val/precision@75": va_p75,
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
# Task 3: Segmentation
# ---------------------------------------------------------------------------

def train_segmentation(args) -> None:
    wandb.init(project=args.wandb_project, name="segmentation",
               config=vars(args))

    tr_dl, va_dl = make_loaders(args, mode="seg")

    model = UNetVGG11(num_classes=3, in_channels=3,
                      dropout_p=args.dropout_p).to(DEVICE)

    # Warm-start encoder from classifier checkpoint
    clf_ckpt = Path(args.ckpt_dir) / "classifier.pth"
    if clf_ckpt.exists():
        model.load_encoder_from_checkpoint(str(clf_ckpt))

    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        print("Encoder frozen.")

    # Up-weight the sparse boundary class (class 2) to prevent ignoring it
    seg_weights = torch.tensor([1.0, 0.8, 3.0], device=DEVICE)
    ce_loss     = nn.CrossEntropyLoss(ignore_index=-1, weight=seg_weights)
    dice_loss   = DiceLoss(ignore_index=-1)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, warmup_epochs=5,
                               total_epochs=args.epochs)

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):

        # ---- Train ----
        model.train()
        tr_loss = tr_total = 0

        for batch in tr_dl:
            imgs  = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = ce_loss(logits, masks) + dice_loss(logits, masks)
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
        seg_all_pred, seg_all_true  = [], []

        with torch.no_grad():
            for batch in va_dl:
                imgs  = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)
                logits = model(imgs)
                preds  = logits.argmax(1)
                valid  = masks >= 0
                n      = imgs.size(0)

                va_loss += (ce_loss(logits, masks) +
                            dice_loss(logits, masks)).item() * n

                for c in range(3):
                    tp = ((preds == c) & (masks == c) & valid).sum().item()
                    fp = ((preds == c) & (masks != c) & valid).sum().item()
                    fn = ((preds != c) & (masks == c) & valid).sum().item()
                    d  = 2 * tp + fp + fn
                    va_dice_sum[c] += (2 * tp / d if d > 0 else 0.0) * n

                va_px_correct += ((preds == masks) & valid).sum().item()
                va_px_total   += valid.sum().item()
                va_total      += n
                seg_all_pred.append(preds[valid].cpu())
                seg_all_true.append(masks[valid].cpu())

        va_loss      /= va_total
        va_dice_mean  = va_dice_sum / va_total
        macro_dice    = float(va_dice_mean.mean())
        va_px_acc     = va_px_correct / max(va_px_total, 1)

        all_p = torch.cat(seg_all_pred).numpy()
        all_t = torch.cat(seg_all_true).numpy()
        seg_f1  = f1_score(all_t, all_p, average="macro", zero_division=0)
        seg_pre = precision_score(all_t, all_p, average="macro", zero_division=0)
        seg_rec = recall_score(all_t, all_p, average="macro",    zero_division=0)

        wandb.log({
            "epoch": epoch, "lr": scheduler.get_last_lr()[0],
            "train/loss":     tr_loss, "val/loss":       va_loss,
            "val/dice_macro": macro_dice,
            **{f"val/dice_{SEG_CLASS_NAMES[i]}": float(va_dice_mean[i])
               for i in range(3)},
            "val/pixel_acc":       va_px_acc,
            "val/macro_f1":        seg_f1,
            "val/macro_precision": seg_pre,
            "val/macro_recall":    seg_rec,
        })

        print(f"[Seg {epoch:03d}/{args.epochs}]  "
              f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
              f"dice={macro_dice:.4f}  px_acc={va_px_acc:.4f}  "
              + "  ".join(f"{SEG_CLASS_NAMES[i]}={va_dice_mean[i]:.3f}"
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
    p = argparse.ArgumentParser(description="DA6401 Assignment-2 Training")
    p.add_argument("--task",
                   choices=["classification", "localization", "segmentation"],
                   default=["classification", "localization", "segmentation"][0])
    p.add_argument("--data_dir",          default=r"D:\code\repo\DL_Assigment_2\temp",
                   help="Root of the Oxford-IIIT Pet dataset")
    p.add_argument("--ckpt_dir",          default="checkpoints")
    p.add_argument("--epochs",            type=int,   default=60)
    p.add_argument("--batch_size",        type=int,   default=32)
    p.add_argument("--lr",                type=float, default=5e-4)
    p.add_argument("--dropout_p",         type=float, default=0.5)
    p.add_argument("--weight_decay",      type=float, default=1e-4)
    p.add_argument("--label_smoothing",   type=float, default=0.1)
    p.add_argument("--mixup_alpha",       type=float, default=0.4,
                   help="Mixup α for classification (0 = disabled)")
    p.add_argument("--freeze_encoder",    action="store_true",
                   help="Freeze encoder when training segmentation")
    p.add_argument("--num_workers",       type=int,   default=4)
    p.add_argument("--conf_matrix_every", type=int,   default=5,
                   help="Log confusion matrix every N epochs")
    p.add_argument("--wandb_project",     default="da6401-assignment2")
    p.add_argument("--ablation",          action="store_true",
                   help="Run BN/Dropout ablation variants (classification only)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.task == "classification":
        train_classification(args)

        if args.ablation:
            # Section 2.1 — no BatchNorm
            train_classification_variant(args, use_bn=False,
                                         dropout_p=0.5, run_suffix="_nobn")
            # Section 2.2 — dropout p=0.2
            train_classification_variant(args, use_bn=True,
                                         dropout_p=0.2, run_suffix="_dp02")
            # Section 2.2 — no dropout
            train_classification_variant(args, use_bn=True,
                                         dropout_p=0.0, run_suffix="_nodp")

    elif args.task == "localization":
        train_localization(args)

    elif args.task == "segmentation":
        train_segmentation(args)