"""
train_task1.py — VGG-11 Classification (Task 1)

Usage:
    python train_tasks/train_task1.py \
        --data_root /path/to/oxford_pets \
        --epochs 60 --batch_size 32 \
        --wandb_project da6401_a2

Add --ablation to also run the BN and Dropout variants for W&B sections 2.1 & 2.2.
"""
import os
import sys
import warnings
# Suppress albumentations offline version-check warning (harmless network timeout)
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


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

from sklearn.metrics import f1_score, precision_score, recall_score

from models.classification import PetClassifier
from models.layers         import CustomDropout
from data.dataset          import OxfordPetDataset, collate_fn
from losses.iou_loss       import IoULoss


DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BREEDS = 37


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(1)
    return torch.stack(
        [(x1 + x2) * 0.5, (y1 + y2) * 0.5,
         (x2 - x1).clamp(0), (y2 - y1).clamp(0)],
        dim=1,
    )


def save_checkpoint(model: nn.Module, tag: str, ckpt_dir: str,
                    epoch: int, metric: float) -> str:
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{tag}.pth")
    torch.save({"state_dict": model.state_dict(),
                "epoch": epoch, "metric": metric}, path)
    return path


def compute_clf_metrics(y_true, y_pred) -> dict:
    t, p = np.array(y_true), np.array(y_pred)
    labels = list(range(N_BREEDS))
    return {
        "macro_f1":  f1_score(t, p, average="macro", zero_division=0, labels=labels),
        "macro_pre": precision_score(t, p, average="macro", zero_division=0, labels=labels),
        "macro_rec": recall_score(t, p, average="macro", zero_division=0, labels=labels),
    }


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_one_config(args, use_bn: bool, dropout_p: float, run_tag: str):
    run_name = f"cls_bn{int(use_bn)}_dp{dropout_p}{run_tag}"
    wandb.init(project=args.wandb_project, name=run_name,
               config={**vars(args), "use_bn": use_bn, "dropout_p": dropout_p},
               reinit=True)
    print(f"\nDevice: {DEVICE}  |  Run: {run_name}")

    tr_ds = OxfordPetDataset(args.data_root, partition="train", mode="cls")
    va_ds = OxfordPetDataset(args.data_root, partition="val",   mode="cls")
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
              pin_memory=True, collate_fn=collate_fn)
    tr_dl = DataLoader(tr_ds, shuffle=True,  **kw)
    va_dl = DataLoader(va_ds, shuffle=False, **kw)

    model = PetClassifier(num_classes=N_BREEDS, drop_rate=dropout_p).to(DEVICE)

    if not use_bn:
        for name, m in list(model.named_modules()):
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                parts = name.rsplit(".", 1)
                parent = model
                if len(parts) == 2:
                    for attr in parts[0].split("."):
                        parent = getattr(parent, attr)
                setattr(parent, parts[-1] if len(parts) == 2 else name,
                        nn.Identity())

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs)

    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        tr_loss = tr_n = 0
        tr_t, tr_p = [], []
        for batch in tr_dl:
            imgs   = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            optimiser.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            tr_loss += loss.item() * imgs.size(0)
            tr_n    += imgs.size(0)
            tr_t.extend(labels.cpu().tolist())
            tr_p.extend(logits.argmax(1).cpu().tolist())
        scheduler.step()
        tr_loss /= max(tr_n, 1)
        tr_m     = compute_clf_metrics(tr_t, tr_p)

        # ---- Validate ----
        model.eval()
        va_loss = va_n = 0
        va_t, va_p = [], []
        with torch.no_grad():
            for batch in va_dl:
                imgs   = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                logits = model(imgs)
                va_loss += criterion(logits, labels).item() * imgs.size(0)
                va_n    += imgs.size(0)
                va_t.extend(labels.cpu().tolist())
                va_p.extend(logits.argmax(1).cpu().tolist())
        va_loss /= max(va_n, 1)
        va_m     = compute_clf_metrics(va_t, va_p)
        va_acc   = float(np.mean(np.array(va_t) == np.array(va_p)))

        wandb.log({
            "epoch":          epoch,
            "lr":             scheduler.get_last_lr()[0],
            "train/loss":     tr_loss,
            "val/loss":       va_loss,
            "val/accuracy":   va_acc,
            "train/macro_f1": tr_m["macro_f1"],
            "val/macro_f1":   va_m["macro_f1"],
            "val/macro_pre":  va_m["macro_pre"],
            "val/macro_rec":  va_m["macro_rec"],
        })

        print(f"  Epoch {epoch:03d}  tr_loss={tr_loss:.4f}  va_loss={va_loss:.4f}  "
              f"acc={va_acc:.4f}  f1={va_m['macro_f1']:.4f}")

        if va_m["macro_f1"] > best_f1:
            best_f1 = va_m["macro_f1"]
            ckpt = save_checkpoint(model, "classifier", args.save_dir,
                                   epoch, best_f1)
            print(f"  Saved → {ckpt}")

    wandb.finish()
    print(f"Best val macro-F1: {best_f1:.4f}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      default=r"D:\code\repo\DL_Assigment_2\temp")
    p.add_argument("--epochs",         type=int,   default=60)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=5e-4)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--save_dir",       default="checkpoints")
    p.add_argument("--wandb_project",  default="da6401_a2")
    p.add_argument("--ablation",       action="store_true",
                   help="Also run BN / Dropout ablation variants")
    args = p.parse_args()

    train_one_config(args, use_bn=True, dropout_p=0.5, run_tag="")

    if args.ablation:
        train_one_config(args, use_bn=False, dropout_p=0.5, run_tag="_nobn")
        train_one_config(args, use_bn=True,  dropout_p=0.2, run_tag="_dp02")
        train_one_config(args, use_bn=True,  dropout_p=0.0, run_tag="_nodp")


if __name__ == "__main__":
    main()