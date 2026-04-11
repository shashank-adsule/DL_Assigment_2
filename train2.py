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

from torch import amp  # ✅ NEW AMP API

import wandb
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    precision_recall_fscore_support, confusion_matrix,
)

from data.dataset          import OxfordPetDataset, collate_fn
from models.classification import PetClassifier
from models.localization   import LocalizationModel
from models.segmentation   import UNetVGG11, DiceCELoss
from losses.iou_loss       import IoULoss

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_BREEDS = 37
IMAGE_SIZE = 224.0

# =========================================================
# Utils
# =========================================================

def make_loaders(args, mode):
    tr = OxfordPetDataset(args.data_dir, partition="train", mode=mode)
    va = OxfordPetDataset(args.data_dir, partition="val",   mode=mode)

    kw = dict(num_workers=args.num_workers, pin_memory=True,
              collate_fn=collate_fn)

    return (
        DataLoader(tr, batch_size=args.batch_size, shuffle=True,  **kw),
        DataLoader(va, batch_size=args.batch_size, shuffle=False, **kw)
    )

# =========================================================
# TASK 1: CLASSIFICATION
# =========================================================

def train_classification(args):
    wandb.init(project=args.wandb_project, name="classification", config=vars(args))

    tr_dl, va_dl = make_loaders(args, mode="cls")

    model = PetClassifier(num_classes=NUM_BREEDS, drop_rate=args.dropout_p).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    scaler = amp.GradScaler(AMP_DEVICE)  # ✅ NEW

    for epoch in range(1, args.epochs + 1):
        model.train()

        for batch in tr_dl:
            imgs   = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()

            with amp.autocast(AMP_DEVICE):  # ✅ NEW
                logits = model(imgs)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        print(f"[Clf {epoch}] loss={loss.item():.4f}")

# =========================================================
# TASK 2: LOCALIZATION
# =========================================================

def train_localization(args):
    wandb.init(project=args.wandb_project, name="localization", config=vars(args))

    tr_dl, va_dl = make_loaders(args, mode="loc")

    model = LocalizationModel(dropout_p=args.dropout_p).to(DEVICE)

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    scaler = amp.GradScaler(AMP_DEVICE)

    for epoch in range(1, args.epochs + 1):
        model.train()

        for batch in tr_dl:
            imgs      = batch["image"].to(DEVICE)
            bbox      = batch["bbox"].to(DEVICE)
            bbox_mask = batch["bbox_mask"].to(DEVICE).bool()

            if bbox_mask.sum() == 0:
                continue

            optimizer.zero_grad()

            with amp.autocast(AMP_DEVICE):
                pred = model(imgs)
                loss = mse_loss(pred, bbox) + iou_loss(pred, bbox)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        print(f"[Loc {epoch}] loss={loss.item():.4f}")

# =========================================================
# TASK 3: SEGMENTATION
# =========================================================

def train_segmentation(args):
    wandb.init(project=args.wandb_project, name="segmentation", config=vars(args))

    # batch size 8
    seg_batch_size = [8,16][1]   # safe
    # or 16 if AMP works well
    tr_dl, va_dl = (
        DataLoader(
            OxfordPetDataset(args.data_dir, partition="train", mode="seg"),
            batch_size=seg_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        ),
        DataLoader(
            OxfordPetDataset(args.data_dir, partition="val", mode="seg"),
            batch_size=seg_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    )

    model = UNetVGG11(num_classes=3, in_channels=3,
                      dropout_p=args.dropout_p).to(DEVICE)

    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    scaler = amp.GradScaler(AMP_DEVICE)

    for epoch in range(1, args.epochs + 1):
        model.train()

        for batch in tr_dl:
            imgs  = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            optimizer.zero_grad()

            with amp.autocast(AMP_DEVICE):
                logits = model(imgs)
                loss   = ce_loss(logits, masks)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        print(f"[Seg {epoch}] loss={loss.item():.4f}")

# =========================================================
# CLI
# =========================================================

def parse_args():
    p = argparse.ArgumentParser(description="Training with AMP")
    p.add_argument("--task", choices=["classification", "localization", "segmentation"], default="classification")
    p.add_argument("--data_dir", default=r"D:\code\repo\DL_Assigment_2\temp")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--dropout_p", type=float, default=0.5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--wandb_project", default="da6401-assignment2")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"\u001b[33mClassification\u001b[0m")
    train_classification(args)
    print(f"\n\u001b[33mLocalization\u001b[0m")
    train_localization(args)
    print(f"\n\u001b[33mSegmentation\u001b[0m")
    train_segmentation(args)

    # if args.task == "classification":
    #     train_classification(args)
    # elif args.task == "localization":
    #     train_localization(args)
    # elif args.task == "segmentation":
    #     train_segmentation(args)
