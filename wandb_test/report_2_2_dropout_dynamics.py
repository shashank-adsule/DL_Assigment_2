"""
report_2_2_dropout_dynamics.py  —  Section 2.2: Internal Dynamics (Dropout)

Trains PetClassifier under 3 dropout conditions, each as its own W&B run:
  (A) No Dropout      p = 0.0
  (B) Custom Dropout  p = 0.2
  (C) Custom Dropout  p = 0.5

Each run logs train_loss, val_loss, and generalisation_gap per epoch.
In W&B, select all 3 runs and overlay the curves to compare.

No checkpoint needed — trains from scratch.

Run from project root:
    python wandb_reports/report_2_2_dropout_dynamics.py --data_root /path/to/pets
"""

import os, sys, warnings
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.classification import PetClassifier
from data.dataset import OxfordPetDataset, collate_fn


def make_loaders(root, batch_size, num_workers):
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=True, collate_fn=collate_fn)
    return (DataLoader(OxfordPetDataset(root, partition="train", mode="cls"),
                       shuffle=True, **kw),
            DataLoader(OxfordPetDataset(root, partition="val",   mode="cls"),
                       shuffle=False, **kw))


def one_epoch(model, loader, opt, crit, device, train):
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)
            loss   = crit(model(imgs), labels)
            if train:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item(); n += 1
    return total / max(n, 1)


def run_one_config(args, dropout_p: float, run_name: str, device):
    wandb.init(project=args.wandb_project, name=run_name,
               config={**vars(args), "dropout_p": dropout_p}, reinit=True)

    tr_dl, va_dl = make_loaders(args.data_root, args.batch_size, args.num_workers)
    model = PetClassifier(num_classes=37, drop_rate=dropout_p).to(device)
    opt   = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    crit  = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(1, args.epochs + 1):
        tr  = one_epoch(model, tr_dl, opt, crit, device, True)
        val = one_epoch(model, va_dl, opt, crit, device, False)
        sched.step()
        gap = val - tr
        wandb.log({"train_loss": tr, "val_loss": val,
                   "generalisation_gap": gap,
                   "lr": opt.param_groups[0]["lr"],
                   "epoch": epoch}, step=epoch)
        print(f"  [{run_name} {epoch:02d}]  "
              f"tr={tr:.4f}  va={val:.4f}  gap={gap:+.4f}")
    wandb.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",     default=r"D:\code\repo\DL_Assigment_2\temp")
    ap.add_argument("--epochs",        type=int,   default=20)
    ap.add_argument("--batch_size",    type=int,   default=32)
    ap.add_argument("--lr",            type=float, default=5e-4)
    ap.add_argument("--num_workers",   type=int,   default=4)
    ap.add_argument("--wandb_project", default="da6401-assignment2")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for p, name in [(0.0, "2.2_no_dropout"),
                    (0.2, "2.2_dropout_p0.2"),
                    (0.5, "2.2_dropout_p0.5")]:
        print(f"\n{'='*55}\nRun: {name}\n{'='*55}")
        run_one_config(args, dropout_p=p, run_name=name, device=device)

    print("\nSection 2.2 done.")
    print("Tip: in W&B select all 3 runs → overlay train_loss, val_loss, "
          "and generalisation_gap.")


if __name__ == "__main__":
    main()