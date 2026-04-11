"""
report_2_2_dropout_dynamics.py
===============================
W&B Report Section 2.2 — Internal Dynamics of Custom Dropout

What this script does:
  Trains VGG11 under 3 dropout conditions and overlays the curves in W&B:
    (A) No Dropout      (p = 0.0)
    (B) Dropout p = 0.2
    (C) Dropout p = 0.5

  Logs per epoch:
    - train_loss / val_loss  (all 3 runs on same W&B chart via consistent keys)
    - generalisation_gap = val_loss - train_loss

Run:
    python report_2_2_dropout_dynamics.py \
        --data_root /path/to/oxford_pets \
        --epochs 20 \
        --batch_size 32 \
        --wandb_project da6401_a2_report
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.vgg11  import VGG11Encoder
from data.dataset  import get_dataloaders


# ---------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train(train)
    total_loss, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs), labels)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); n += 1
    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
def train_one_config(args, dropout_p: float, run_name: str, device):
    """Train a single dropout config; log to its own W&B run."""
    wandb.init(project=args.wandb_project, name=run_name,
               config={**vars(args), 'dropout_p': dropout_p},
               reinit=True)

    train_loader, val_loader, _ = get_dataloaders(
        root=args.data_root, task='classification',
        batch_size=args.batch_size, num_workers=args.num_workers)

    model = VGG11Encoder(num_classes=37, dropout_p=dropout_p).to(device)
    # If p=0, effectively disable dropout (CustomDropout with p=0 passes through)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        tr  = run_epoch(model, train_loader, optimizer, criterion, device, True)
        val = run_epoch(model, val_loader,   optimizer, criterion, device, False)
        scheduler.step(val)

        gap = val - tr
        wandb.log({
            'train_loss':        tr,
            'val_loss':          val,
            'generalisation_gap': gap,
            'lr':                optimizer.param_groups[0]['lr'],
            'epoch':             epoch,
        }, step=epoch)

        print(f'  [{run_name}] Epoch {epoch:02d}  train={tr:.4f}  '
              f'val={val:.4f}  gap={gap:.4f}')

    wandb.finish()


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',     required=True)
    parser.add_argument('--epochs',        type=int,   default=20)
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--num_workers',   type=int,   default=4)
    parser.add_argument('--wandb_project', default='da6401_a2_report')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    configs = [
        (0.0, '2.2_dropout_p0.0_no_dropout'),
        (0.2, '2.2_dropout_p0.2'),
        (0.5, '2.2_dropout_p0.5'),
    ]

    for p, name in configs:
        print(f'\n--- Training: {name} ---')
        train_one_config(args, dropout_p=p, run_name=name, device=device)

    print('\nSection 2.2 complete.')
    print('In W&B: go to your project → select all 3 runs → group by run name')
    print('→ plot train_loss and val_loss overlaid to see the generalisation gap.')


if __name__ == '__main__':
    main()
