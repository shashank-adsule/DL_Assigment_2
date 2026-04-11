"""
report_2_3_transfer_learning.py
================================
W&B Report Section 2.3 — Transfer Learning Showdown

Trains U-Net segmentation under 3 strategies and compares:
  (A) frozen   — entire VGG11 encoder frozen
  (B) partial  — first 3 enc blocks frozen, last 2 unfrozen
  (C) full     — all weights trainable

Logs per epoch (each strategy = its own W&B run):
  - train_loss, val_loss
  - val_dice, val_pixel_acc
  - trainable_params count
  - time_per_epoch (seconds)

Run:
    python report_2_3_transfer_learning.py \
        --data_root   /path/to/oxford_pets \
        --vgg11_ckpt  outputs/vgg11_bn1_dp0.5_best.pt \
        --epochs 20 \
        --batch_size 16 \
        --wandb_project da6401_a2_report
"""

import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.segmentation import UNetVGG11, DiceCELoss
from data.dataset        import get_dataloaders
from utils.metrics       import compute_dice, compute_pixel_acc


# ---------------------------------------------------------------------------
def apply_strategy(model: UNetVGG11, strategy: str):
    enc_blocks = [model.enc1, model.enc2, model.enc3, model.enc4, model.enc5]

    # First make everything trainable
    for p in model.parameters():
        p.requires_grad = True

    if strategy == 'frozen':
        for block in enc_blocks:
            for p in block.parameters():
                p.requires_grad = False

    elif strategy == 'partial':
        for block in enc_blocks[:3]:          # freeze early
            for p in block.parameters():
                p.requires_grad = False
        # enc4, enc5 remain trainable

    elif strategy == 'full':
        pass  # all already trainable

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f'  Strategy "{strategy}": {trainable:,} / {total:,} params trainable '
          f'({100*trainable/total:.1f}%)')
    return trainable


def load_encoder(model: UNetVGG11, ckpt_path: str):
    """Copy VGG11 backbone weights into UNet encoder blocks."""
    sd = torch.load(ckpt_path, map_location='cpu')['model']

    # features.N.* → enc block mapping
    ranges = [(0, 3, 'enc1'), (4, 7, 'enc2'), (8, 14, 'enc3'),
              (15, 21, 'enc4'), (22, 28, 'enc5')]

    for start, end, attr in ranges:
        block = getattr(model, attr)
        sub = {}
        for k, v in sd.items():
            if not k.startswith('features.'): continue
            idx = int(k.split('.')[1])
            if start <= idx < end:
                sub_key = '.'.join(k.split('.')[2:])
                sub[f'{idx - start}.{sub_key}'] = v
        block.load_state_dict(sub, strict=False)

    print(f'  Encoder weights loaded from {ckpt_path}')


# ---------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train(train)
    total_loss, n = 0.0, 0
    all_logits, all_masks = [], []

    with torch.set_grad_enabled(train):
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss   = criterion(logits, masks)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); n += 1
            if not train:
                all_logits.append(logits.detach().cpu())
                all_masks.append(masks.detach().cpu())

    metrics = {}
    if not train and all_logits:
        logits_cat = torch.cat(all_logits)
        masks_cat  = torch.cat(all_masks)
        metrics['val_dice']      = compute_dice(logits_cat, masks_cat)
        metrics['val_pixel_acc'] = compute_pixel_acc(logits_cat, masks_cat)

    return total_loss / max(n, 1), metrics


# ---------------------------------------------------------------------------
def train_strategy(args, strategy: str, device):
    run_name = f'2.3_transfer_{strategy}'
    wandb.init(project=args.wandb_project, name=run_name,
               config={**vars(args), 'strategy': strategy}, reinit=True)

    train_loader, val_loader, _ = get_dataloaders(
        root=args.data_root, task='segmentation',
        batch_size=args.batch_size, num_workers=args.num_workers)

    model = UNetVGG11(num_classes=3).to(device)

    if args.vgg11_ckpt and os.path.exists(args.vgg11_ckpt):
        load_encoder(model, args.vgg11_ckpt)

    trainable = apply_strategy(model, strategy)
    wandb.config.update({'trainable_params': trainable})

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = DiceCELoss(num_classes=3)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, _         = run_epoch(model, train_loader, optimizer,
                                       criterion, device, train=True)
        val_loss, val_mets = run_epoch(model, val_loader,   optimizer,
                                       criterion, device, train=False)
        epoch_time = time.time() - t0
        scheduler.step(val_loss)

        log = {
            'train_loss':      tr_loss,
            'val_loss':        val_loss,
            'time_per_epoch':  epoch_time,
            'lr':              optimizer.param_groups[0]['lr'],
            'epoch':           epoch,
            **val_mets,
        }
        wandb.log(log, step=epoch)
        print(f'  [{strategy}] Epoch {epoch:02d}  '
              f'train={tr_loss:.4f}  val={val_loss:.4f}  '
              f'dice={val_mets.get("val_dice", 0):.4f}  '
              f't={epoch_time:.1f}s')

    wandb.finish()


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',     required=True)
    parser.add_argument('--vgg11_ckpt',    default=None,
                        help='Path to Task-1 checkpoint (optional but recommended)')
    parser.add_argument('--strategy',      default='all',
                        choices=['frozen', 'partial', 'full', 'all'])
    parser.add_argument('--epochs',        type=int,   default=20)
    parser.add_argument('--batch_size',    type=int,   default=16)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--num_workers',   type=int,   default=4)
    parser.add_argument('--wandb_project', default='da6401_a2_report')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    strategies = ['frozen', 'partial', 'full'] if args.strategy == 'all' \
                 else [args.strategy]

    for s in strategies:
        print(f'\n=== Strategy: {s} ===')
        train_strategy(args, s, device)

    print('\nSection 2.3 complete.')
    print('In W&B: select the 3 runs and overlay val_dice / val_loss curves.')


if __name__ == '__main__':
    main()
