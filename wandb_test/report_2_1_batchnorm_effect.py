"""
report_2_1_batchnorm_effect.py
==============================
W&B Report Section 2.1 — The Regularization Effect of BatchNorm

What this script does:
  1. Builds two tiny VGG11-style models: one WITH BatchNorm, one WITHOUT.
  2. Trains both on the Pet classification task for a fixed number of epochs.
  3. After each epoch logs:
       - train/loss and val/loss for both models (overlaid in W&B)
       - Activation histogram of the 3rd conv layer on a fixed probe image
  4. At the end logs a side-by-side activation distribution plot.

Run:
    python report_2_1_batchnorm_effect.py \
        --data_root /path/to/oxford_pets \
        --epochs 15 \
        --batch_size 32 \
        --wandb_project da6401_a2_report

Outputs logged to W&B:
  - train_loss / val_loss curves for both runs
  - Activation histograms at epochs 1, mid, final
  - Final matplotlib figure comparing distributions
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import wandb

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.layers import CustomDropout
from data.dataset  import get_dataloaders


# ---------------------------------------------------------------------------
# Build models
# ---------------------------------------------------------------------------
def make_vgg11_small(use_bn: bool, num_classes: int = 37) -> nn.Module:
    """Two-block mini VGG11 for quick ablation (same topology, BN toggle)."""
    layers = []

    def conv_block(in_ch, out_ch):
        block = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
        if use_bn:
            block.append(nn.BatchNorm2d(out_ch))
        block.append(nn.ReLU(inplace=True))
        return block

    # Block 1
    layers += conv_block(3,   64);  layers.append(nn.MaxPool2d(2, 2))
    # Block 2
    layers += conv_block(64,  128); layers.append(nn.MaxPool2d(2, 2))
    # Block 3 (this is the layer whose activations we probe)
    layers += conv_block(128, 256); layers.append(nn.MaxPool2d(2, 2))
    # Block 4
    layers += conv_block(256, 512); layers.append(nn.MaxPool2d(2, 2))
    # Block 5
    layers += conv_block(512, 512); layers.append(nn.MaxPool2d(2, 2))

    features = nn.Sequential(*layers)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = features
            self.pool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 1024),
                nn.ReLU(inplace=True),
                CustomDropout(0.5),
                nn.Linear(1024, num_classes),
            )
        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

    return Model()


# ---------------------------------------------------------------------------
# Activation hook — captures output of a specific layer
# ---------------------------------------------------------------------------
def get_activation(model: nn.Module, layer_index: int,
                   probe_img: torch.Tensor) -> np.ndarray:
    """Run probe_img through model, return flattened activations at layer_index."""
    captured = {}
    def hook(m, inp, out):
        captured['act'] = out.detach().cpu().numpy()
    h = list(model.features.children())[layer_index].register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        model(probe_img)
    h.remove()
    return captured['act'].flatten()


# ---------------------------------------------------------------------------
# One epoch helpers
# ---------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train(train)
    total_loss, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',     required=True)
    parser.add_argument('--epochs',        type=int,   default=15)
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--num_workers',   type=int,   default=4)
    parser.add_argument('--wandb_project', default='da6401_a2_report')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    wandb.init(project=args.wandb_project, name='2.1_batchnorm_effect',
               config=vars(args))

    train_loader, val_loader, _ = get_dataloaders(
        root=args.data_root, task='classification',
        batch_size=args.batch_size, num_workers=args.num_workers)

    # Fixed probe image (first val batch, single image)
    probe_img, _ = next(iter(val_loader))
    probe_img = probe_img[:1].to(device)

    criterion = nn.CrossEntropyLoss()

    # Build both models
    models = {
        'with_bn':    make_vgg11_small(use_bn=True).to(device),
        'without_bn': make_vgg11_small(use_bn=False).to(device),
    }
    optimizers  = {k: Adam(m.parameters(), lr=args.lr) for k, m in models.items()}
    schedulers  = {k: ReduceLROnPlateau(optimizers[k], patience=3, factor=0.5)
                   for k in models}

    # 3rd conv layer index in features Sequential: index 6 (conv of block 3)
    PROBE_LAYER = 6

    for epoch in range(1, args.epochs + 1):
        log = {'epoch': epoch}

        for tag, model in models.items():
            train_loss = run_epoch(model, train_loader, optimizers[tag],
                                   criterion, device, train=True)
            val_loss   = run_epoch(model, val_loader,   optimizers[tag],
                                   criterion, device, train=False)
            schedulers[tag].step(val_loss)

            log[f'{tag}/train_loss'] = train_loss
            log[f'{tag}/val_loss']   = val_loss

            # Activation histogram every epoch
            acts = get_activation(model, PROBE_LAYER, probe_img)
            log[f'{tag}/activation_hist_layer3'] = wandb.Histogram(acts)

            print(f'  Epoch {epoch} [{tag}] train={train_loss:.4f} val={val_loss:.4f}')

        wandb.log(log, step=epoch)

    # --- Final side-by-side distribution plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (tag, model) in zip(axes, models.items()):
        acts = get_activation(model, PROBE_LAYER, probe_img)
        ax.hist(acts, bins=80, color='steelblue' if 'with' in tag else 'coral',
                alpha=0.8, edgecolor='none')
        ax.set_title(f'3rd Conv activations — {tag}')
        ax.set_xlabel('Activation value')
        ax.set_ylabel('Count')
        mean, std = acts.mean(), acts.std()
        ax.axvline(mean, color='black', linestyle='--', linewidth=1,
                   label=f'mean={mean:.2f}  std={std:.2f}')
        ax.legend(fontsize=9)
    fig.suptitle('Activation distribution at 3rd Conv layer (same probe image)',
                 fontsize=13)
    plt.tight_layout()
    wandb.log({'activation_distribution_comparison': wandb.Image(fig)})
    plt.close(fig)

    print('\nSection 2.1 complete. Check your W&B dashboard.')
    wandb.finish()


if __name__ == '__main__':
    main()
