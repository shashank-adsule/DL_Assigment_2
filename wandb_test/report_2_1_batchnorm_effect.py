"""
report_2_1_batchnorm_effect.py  —  Section 2.1: Regularization Effect of BatchNorm

Trains PetClassifier with and without BatchNorm for --epochs epochs.
Captures activations at the 3rd convolutional block (block3[0]) on a
fixed probe image each epoch and logs them to W&B.

No checkpoint needed — trains from scratch.

Run from project root:
    python wandb_reports/report_2_1_batchnorm_effect.py --data_root /path/to/pets
"""

import os, sys, warnings
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def build_model(use_bn: bool) -> PetClassifier:
    model = PetClassifier(num_classes=37, drop_rate=0.5)
    if not use_bn:
        # Replace all BN layers with Identity
        for name, m in list(model.named_modules()):
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                parts = name.rsplit(".", 1)
                parent = model
                if len(parts) == 2:
                    for attr in parts[0].split("."):
                        parent = getattr(parent, attr)
                setattr(parent, parts[-1] if len(parts) == 2 else name,
                        nn.Identity())
    return model


def get_block3_acts(model, probe):
    """Hook into first conv of encoder.block3 and return flat activations."""
    buf = {}
    h = model.encoder.block3[0].register_forward_hook(
        lambda m, i, o: buf.update({"a": o.detach().cpu().numpy()}))
    model.eval()
    with torch.no_grad():
        model(probe)
    h.remove()
    return buf["a"].flatten()


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",     default=r"D:\code\repo\DL_Assigment_2\temp")
    ap.add_argument("--epochs",        type=int,   default=15)
    ap.add_argument("--batch_size",    type=int,   default=32)
    ap.add_argument("--lr",            type=float, default=5e-4)
    ap.add_argument("--num_workers",   type=int,   default=4)
    ap.add_argument("--wandb_project", default="da6401-assignment2")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project=args.wandb_project, name="2.1_batchnorm_effect",
               config=vars(args))

    tr_dl, va_dl = make_loaders(args.data_root, args.batch_size, args.num_workers)
    probe = next(iter(va_dl))["image"][:1].to(device)
    crit  = nn.CrossEntropyLoss(label_smoothing=0.1)

    configs = {
        "with_bn":    build_model(True).to(device),
        "without_bn": build_model(False).to(device),
    }
    opts   = {k: AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
               for k, m in configs.items()}
    scheds = {k: CosineAnnealingLR(opts[k], T_max=args.epochs)
               for k in configs}

    for epoch in range(1, args.epochs + 1):
        log = {"epoch": epoch}
        for tag, model in configs.items():
            tr  = one_epoch(model, tr_dl, opts[tag], crit, device, True)
            val = one_epoch(model, va_dl, opts[tag], crit, device, False)
            scheds[tag].step()
            acts = get_block3_acts(model, probe)
            log[f"{tag}/train_loss"] = tr
            log[f"{tag}/val_loss"]   = val
            log[f"{tag}/act_hist"]   = wandb.Histogram(acts)
            print(f"  Ep {epoch:02d} [{tag}]  tr={tr:.4f}  va={val:.4f}  "
                  f"act_mean={acts.mean():.3f}")
        wandb.log(log, step=epoch)

    # Side-by-side final activation distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (tag, model), color in zip(axes, configs.items(),
                                        ["#378ADD", "#D85A30"]):
        acts = get_block3_acts(model, probe)
        ax.hist(acts, bins=120, color=color, alpha=0.85, edgecolor="none")
        ax.axvline(acts.mean(), color="black", lw=1.5, ls="--",
                   label=f"mean={acts.mean():.3f}  std={acts.std():.3f}")
        ax.set_title(f"{'With' if 'with' in tag else 'Without'} BatchNorm — block3 activations")
        ax.set_xlabel("Activation value"); ax.set_ylabel("Count")
        ax.legend(fontsize=9)
    fig.suptitle("Block-3 conv activation distribution on same probe image")
    plt.tight_layout()
    wandb.log({"activation_distribution_comparison": wandb.Image(fig)})
    plt.close(fig)

    print("Section 2.1 done.")
    wandb.finish()


if __name__ == "__main__":
    main()