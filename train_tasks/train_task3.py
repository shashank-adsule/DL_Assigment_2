"""
train_task3.py  —  U-Net Segmentation (Task 3)

Runs three transfer-learning strategies for W&B report section 2.3:
  1. frozen   — entire VGG11 encoder frozen
  2. partial  — last 2 enc blocks unfrozen
  3. full     — entire network fine-tuned

Usage:
    python train_task3.py --data_root /path/to/oxford_pets \
                          --vgg11_ckpt outputs/vgg11_bn1_dp0.5_best.pt \
                          --strategy frozen  # or partial / full / all
"""

import argparse
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

from models import UNetVGG11, DiceCELoss, VGG11
from data   import get_dataloaders
from utils  import (Trainer, compute_dice, compute_pixel_acc,
                    init_wandb, log_seg_samples)


# ---------------------------------------------------------------------------
# Loss / metric wrappers
# ---------------------------------------------------------------------------
_dice_ce = DiceCELoss(num_classes=3)


def seg_loss_fn(outputs, batch):
    logits = outputs
    masks  = batch[1].to(logits.device)
    loss   = _dice_ce(logits, masks)
    return loss, {}


def seg_metric_fn(all_outputs, all_batches):
    logits = torch.cat(all_outputs,            dim=0)
    masks  = torch.cat([b[1] for b in all_batches], dim=0)
    return {
        "val/dice":      compute_dice(logits, masks, num_classes=3),
        "val/pixel_acc": compute_pixel_acc(logits, masks),
    }


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------
def apply_strategy(model: UNetVGG11, strategy: str):
    """
    frozen  : all encoder blocks frozen
    partial : enc1-enc3 frozen, enc4-enc5 trainable
    full    : everything trainable
    """
    enc_blocks = [model.enc1, model.enc2, model.enc3, model.enc4, model.enc5]

    if strategy == "frozen":
        for block in enc_blocks:
            for p in block.parameters():
                p.requires_grad = False

    elif strategy == "partial":
        for block in enc_blocks[:3]:   # freeze early blocks
            for p in block.parameters():
                p.requires_grad = False
        for block in enc_blocks[3:]:   # unfreeze last 2
            for p in block.parameters():
                p.requires_grad = True

    elif strategy == "full":
        for p in model.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Strategy '{strategy}': {trainable:,} / {total:,} params trainable")


def load_encoder_weights(model: UNetVGG11, vgg11_ckpt: str):
    """Transfer VGG11 weights into UNet encoder blocks."""
    ckpt = torch.load(vgg11_ckpt, map_location="cpu")
    sd   = ckpt["model"]

    # Map features.N.* → enc blocks by layer index
    # VGG11.features layout (0-indexed):
    # Block1: 0-2 (conv,bn,relu) → enc1
    # Block2: 4-6                → enc2
    # Block3: 8-13               → enc3
    # Block4: 15-20              → enc4
    # Block5: 22-27              → enc5
    ranges = [(0, 3, "enc1"), (4, 7, "enc2"), (8, 14, "enc3"),
              (15, 21, "enc4"), (22, 28, "enc5")]

    for start, end, attr in ranges:
        block: torch.nn.Sequential = getattr(model, attr)
        sub_sd = {}
        for k, v in sd.items():
            if not k.startswith("features."):
                continue
            idx = int(k.split(".")[1])
            if start <= idx < end:
                sub_key = ".".join(k.split(".")[2:])
                # map into block's 0-indexed children
                local_idx = idx - start
                # find param name within block
                sub_sd[f"{local_idx}.{sub_key}"] = v
        missing, unexpected = block.load_state_dict(sub_sd, strict=False)
        # Partial load is acceptable; report if needed.

    print(f"  Encoder weights loaded from {vgg11_ckpt}")


# ---------------------------------------------------------------------------
# Train one strategy
# ---------------------------------------------------------------------------
def train_strategy(args, strategy: str):
    run_name = f"task3_unet_{strategy}"
    init_wandb(project=args.wandb_project, run_name=run_name,
               config={**vars(args), "strategy": strategy})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}  |  Strategy: {strategy}")

    model = UNetVGG11(num_classes=3)
    load_encoder_weights(model, args.vgg11_ckpt)
    apply_strategy(model, strategy)

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        root=args.data_root, task="segmentation",
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=seg_loss_fn,
        metric_fn=seg_metric_fn,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir,
        run_name=run_name,
    )
    trainer.fit(args.epochs)

    # Log segmentation samples
    model.eval()
    images, masks = next(iter(test_loader))
    images = images.to(device)
    with torch.no_grad():
        logits = model(images)
    log_seg_samples(images.cpu(), masks, logits.cpu(), n=5, step=args.epochs)

    wandb.finish()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",     required=True)
    parser.add_argument("--vgg11_ckpt",    required=True)
    parser.add_argument("--strategy",      default="all",
                        choices=["frozen", "partial", "full", "all"])
    parser.add_argument("--epochs",        type=int,   default=25)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--save_dir",      default="outputs")
    parser.add_argument("--wandb_project", default="da6401_a2")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    strategies = ["frozen", "partial", "full"] if args.strategy == "all" else [args.strategy]
    for s in strategies:
        train_strategy(args, s)


if __name__ == "__main__":
    main()
