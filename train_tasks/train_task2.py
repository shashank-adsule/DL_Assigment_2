"""
train_task2.py  —  Bounding-Box Localization (Task 2)

Usage:
    python train_task2.py --data_root /path/to/oxford_pets \
                          --vgg11_ckpt outputs/vgg11_bn1_dp0.5_best.pt \
                          --epochs 20 --batch_size 32
"""

import argparse
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb

from models import VGG11Encoder, LocalizationModel, IoULoss
from data   import get_dataloaders
from utils  import (Trainer, compute_iou_batch, compute_map,
                    init_wandb, log_images_bbox)


# ---------------------------------------------------------------------------
# Loss wrapper
# ---------------------------------------------------------------------------
_iou_loss = IoULoss()


def loc_loss_fn(outputs, batch):
    """outputs = (B, 4), batch = (images, bboxes)"""
    pred_boxes = outputs
    gt_boxes   = batch[1].to(pred_boxes.device)
    loss = _iou_loss(pred_boxes, gt_boxes)
    return loss, {}


def loc_metric_fn(all_outputs, all_batches):
    preds = torch.cat(all_outputs, dim=0)
    gts   = torch.cat([b[1] for b in all_batches], dim=0)
    ious  = compute_iou_batch(preds, gts)
    mAP   = compute_map([preds], [gts])
    return {
        "val/mean_iou": ious.mean().item(),
        "val/mAP":      mAP,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",      required=True)
    parser.add_argument("--vgg11_ckpt",     required=True,
                        help="Path to best Task-1 checkpoint (.pt)")
    parser.add_argument("--freeze_backbone",action="store_true", default=True)
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--save_dir",       default="outputs")
    parser.add_argument("--wandb_project",  default="da6401_a2")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    init_wandb(project=args.wandb_project, run_name="task2_localization",
               config=vars(args))

    # Load pretrained VGG11
    vgg11 = VGG11(num_classes=37)
    ckpt  = torch.load(args.vgg11_ckpt, map_location="cpu")
    vgg11.load_state_dict(ckpt["model"])
    print(f"Loaded VGG11 from {args.vgg11_ckpt} (epoch {ckpt['epoch']})")

    model = LocalizationModel(vgg11, freeze_backbone=args.freeze_backbone)

    # Only optimise unfrozen parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        root=args.data_root, task="localization",
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loc_loss_fn,
        metric_fn=loc_metric_fn,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir,
        run_name="task2_localization",
    )
    trainer.fit(args.epochs)

    # --- W&B report: bbox overlay table ---
    model.eval()
    images, gt_boxes = next(iter(test_loader))
    images   = images.to(device)
    gt_boxes = gt_boxes.to(device)
    with torch.no_grad():
        pred_boxes = model(images)
    from utils.metrics import compute_iou_batch
    ious = compute_iou_batch(pred_boxes.cpu(), gt_boxes.cpu())
    log_images_bbox(images, pred_boxes.cpu(), gt_boxes.cpu(), ious, n=10)

    wandb.finish()


if __name__ == "__main__":
    main()
