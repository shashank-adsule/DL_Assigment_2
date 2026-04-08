"""
train_task4.py  —  Unified Multi-Task Pipeline (Task 4)

Loads the best weights from Tasks 1–3 into corresponding parts of the
MultiTaskVGG11 model, then trains end-to-end with the combined loss.

Usage:
    python train_task4.py --data_root /path/to/oxford_pets \
                          --vgg11_ckpt  outputs/vgg11_bn1_dp0.5_best.pt \
                          --unet_ckpt   outputs/task3_unet_full_best.pt \
                          --epochs 30
"""

import argparse
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

from models import MultiTaskVGG11, MultiTaskLoss
from data   import get_dataloaders
from utils  import (Trainer, compute_f1_macro, compute_dice,
                    compute_pixel_acc, compute_map, compute_iou_batch,
                    init_wandb, log_seg_samples)


# ---------------------------------------------------------------------------
# Loss / metric wrappers for multi-task batch
# ---------------------------------------------------------------------------
_mt_loss = None   # initialised in main after args are parsed


def mt_loss_fn(outputs, batch):
    """
    outputs = (cls_logits, bbox, seg_logits)
    batch   = (images, labels, bboxes, masks)
    """
    cls_logits, bbox_pred, seg_logits = outputs
    _, labels, bboxes, masks = batch
    device = cls_logits.device

    loss, extra = _mt_loss(
        cls_logits, bbox_pred, seg_logits,
        labels.to(device), bboxes.to(device), masks.to(device)
    )
    return loss, extra


def mt_metric_fn(all_outputs, all_batches):
    all_cls   = torch.cat([o[0] for o in all_outputs])
    all_bbox  = torch.cat([o[1] for o in all_outputs])
    all_seg   = torch.cat([o[2] for o in all_outputs])

    all_labels = torch.cat([b[1] for b in all_batches])
    all_gts    = torch.cat([b[2] for b in all_batches])
    all_masks  = torch.cat([b[3] for b in all_batches])

    preds_cls = all_cls.argmax(dim=1).numpy()
    f1  = compute_f1_macro(preds_cls.tolist(), all_labels.numpy().tolist())
    mAP = compute_map([all_bbox], [all_gts])
    dice = compute_dice(all_seg, all_masks, num_classes=3)
    pxacc = compute_pixel_acc(all_seg, all_masks)

    return {
        "val/f1_macro":  f1,
        "val/mAP":       mAP,
        "val/dice":      dice,
        "val/pixel_acc": pxacc,
    }


# ---------------------------------------------------------------------------
# Weight initialisation from single-task checkpoints
# ---------------------------------------------------------------------------
def load_pretrained_weights(model: MultiTaskVGG11, vgg11_ckpt: str, unet_ckpt: str = None):
    """
    Load encoder weights from VGG11 Task-1 checkpoint.
    Optionally load seg decoder from UNet Task-3 checkpoint.
    """
    # ---- Encoder from VGG11 ----
    vgg_sd = torch.load(vgg11_ckpt, map_location="cpu")["model"]
    ranges = [(0, 3, "enc1"), (4, 7, "enc2"), (8, 14, "enc3"),
              (15, 21, "enc4"), (22, 28, "enc5")]

    for start, end, attr in ranges:
        block = getattr(model, attr)
        sub_sd = {}
        for k, v in vgg_sd.items():
            if not k.startswith("features."):
                continue
            idx = int(k.split(".")[1])
            if start <= idx < end:
                sub_key = ".".join(k.split(".")[2:])
                sub_sd[f"{idx - start}.{sub_key}"] = v
        block.load_state_dict(sub_sd, strict=False)

    # ---- Classifier head from VGG11 ----
    cls_sd = {k.replace("classifier.", ""): v
              for k, v in vgg_sd.items() if k.startswith("classifier.")}
    model.cls_head.load_state_dict(cls_sd, strict=False)

    print(f"  Loaded encoder + cls head from {vgg11_ckpt}")

    # ---- Seg decoder from UNet (optional) ----
    if unet_ckpt and os.path.exists(unet_ckpt):
        unet_sd = torch.load(unet_ckpt, map_location="cpu")["model"]
        decoder_keys = ["bottleneck", "up5", "dec5", "up4", "dec4",
                        "up3", "dec3", "up2", "dec2", "up1", "dec1", "seg_out"]
        dec_sd = {k: v for k, v in unet_sd.items()
                  if any(k.startswith(dk) for dk in decoder_keys)}
        missing, unexpected = model.load_state_dict(dec_sd, strict=False)
        print(f"  Loaded seg decoder from {unet_ckpt} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global _mt_loss

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",     required=True)
    parser.add_argument("--vgg11_ckpt",    required=True)
    parser.add_argument("--unet_ckpt",     default=None)
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--lr",            type=float, default=5e-5)
    parser.add_argument("--lambda_cls",    type=float, default=1.0)
    parser.add_argument("--lambda_loc",    type=float, default=1.0)
    parser.add_argument("--lambda_seg",    type=float, default=1.0)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--save_dir",      default="outputs")
    parser.add_argument("--wandb_project", default="da6401_a2")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    init_wandb(project=args.wandb_project, run_name="task4_multitask",
               config=vars(args))

    model = MultiTaskVGG11(num_classes=37, num_seg_classes=3)
    load_pretrained_weights(model, args.vgg11_ckpt, args.unet_ckpt)

    _mt_loss = MultiTaskLoss(
        num_seg_classes=3,
        lambda_cls=args.lambda_cls,
        lambda_loc=args.lambda_loc,
        lambda_seg=args.lambda_seg,
    )

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, patience=4, factor=0.5, verbose=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        root=args.data_root, task="multitask",
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=mt_loss_fn,
        metric_fn=mt_metric_fn,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir,
        run_name="task4_multitask",
    )
    trainer.fit(args.epochs)

    # Final showcase: log seg samples
    model.eval()
    batch = next(iter(test_loader))
    images = batch[0].to(device)
    with torch.no_grad():
        _, _, seg_logits = model(images)
    log_seg_samples(images.cpu(), batch[3], seg_logits.cpu(), n=5,
                    step=args.epochs)

    wandb.finish()


if __name__ == "__main__":
    main()
