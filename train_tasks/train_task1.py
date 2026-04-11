import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.backends.cudnn.benchmark = True

import wandb
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import VGG11Encoder, CustomDropout
from data.dataset   import get_dataloaders
from utils  import (Trainer, compute_f1_macro, init_wandb,
                    log_feature_maps, log_activation_hist)


# ---------------------------------------------------------------------------
# Loss + metric wrappers for the Trainer
# ---------------------------------------------------------------------------
def cls_loss_fn(outputs, batch):
    """outputs = logits (B, 37), batch = (images, labels)"""
    logits = outputs
    labels = batch[1].to(logits.device)
    loss   = nn.CrossEntropyLoss()(logits, labels)
    return loss, {}


def cls_metric_fn(all_outputs, all_batches):
    preds  = torch.cat(all_outputs).argmax(dim=1).numpy()
    labels = torch.cat([b[1] for b in all_batches]).numpy()
    f1 = compute_f1_macro(preds.tolist(), labels.tolist())
    return {"val/f1_macro": f1}


# ---------------------------------------------------------------------------
# Train one configuration
# ---------------------------------------------------------------------------
def train_vgg11(args, use_bn: bool = True, dropout_p: float = 0.5,
                run_suffix: str = ""):
    run_name = f"vgg11_bn{int(use_bn)}_dp{dropout_p}{run_suffix}"
    config = vars(args) | {"use_bn": use_bn, "dropout_p": dropout_p}
    init_wandb(project=args.wandb_project, run_name=run_name, config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Run: {run_name}")

    train_loader, val_loader, _ = get_dataloaders(
        root=args.data_root, task="classification",
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = VGG11Encoder(num_classes=37, dropout_p=dropout_p).to(device)

    if not use_bn:
        # Remove BatchNorm layers for ablation study
        def remove_bn(module):
            for name, child in list(module.named_children()):
                if isinstance(child, nn.BatchNorm2d):
                    setattr(module, name, nn.Identity())
                else:
                    remove_bn(child)
        remove_bn(model)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=cls_loss_fn,
        metric_fn=cls_metric_fn,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir,
        run_name=run_name,
    )
    trainer.fit(args.epochs)

    # --- W&B report: feature map visualisation ---
    sample_img, _ = next(iter(val_loader))
    sample_img = sample_img[:1].to(device)
    log_feature_maps(model, sample_img, layer_indices=[0, -3], step=args.epochs)

    # --- W&B report: activation histogram ---
    log_activation_hist(model, sample_img, target_layer_idx=6,
                        step=args.epochs, tag=run_name)

    best_path = os.path.join(args.save_dir, f"{run_name}_best.pt")
    wandb.finish()
    return best_path


# ---------------------------------------------------------------------------
# Main: train base + ablation variants
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_root",      required=True)
    parser.add_argument("--data_root",      default=r"D:\code\repo\DL_Assigment_2\temp")
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--batch_size",     type=int,   default=[32,64][0])
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--save_dir",       default="outputs")
    parser.add_argument("--wandb_project",  default="da6401_a2")
    parser.add_argument("--ablation",       action="store_true",
                        



                        help="Also run BN and Dropout ablation variants")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Base model (BN + Dropout 0.5)
    best_ckpt = train_vgg11(args, use_bn=True, dropout_p=0.5)
    print(f"\nBase model saved: {best_ckpt}")

    if args.ablation:
        # Section 2.1: no BatchNorm
        train_vgg11(args, use_bn=False, dropout_p=0.5, run_suffix="_nobn")
        # Section 2.2: Dropout p=0.2
        train_vgg11(args, use_bn=True,  dropout_p=0.2, run_suffix="_dp02")
        # Section 2.2: No Dropout
        train_vgg11(args, use_bn=True,  dropout_p=0.0, run_suffix="_nodp")


if __name__ == "__main__":
    main()
