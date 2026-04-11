"""
report_2_6_segmentation_eval.py
================================
W&B Report Section 2.6 — Segmentation: Dice vs Pixel Accuracy

What this script does:
  1. Loads trained UNet checkpoint.
  2. Runs inference on the test set.
  3. Logs 5 sample triplets: [Original | GT mask | Predicted mask].
  4. Tracks and plots Pixel Accuracy vs Dice Score epoch-by-epoch on val set.
  5. Shows WHY pixel accuracy is misleading for imbalanced segmentation.

Run:
    python report_2_6_segmentation_eval.py \
        --data_root   /path/to/oxford_pets \
        --unet_ckpt   outputs/task3_unet_full_best.pt \
        --wandb_project da6401_a2_report
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb

from models.segmentation import UNetVGG11
from data.dataset        import get_dataloaders
from utils.metrics       import compute_dice, compute_pixel_acc


# Trimap colour palette: 0=foreground(pet), 1=background, 2=boundary
PALETTE = np.array([
    [255, 128,   0],   # orange  — foreground pet
    [ 70, 130, 180],   # steel blue — background
    [255, 255,   0],   # yellow  — boundary
], dtype=np.uint8)


def mask_to_rgb(mask_tensor):
    """mask_tensor: (H, W) long → (H, W, 3) uint8"""
    m = mask_tensor.cpu().numpy().clip(0, 2)
    return PALETTE[m]


def denorm(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (img_tensor * std + mean).clamp(0, 1).permute(1,2,0).numpy()


# ---------------------------------------------------------------------------
def log_sample_triplets(model, loader, device, n=5):
    """Log n sample triplets as W&B images."""
    model.eval()
    samples = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs_dev = imgs.to(device)
            logits   = model(imgs_dev)
            preds    = logits.argmax(dim=1).cpu()

            for i in range(len(imgs)):
                if len(samples) >= n:
                    break
                orig_np    = (denorm(imgs[i]) * 255).astype(np.uint8)
                gt_rgb     = mask_to_rgb(masks[i])
                pred_rgb   = mask_to_rgb(preds[i])

                fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
                axes[0].imshow(orig_np);  axes[0].set_title('Original');       axes[0].axis('off')
                axes[1].imshow(gt_rgb);   axes[1].set_title('Ground Truth');   axes[1].axis('off')
                axes[2].imshow(pred_rgb); axes[2].set_title('Predicted Mask'); axes[2].axis('off')
                plt.tight_layout()
                samples.append(wandb.Image(fig, caption=f'Sample {len(samples)+1}'))
                plt.close(fig)

            if len(samples) >= n:
                break

    wandb.log({'segmentation_samples': samples})


# ---------------------------------------------------------------------------
def compute_class_balance(loader):
    """Report the pixel-class distribution in the val set."""
    counts = torch.zeros(3)
    for _, masks in loader:
        for c in range(3):
            counts[c] += (masks == c).sum()
    total = counts.sum()
    return {f'class_{c}_pct': (counts[c]/total*100).item() for c in range(3)}


# ---------------------------------------------------------------------------
def analyse_metrics(model, loader, device):
    """Return dice and pixel_acc over the full loader."""
    model.eval()
    all_logits, all_masks = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            logits = model(imgs.to(device))
            all_logits.append(logits.cpu())
            all_masks.append(masks)
    logits_cat = torch.cat(all_logits)
    masks_cat  = torch.cat(all_masks)
    return (compute_dice(logits_cat, masks_cat),
            compute_pixel_acc(logits_cat, masks_cat))


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',     required=True)
    parser.add_argument('--unet_ckpt',     required=True)
    parser.add_argument('--batch_size',    type=int, default=8)
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--wandb_project', default='da6401_a2_report')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(project=args.wandb_project, name='2.6_dice_vs_pixelacc',
               config=vars(args))

    # --- Load model ---
    model = UNetVGG11(num_classes=3)
    ckpt  = torch.load(args.unet_ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    print('UNet loaded.')

    _, val_loader, test_loader = get_dataloaders(
        root=args.data_root, task='segmentation',
        batch_size=args.batch_size, num_workers=args.num_workers)

    # --- 1. Log sample triplets ---
    log_sample_triplets(model, test_loader, device, n=5)

    # --- 2. Compute metrics on val ---
    dice, pxacc = analyse_metrics(model, val_loader, device)
    wandb.log({'val/dice': dice, 'val/pixel_acc': pxacc})
    print(f'Val Dice={dice:.4f}  Pixel Acc={pxacc:.4f}')

    # --- 3. Class balance analysis ---
    balance = compute_class_balance(val_loader)
    wandb.log(balance)
    print('Class balance:', balance)

    # --- 4. Illustrative plot: why pixel acc is misleading ---
    # Simulate a naive "predict all background" baseline
    model.eval()
    bg_preds, all_masks = [], []
    with torch.no_grad():
        for imgs, masks in val_loader:
            # Fake prediction: all background (class 1)
            fake_logits = torch.zeros(masks.shape[0], 3, masks.shape[1], masks.shape[2])
            fake_logits[:, 1, :, :] = 10.0   # argmax always picks class 1 (bg)
            bg_preds.append(fake_logits)
            all_masks.append(masks)

    bg_logits = torch.cat(bg_preds)
    mk        = torch.cat(all_masks)
    naive_dice   = compute_dice(bg_logits, mk)
    naive_pxacc  = compute_pixel_acc(bg_logits, mk)

    fig, ax = plt.subplots(figsize=(7, 5))
    metrics = ['Pixel Accuracy', 'Dice Score']
    model_vals = [pxacc,      dice]
    naive_vals = [naive_pxacc, naive_dice]

    x = np.arange(len(metrics))
    w = 0.32
    ax.bar(x - w/2, model_vals, w, label='Trained UNet',       color='steelblue')
    ax.bar(x + w/2, naive_vals, w, label='"All Background" baseline', color='coral')
    ax.set_ylim(0, 1)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel('Score'); ax.set_title('Dice vs Pixel Accuracy — why Dice matters')
    ax.legend()
    for i, (mv, nv) in enumerate(zip(model_vals, naive_vals)):
        ax.text(i-w/2, mv+0.01, f'{mv:.3f}', ha='center', fontsize=9)
        ax.text(i+w/2, nv+0.01, f'{nv:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    wandb.log({'dice_vs_pixelacc_comparison': wandb.Image(fig)})
    plt.close(fig)

    print(f'\nNaive baseline — PixelAcc={naive_pxacc:.3f}  Dice={naive_dice:.3f}')
    print('This illustrates: high pixel accuracy does NOT mean good segmentation.')
    print('Section 2.6 complete.')
    wandb.finish()


if __name__ == '__main__':
    main()
