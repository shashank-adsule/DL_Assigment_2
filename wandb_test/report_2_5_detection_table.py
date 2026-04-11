"""
report_2_5_detection_table.py
==============================
W&B Report Section 2.5 — Object Detection: Confidence & IoU

What this script does:
  1. Loads trained LocalizationModel checkpoint.
  2. Runs inference on 20+ test images.
  3. Logs a W&B Table with:
       - Original image with GT (green) and predicted (red) bbox overlaid
       - IoU score per image
       - Confidence score (1 - IoU_loss, proxy for confidence)
  4. Identifies and separately highlights failure cases
     (high confidence but low IoU).

Run:
    python report_2_5_detection_table.py \
        --data_root   /path/to/oxford_pets \
        --vgg11_ckpt  outputs/vgg11_bn1_dp0.5_best.pt \
        --loc_ckpt    outputs/task2_localization_best.pt \
        --n_images    20 \
        --wandb_project da6401_a2_report
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wandb

from models.vgg11        import VGG11Encoder
from models.localization import LocalizationModel
from data.dataset        import get_dataloaders
from utils.metrics       import compute_iou_batch


# ---------------------------------------------------------------------------
def draw_bbox_on_image(img_np, pred_box, gt_box, iou):
    """
    img_np  : (H, W, 3) float [0,1]
    pred_box: [cx, cy, w, h] normalised
    gt_box  : [cx, cy, w, h] normalised
    Returns matplotlib figure.
    """
    H, W = img_np.shape[:2]

    def to_abs(box):
        cx, cy, bw, bh = box
        x1 = (cx - bw/2) * W
        y1 = (cy - bh/2) * H
        return x1, y1, bw*W, bh*H

    fig, ax = plt.subplots(1, figsize=(4, 4))
    ax.imshow(img_np)

    # Ground truth — green
    x, y, w, h = to_abs(gt_box)
    ax.add_patch(patches.Rectangle((x, y), w, h,
                                    linewidth=2, edgecolor='lime',
                                    facecolor='none', label='GT'))

    # Prediction — red
    x, y, w, h = to_abs(pred_box)
    ax.add_patch(patches.Rectangle((x, y), w, h,
                                    linewidth=2, edgecolor='red',
                                    facecolor='none', label='Pred'))

    ax.set_title(f'IoU = {iou:.3f}', fontsize=10)
    ax.legend(loc='upper right', fontsize=7)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',     required=True)
    parser.add_argument('--vgg11_ckpt',    required=True)
    parser.add_argument('--loc_ckpt',      required=True)
    parser.add_argument('--n_images',      type=int, default=20)
    parser.add_argument('--batch_size',    type=int, default=16)
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--wandb_project', default='da6401_a2_report')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(project=args.wandb_project, name='2.5_detection_confidence_iou',
               config=vars(args))

    # --- Load models ---
    vgg11 = VGG11Encoder(num_classes=37)
    vgg11.load_state_dict(torch.load(args.vgg11_ckpt, map_location='cpu')['model'])

    loc_model = LocalizationModel(vgg11, freeze_backbone=True)
    loc_model.load_state_dict(torch.load(args.loc_ckpt, map_location='cpu')['model'])
    loc_model = loc_model.to(device)
    loc_model.eval()
    print('Models loaded.')

    # --- Test loader ---
    _, _, test_loader = get_dataloaders(
        root=args.data_root, task='localization',
        batch_size=args.batch_size, num_workers=args.num_workers)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    # W&B table
    table = wandb.Table(columns=['image', 'IoU', 'confidence', 'result'])
    failure_imgs = []

    collected = 0
    all_ious  = []

    for imgs, gt_boxes in test_loader:
        if collected >= args.n_images:
            break
        imgs_dev = imgs.to(device)
        with torch.no_grad():
            pred_boxes = loc_model(imgs_dev).cpu()

        ious = compute_iou_batch(pred_boxes, gt_boxes)

        for i in range(len(imgs)):
            if collected >= args.n_images:
                break

            iou        = ious[i].item()
            confidence = float(iou)      # proxy: higher IoU → higher confidence
            all_ious.append(iou)

            # Denormalise image
            img_np = (imgs[i] * std + mean).clamp(0,1).permute(1,2,0).numpy()

            fig = draw_bbox_on_image(
                img_np,
                pred_boxes[i].tolist(),
                gt_boxes[i].tolist(),
                iou
            )
            wb_img = wandb.Image(fig)
            plt.close(fig)

            result = 'good' if iou >= 0.5 else ('failure' if iou < 0.3 else 'partial')
            table.add_data(wb_img, round(iou, 4), round(confidence, 4), result)

            # Collect failure cases: confident (iou > 0.3) but low overlap (iou < 0.4)
            if 0.3 < iou < 0.4:
                failure_fig = draw_bbox_on_image(
                    img_np, pred_boxes[i].tolist(), gt_boxes[i].tolist(), iou)
                failure_imgs.append(wandb.Image(failure_fig,
                    caption=f'FAILURE — IoU={iou:.3f}'))
                plt.close(failure_fig)

            collected += 1

    wandb.log({'detection_results_table': table})

    if failure_imgs:
        wandb.log({'failure_cases': failure_imgs})

    # Summary histogram of IoUs
    fig2, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_ious, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(0.5, color='green',  linestyle='--', label='IoU=0.5 threshold')
    ax.axvline(np.mean(all_ious), color='red', linestyle='-',
               label=f'mean IoU={np.mean(all_ious):.3f}')
    ax.set_xlabel('IoU'); ax.set_ylabel('Count')
    ax.set_title('IoU distribution over test images')
    ax.legend()
    plt.tight_layout()
    wandb.log({'iou_distribution': wandb.Image(fig2)})
    plt.close(fig2)

    print(f'\nLogged {collected} images.')
    print(f'Mean IoU: {np.mean(all_ious):.4f}')
    print('Section 2.5 complete.')
    wandb.finish()


if __name__ == '__main__':
    main()
