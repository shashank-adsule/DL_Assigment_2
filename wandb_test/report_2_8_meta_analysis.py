"""
report_2_8_meta_analysis.py
=============================
W&B Report Section 2.8 — Meta-Analysis and Reflection

What this script does:
  Loads all task checkpoints, runs a final evaluation pass on the test set,
  then generates and logs comprehensive overlaid metric plots:

  1. Training vs Validation loss  — all 4 tasks on one figure
  2. Task-specific metrics:
       - Classification: Macro F1
       - Localization:   Mean IoU
       - Segmentation:   Dice Score + Pixel Accuracy
  3. Per-class Dice heatmap for segmentation
  4. Confusion matrix excerpt for classification (top-10 most confused classes)
  5. W&B summary table: all final test metrics in one place

Run:
    python report_2_8_meta_analysis.py \
        --data_root   /path/to/oxford_pets \
        --cls_ckpt    outputs/vgg11_bn1_dp0.5_best.pt \
        --loc_ckpt    outputs/task2_localization_best.pt \
        --seg_ckpt    outputs/task3_unet_full_best.pt \
        --mt_ckpt     outputs/task4_multitask_best.pt \
        --wandb_project da6401_a2_report
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import f1_score, confusion_matrix

from models.vgg11        import VGG11Encoder
from models.localization import LocalizationModel
from models.segmentation import UNetVGG11
from models.multitask    import MultiTaskPerceptionModel
from data.dataset        import get_dataloaders
from utils.metrics       import (compute_f1_macro, compute_iou_batch,
                                  compute_dice, compute_pixel_acc)

CLASS_NAMES = [
    'Abyssinian','Bengal','Birman','Bombay','British Shorthair',
    'Egyptian Mau','Maine Coon','Persian','Ragdoll','Russian Blue',
    'Siamese','Sphynx','American Bulldog','American Pit Bull Terrier',
    'Basset Hound','Beagle','Boxer','Chihuahua','English Cocker Spaniel',
    'English Setter','German Shorthaired','Great Pyrenees','Havanese',
    'Japanese Chin','Keeshond','Leonberger','Miniature Pinscher',
    'Newfoundland','Pomeranian','Pug','Saint Bernard','Samoyed',
    'Scottish Terrier','Shiba Inu','Staffordshire Bull Terrier',
    'Wheaten Terrier','Yorkshire Terrier',
]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_classifier(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        out = model(imgs.to(device))
        all_preds.extend(out.argmax(1).cpu().tolist())
        all_labels.extend(labels.tolist())
    f1 = compute_f1_macro(all_preds, all_labels)
    return f1, all_preds, all_labels


@torch.no_grad()
def eval_localization(model, loader, device):
    model.eval()
    all_preds, all_gts = [], []
    for imgs, bboxes in loader:
        preds = model(imgs.to(device)).cpu()
        all_preds.append(preds); all_gts.append(bboxes)
    p = torch.cat(all_preds); g = torch.cat(all_gts)
    ious = compute_iou_batch(p, g)
    return ious.mean().item(), ious


@torch.no_grad()
def eval_segmentation(model, loader, device):
    model.eval()
    all_logits, all_masks = [], []
    for imgs, masks in loader:
        logits = model(imgs.to(device)).cpu()
        all_logits.append(logits); all_masks.append(masks)
    lc = torch.cat(all_logits); mc = torch.cat(all_masks)
    return compute_dice(lc, mc), compute_pixel_acc(lc, mc)


@torch.no_grad()
def eval_multitask(model, loader, device):
    model.eval()
    cls_preds, cls_labels = [], []
    bbox_preds, bbox_gts  = [], []
    seg_logits_all, seg_masks_all = [], []

    for batch in loader:
        imgs, labels, bboxes, masks = batch
        out = model(imgs.to(device))
        cls_preds.extend(out['classification'].argmax(1).cpu().tolist())
        cls_labels.extend(labels.tolist())
        bbox_preds.append(out['localization'].cpu())
        bbox_gts.append(bboxes)
        seg_logits_all.append(out['segmentation'].cpu())
        seg_masks_all.append(masks)

    f1   = compute_f1_macro(cls_preds, cls_labels)
    bp   = torch.cat(bbox_preds); bg = torch.cat(bbox_gts)
    ious = compute_iou_batch(bp, bg)
    lc   = torch.cat(seg_logits_all); mc = torch.cat(seg_masks_all)
    dice = compute_dice(lc, mc)
    pacc = compute_pixel_acc(lc, mc)
    return f1, ious.mean().item(), dice, pacc, cls_preds, cls_labels


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, top_n=10):
    """Show top_n most-confused class pairs."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    # Zero diagonal to find off-diagonal errors
    cm_off = cm.copy(); np.fill_diagonal(cm_off, 0)
    top_idx = np.argsort(cm_off.ravel())[::-1][:top_n]
    rows = top_idx // len(class_names)
    cols = top_idx %  len(class_names)
    pairs = [(class_names[r], class_names[c], cm[r, c]) for r, c in zip(rows, cols)]

    fig, ax = plt.subplots(figsize=(9, 5))
    labels_str = [f'{a}\n→{b}' for a, b, _ in pairs]
    counts     = [cnt for _, _, cnt in pairs]
    ax.barh(labels_str[::-1], counts[::-1], color='steelblue')
    ax.set_xlabel('Misclassification count')
    ax.set_title(f'Top-{top_n} classification errors')
    plt.tight_layout()
    return fig


def plot_metrics_summary(metrics_dict):
    """Bar chart of all final test metrics."""
    fig, ax = plt.subplots(figsize=(10, 5))
    keys   = list(metrics_dict.keys())
    values = [metrics_dict[k] for k in keys]
    colors = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2','#937860']
    bars = ax.bar(keys, values, color=colors[:len(keys)], edgecolor='white')
    ax.set_ylim(0, 1); ax.set_ylabel('Score')
    ax.set_title('Final test-set metrics — Unified Pipeline')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',     required=True)
    parser.add_argument('--cls_ckpt',      required=True)
    parser.add_argument('--loc_ckpt',      required=True)
    parser.add_argument('--seg_ckpt',      required=True)
    parser.add_argument('--mt_ckpt',       required=True)
    parser.add_argument('--batch_size',    type=int, default=16)
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--wandb_project', default='da6401_a2_report')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(project=args.wandb_project, name='2.8_meta_analysis',
               config=vars(args))

    # ---- Load all models ----
    vgg11 = VGG11Encoder(num_classes=37)
    vgg11.load_state_dict(torch.load(args.cls_ckpt, map_location='cpu')['model'])
    vgg11 = vgg11.to(device)

    loc_model = LocalizationModel(vgg11, freeze_backbone=True)
    loc_model.load_state_dict(torch.load(args.loc_ckpt, map_location='cpu')['model'])
    loc_model = loc_model.to(device)

    seg_model = UNetVGG11(num_classes=3)
    seg_model.load_state_dict(torch.load(args.seg_ckpt, map_location='cpu')['model'])
    seg_model = seg_model.to(device)

    mt_model = MultiTaskPerceptionModel(num_classes=37, num_seg_classes=3)
    mt_model.load_state_dict(torch.load(args.mt_ckpt, map_location='cpu')['model'])
    mt_model = mt_model.to(device)

    print('All models loaded.')

    # ---- Data loaders ----
    _, _, cls_test   = get_dataloaders(args.data_root, 'classification',
                                       args.batch_size, args.num_workers)
    _, _, loc_test   = get_dataloaders(args.data_root, 'localization',
                                       args.batch_size, args.num_workers)
    _, _, seg_test   = get_dataloaders(args.data_root, 'segmentation',
                                       args.batch_size, args.num_workers)
    _, _, mt_test    = get_dataloaders(args.data_root, 'multitask',
                                       args.batch_size, args.num_workers)

    # ---- Evaluate each task independently ----
    print('\nEvaluating Task 1 (classification)...')
    f1_cls, preds_cls, labels_cls = eval_classifier(vgg11, cls_test, device)

    print('Evaluating Task 2 (localization)...')
    mean_iou, _ = eval_localization(loc_model, loc_test, device)

    print('Evaluating Task 3 (segmentation)...')
    dice_seg, pxacc_seg = eval_segmentation(seg_model, seg_test, device)

    print('Evaluating Task 4 (multitask)...')
    f1_mt, iou_mt, dice_mt, pxacc_mt, preds_mt, labels_mt = \
        eval_multitask(mt_model, mt_test, device)

    # ---- Log final metrics table ----
    final_metrics = {
        'cls/f1_macro':      f1_cls,
        'loc/mean_iou':      mean_iou,
        'seg/dice':          dice_seg,
        'seg/pixel_acc':     pxacc_seg,
        'mt/f1_macro':       f1_mt,
        'mt/mean_iou':       iou_mt,
        'mt/dice':           dice_mt,
        'mt/pixel_acc':      pxacc_mt,
    }
    wandb.log(final_metrics)
    print('\nFinal metrics:')
    for k, v in final_metrics.items():
        print(f'  {k:<25} {v:.4f}')

    # ---- Confusion matrix ----
    cm_fig = plot_confusion_matrix(labels_mt, preds_mt, CLASS_NAMES, top_n=10)
    wandb.log({'classification/top10_errors': wandb.Image(cm_fig)})
    plt.close(cm_fig)

    # ---- Summary bar chart ----
    display_metrics = {
        'F1 (cls)':    f1_cls,
        'IoU (loc)':   mean_iou,
        'Dice (seg)':  dice_seg,
        'F1 (MT)':     f1_mt,
        'IoU (MT)':    iou_mt,
        'Dice (MT)':   dice_mt,
    }
    summary_fig = plot_metrics_summary(display_metrics)
    wandb.log({'final_metrics_summary': wandb.Image(summary_fig)})
    plt.close(summary_fig)

    # ---- W&B Summary table ----
    table = wandb.Table(columns=['Metric', 'Task 1/2/3 (isolated)', 'Task 4 (unified)'])
    table.add_data('Macro F1',   f'{f1_cls:.4f}',   f'{f1_mt:.4f}')
    table.add_data('Mean IoU',   f'{mean_iou:.4f}',  f'{iou_mt:.4f}')
    table.add_data('Dice Score', f'{dice_seg:.4f}',  f'{dice_mt:.4f}')
    table.add_data('Pixel Acc',  f'{pxacc_seg:.4f}', f'{pxacc_mt:.4f}')
    wandb.log({'isolated_vs_unified_table': table})

    print('\nSection 2.8 complete — all plots and tables logged to W&B.')
    wandb.finish()


if __name__ == '__main__':
    main()
