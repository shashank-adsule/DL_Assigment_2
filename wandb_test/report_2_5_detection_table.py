"""
report_2_5_detection_table.py  —  Section 2.5: Object Detection Confidence & IoU

Loads LocalizationModel from localizer.pth, runs on N test images, and logs
a W&B Table with GREEN (GT) and RED (pred) bounding boxes, plus IoU score.

Requires:
  checkpoints/classifier.pth   (for encoder warm-start — used inside LocalizationModel)
  checkpoints/localizer.pth    (produced by train.py --task localization)

Run from project root:
    python wandb_reports/report_2_5_detection_table.py \
        --data_root /path/to/pets \
        --loc_ckpt  checkpoints/localizer.pth \
        --n_images  20
"""

import os, sys, warnings
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from models.localization import LocalizationModel
from data.dataset import OxfordPetDataset, collate_fn

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
IMG_SIZE = 224.0


def denorm(t):
    return (t * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


def xyxy_to_cxcywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(1)
    return torch.stack([(x1+x2)*0.5, (y1+y2)*0.5,
                        (x2-x1).clamp(0), (y2-y1).clamp(0)], dim=1)


def iou_cxcywh(pred, gt, eps=1e-6):
    def corners(b):
        cx, cy, w, h = b.unbind(1)
        return cx-w/2, cy-h/2, cx+w/2, cy+h/2
    px1,py1,px2,py2 = corners(pred)
    gx1,gy1,gx2,gy2 = corners(gt)
    iw = (torch.min(px2,gx2)-torch.max(px1,gx1)).clamp(0)
    ih = (torch.min(py2,gy2)-torch.max(py1,gy1)).clamp(0)
    inter = iw*ih
    ap = (px2-px1).clamp(0)*(py2-py1).clamp(0)
    ag = (gx2-gx1).clamp(0)*(gy2-gy1).clamp(0)
    return inter/(ap+ag-inter+eps)


def draw_bbox_figure(img_np, pred_cxcywh, gt_xyxy, iou_val):
    H, W = img_np.shape[:2]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_np)

    # Ground truth box — GREEN (xyxy pixel → matplotlib)
    x1, y1, x2, y2 = gt_xyxy
    ax.add_patch(patches.Rectangle(
        (x1, y1), x2-x1, y2-y1,
        linewidth=2.5, edgecolor="lime", facecolor="none", label="GT"))

    # Prediction box — RED (cxcywh pixel → matplotlib)
    cx, cy, bw, bh = pred_cxcywh
    ax.add_patch(patches.Rectangle(
        (cx-bw/2, cy-bh/2), bw, bh,
        linewidth=2.5, edgecolor="red", facecolor="none", label="Pred"))

    ax.set_title(f"IoU = {iou_val:.3f}", fontsize=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
    ax.axis("off"); plt.tight_layout(pad=0.2)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",     default=r"D:\code\repo\DL_Assigment_2\temp")
    ap.add_argument("--loc_ckpt",      default="checkpoints/localizer.pth")
    ap.add_argument("--n_images",      type=int, default=20)
    ap.add_argument("--batch_size",    type=int, default=16)
    ap.add_argument("--num_workers",   type=int, default=4)
    ap.add_argument("--wandb_project", default="da6401-assignment2")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project=args.wandb_project, name="2.5_detection_table",
               config=vars(args))

    # Load LocalizationModel
    model = LocalizationModel()
    raw   = torch.load(args.loc_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(raw.get("state_dict", raw))
    model = model.to(device)
    model.eval()
    print(f"Loaded {args.loc_ckpt}")

    te_dl = DataLoader(
        OxfordPetDataset(args.data_root, partition="test", mode="loc"),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn)

    table     = wandb.Table(columns=["image", "pred_iou", "confidence", "result"])
    failures  = []
    all_ious  = []
    collected = 0

    for batch in te_dl:
        if collected >= args.n_images:
            break
        imgs      = batch["image"]
        bbox_xyxy = batch["bbox"]              # (B,4) xyxy pixel
        valid     = batch["bbox_mask"].bool()

        with torch.no_grad():
            preds = model(imgs.to(device)).cpu()  # (B,4) cxcywh pixel

        gt_cxcywh = xyxy_to_cxcywh(bbox_xyxy)
        ious      = iou_cxcywh(preds, gt_cxcywh)

        for i in range(len(imgs)):
            if collected >= args.n_images:
                break
            if not valid[i]:
                continue

            iou_val  = ious[i].item()
            all_ious.append(iou_val)
            img_np   = denorm(imgs[i])
            result   = "good" if iou_val >= 0.5 else \
                       ("partial" if iou_val >= 0.25 else "failure")

            fig = draw_bbox_figure(
                img_np,
                preds[i].tolist(),
                bbox_xyxy[i].tolist(),
                iou_val)
            table.add_data(wandb.Image(fig), round(iou_val, 4),
                           round(iou_val, 4), result)
            plt.close(fig)

            if result in ("failure", "partial"):
                fig2 = draw_bbox_figure(img_np, preds[i].tolist(),
                                         bbox_xyxy[i].tolist(), iou_val)
                failures.append(wandb.Image(
                    fig2, caption=f"{result} — IoU={iou_val:.3f}"))
                plt.close(fig2)

            collected += 1

    wandb.log({"detection_table": table})
    if failures:
        wandb.log({"failure_cases": failures})

    # IoU histogram
    fig_h, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_ious, bins=20, color="#378ADD", edgecolor="white", alpha=0.85)
    ax.axvline(0.5, color="green",  ls="--", label="IoU threshold 0.5")
    ax.axvline(np.mean(all_ious), color="red", ls="-",
               label=f"mean IoU = {np.mean(all_ious):.3f}")
    ax.set_xlabel("IoU"); ax.set_ylabel("Count")
    ax.set_title("IoU distribution over test images"); ax.legend()
    plt.tight_layout()
    wandb.log({"iou_distribution": wandb.Image(fig_h)})
    plt.close(fig_h)

    print(f"Logged {collected} images  |  Mean IoU = {np.mean(all_ious):.4f}")
    print("Section 2.5 done.")
    wandb.finish()


if __name__ == "__main__":
    main()