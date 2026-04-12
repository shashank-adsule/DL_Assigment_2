"""
report_2_8_meta_analysis.py  —  Section 2.8: Meta-Analysis and Reflection

Loads all three individual checkpoints plus the unified MultiTaskPerceptionModel,
evaluates each on the test set, and logs:
  1. Final metrics summary bar chart
  2. Isolated task vs unified pipeline comparison W&B Table
  3. Top-10 confusion pairs for classification
  4. Per-class Dice scores for segmentation
  5. All scalar metrics

Requires:
  checkpoints/classifier.pth   (train.py --task classification)
  checkpoints/localizer.pth    (train.py --task localization)
  checkpoints/unet.pth         (train.py --task segmentation)
  (MultiTaskPerceptionModel auto-downloads from Google Drive)

Run from project root:
    python wandb_reports/report_2_8_meta_analysis.py \
        --data_root /path/to/pets \
        --cls_ckpt  checkpoints/classifier.pth \
        --loc_ckpt  checkpoints/localizer.pth \
        --seg_ckpt  checkpoints/unet.pth
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
import wandb
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader

from models.classification import PetClassifier
from models.localization   import LocalizationModel
from models.segmentation   import UNetVGG11
from models.multitask      import MultiTaskPerceptionModel
from data.dataset     import OxfordPetDataset, collate_fn

CLASS_NAMES = [
    "Abyssinian","Bengal","Birman","Bombay","British Shorthair",
    "Egyptian Mau","Maine Coon","Persian","Ragdoll","Russian Blue",
    "Siamese","Sphynx","American Bulldog","American Pit Bull Terrier",
    "Basset Hound","Beagle","Boxer","Chihuahua","English Cocker Spaniel",
    "English Setter","German Shorthaired","Great Pyrenees","Havanese",
    "Japanese Chin","Keeshond","Leonberger","Miniature Pinscher",
    "Newfoundland","Pomeranian","Pug","Saint Bernard","Samoyed",
    "Scottish Terrier","Shiba Inu","Staffordshire Bull Terrier",
    "Wheaten Terrier","Yorkshire Terrier",
]


def load_sd(path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return raw.get("state_dict", raw)


def xyxy_to_cxcywh(boxes):
    x1,y1,x2,y2 = boxes.unbind(1)
    return torch.stack([(x1+x2)*0.5,(y1+y2)*0.5,
                        (x2-x1).clamp(0),(y2-y1).clamp(0)], dim=1)


def iou_cxcywh(pred, gt, eps=1e-6):
    def c(b):
        cx,cy,w,h = b.unbind(1); return cx-w/2,cy-h/2,cx+w/2,cy+h/2
    px1,py1,px2,py2 = c(pred); gx1,gy1,gx2,gy2 = c(gt)
    iw = (torch.min(px2,gx2)-torch.max(px1,gx1)).clamp(0)
    ih = (torch.min(py2,gy2)-torch.max(py1,gy1)).clamp(0)
    inter = iw*ih
    return inter/((px2-px1).clamp(0)*(py2-py1).clamp(0)+
                  (gx2-gx1).clamp(0)*(gy2-gy1).clamp(0)-inter+eps)


def macro_dice(logits, masks, ignore=-1):
    preds = logits.argmax(1); valid = masks != ignore
    total, count = 0.0, 0
    for c in range(logits.size(1)):
        tp = ((preds==c)&(masks==c)&valid).sum().item()
        fp = ((preds==c)&(masks!=c)&valid).sum().item()
        fn = ((preds!=c)&(masks==c)&valid).sum().item()
        d  = 2*tp+fp+fn
        if d > 0: total += 2*tp/d; count += 1
    return total / max(count, 1)


def pixel_acc(logits, masks, ignore=-1):
    preds = logits.argmax(1); valid = masks != ignore
    return ((preds==masks)&valid).sum().item() / max(valid.sum().item(),1)


def per_class_dice(logits, masks):
    preds = logits.argmax(1)
    scores = []
    for c in range(3):
        tp = ((preds==c)&(masks==c)).sum().item()
        fp = ((preds==c)&(masks!=c)).sum().item()
        fn = ((preds!=c)&(masks==c)).sum().item()
        d  = 2*tp+fp+fn
        scores.append(2*tp/d if d>0 else 0.0)
    return scores


def make_loader(root, mode, batch_size, num_workers):
    return DataLoader(OxfordPetDataset(root, partition="test", mode=mode),
                      batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, collate_fn=collate_fn)


# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_classifier(model, loader, device):
    model.eval(); preds, labels = [], []
    for batch in loader:
        preds.extend(model(batch["image"].to(device)).argmax(1).cpu().tolist())
        labels.extend(batch["label"].tolist())
    return f1_score(labels, preds, average="macro", zero_division=0), preds, labels


@torch.no_grad()
def eval_localizer(model, loader, device):
    model.eval(); ious = []
    for batch in loader:
        pred  = model(batch["image"].to(device)).cpu()
        gt_cx = xyxy_to_cxcywh(batch["bbox"])
        valid = batch["bbox_mask"].bool()
        if valid.sum() > 0:
            ious.extend(iou_cxcywh(pred[valid], gt_cx[valid]).tolist())
    return float(np.mean(ious)) if ious else 0.0


@torch.no_grad()
def eval_segmenter(model, loader, device):
    model.eval(); ll, mm = [], []
    for batch in loader:
        ll.append(model(batch["image"].to(device)).cpu())
        mm.append(batch["mask"])
    lc, mc = torch.cat(ll), torch.cat(mm)
    return macro_dice(lc, mc), pixel_acc(lc, mc), per_class_dice(lc, mc)


@torch.no_grad()
def eval_multitask(model, loader, device):
    model.eval()
    cp, cl, ious, ll, mm = [], [], [], [], []
    for batch in loader:
        imgs  = batch["image"].to(device)
        out   = model(imgs)
        cp.extend(out["classification"].argmax(1).cpu().tolist())
        cl.extend(batch["label"].tolist())
        gt_cx = xyxy_to_cxcywh(batch["bbox"])
        valid = batch["bbox_mask"].bool()
        if valid.sum() > 0:
            ious.extend(iou_cxcywh(out["localization"].cpu()[valid],
                                   gt_cx[valid]).tolist())
        ll.append(out["segmentation"].cpu())
        mm.append(batch["mask"])
    lc, mc = torch.cat(ll), torch.cat(mm)
    f1  = f1_score(cl, cp, average="macro", zero_division=0)
    iou = float(np.mean(ious)) if ious else 0.0
    return f1, iou, macro_dice(lc, mc), pixel_acc(lc, mc), cp, cl


# ---------------------------------------------------------------------------
def plot_summary(metrics):
    fig, ax = plt.subplots(figsize=(12, 5))
    keys, vals = list(metrics.keys()), list(metrics.values())
    colors = ["#378ADD","#D85A30","#1D9E75","#BA7517",
              "#534AB7","#993C1D","#0F6E56","#5F5E5A"]
    bars = ax.bar(keys, vals, color=colors[:len(keys)], edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
                f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title("Final test-set metrics — all tasks")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    return fig


def plot_top10_confusion(y_true, y_pred):
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    off = cm.copy(); np.fill_diagonal(off, 0)
    top = np.argsort(off.ravel())[::-1][:10]
    rows, cols = top//len(CLASS_NAMES), top%len(CLASS_NAMES)
    labels = [f"{CLASS_NAMES[r]}\n→ {CLASS_NAMES[c]}"
              for r,c in zip(rows,cols)]
    counts = [cm[r,c] for r,c in zip(rows,cols)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels[::-1], counts[::-1], color="#378ADD")
    ax.set_xlabel("Misclassification count")
    ax.set_title("Top-10 classification errors"); plt.tight_layout()
    return fig


def plot_per_class_dice(scores):
    names  = ["Foreground","Background","Boundary"]
    colors = ["#FF8000","#4682B4","#FFD700"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, scores, color=colors, edgecolor="white")
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x()+bar.get_width()/2, s+0.01,
                f"{s:.3f}", ha="center", fontsize=11)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Dice score")
    ax.set_title("Per-class Dice — segmentation"); plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",     default=r"D:\code\repo\DL_Assigment_2\temp")
    ap.add_argument("--cls_ckpt",      default="checkpoints/classifier.pth")
    ap.add_argument("--loc_ckpt",      default="checkpoints/localizer.pth")
    ap.add_argument("--seg_ckpt",      default="checkpoints/unet.pth")
    ap.add_argument("--batch_size",    type=int, default=16)
    ap.add_argument("--num_workers",   type=int, default=4)
    ap.add_argument("--wandb_project", default="da6401-assignment2")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project=args.wandb_project, name="2.8_meta_analysis",
               config=vars(args))

    # --- Load models ---
    clf_model = PetClassifier(num_classes=37)
    clf_model.load_state_dict(load_sd(args.cls_ckpt))
    clf_model = clf_model.to(device)

    loc_model = LocalizationModel()
    loc_model.load_state_dict(load_sd(args.loc_ckpt))
    loc_model = loc_model.to(device)

    seg_model = UNetVGG11(num_classes=3)
    seg_model.load_state_dict(load_sd(args.seg_ckpt))
    seg_model = seg_model.to(device)

    # MultiTaskPerceptionModel auto-downloads checkpoints
    mt_model = MultiTaskPerceptionModel()
    mt_model  = mt_model.to(device)
    print("All models loaded.")

    # --- Loaders ---
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    cls_dl = make_loader(args.data_root, "cls", **kw)
    loc_dl = make_loader(args.data_root, "loc", **kw)
    seg_dl = make_loader(args.data_root, "seg", **kw)
    mt_dl  = make_loader(args.data_root, "all", **kw)

    # --- Evaluate ---
    print("\nEvaluating isolated models...")
    f1_cls, preds_cls, labels_cls = eval_classifier(clf_model, cls_dl, device)
    iou_loc                        = eval_localizer(loc_model,  loc_dl, device)
    dice_seg, pacc_seg, pc_dice    = eval_segmenter(seg_model,  seg_dl, device)

    print("Evaluating unified pipeline...")
    f1_mt, iou_mt, dice_mt, pacc_mt, preds_mt, labels_mt = \
        eval_multitask(mt_model, mt_dl, device)

    # --- Log metrics ---
    metrics = {
        "cls/f1_macro":  f1_cls,  "loc/mean_iou":  iou_loc,
        "seg/dice":      dice_seg, "seg/pixel_acc": pacc_seg,
        "mt/f1_macro":   f1_mt,   "mt/mean_iou":   iou_mt,
        "mt/dice":       dice_mt,  "mt/pixel_acc":  pacc_mt,
    }
    wandb.log(metrics)
    print("\nFinal metrics:")
    for k, v in metrics.items():
        print(f"  {k:<22}  {v:.4f}")

    # --- Figures ---
    fig_sum  = plot_summary({
        "F1 (cls)": f1_cls, "IoU (loc)": iou_loc,
        "Dice (seg)": dice_seg, "F1 (MT)": f1_mt,
        "IoU (MT)": iou_mt,  "Dice (MT)": dice_mt,
    })
    fig_cm   = plot_top10_confusion(labels_mt, preds_mt)
    fig_dice = plot_per_class_dice(pc_dice)

    wandb.log({
        "summary_chart":   wandb.Image(fig_sum),
        "top10_confusion": wandb.Image(fig_cm),
        "per_class_dice":  wandb.Image(fig_dice),
    })
    for f in [fig_sum, fig_cm, fig_dice]:
        plt.close(f)

    # --- Comparison table ---
    tbl = wandb.Table(columns=["Metric", "Isolated model", "Unified MT pipeline"])
    tbl.add_data("Macro F1",   f"{f1_cls:.4f}",   f"{f1_mt:.4f}")
    tbl.add_data("Mean IoU",   f"{iou_loc:.4f}",   f"{iou_mt:.4f}")
    tbl.add_data("Dice Score", f"{dice_seg:.4f}",  f"{dice_mt:.4f}")
    tbl.add_data("Pixel Acc",  f"{pacc_seg:.4f}",  f"{pacc_mt:.4f}")
    wandb.log({"isolated_vs_unified": tbl})

    print("\nSection 2.8 done.")
    wandb.finish()


if __name__ == "__main__":
    main()