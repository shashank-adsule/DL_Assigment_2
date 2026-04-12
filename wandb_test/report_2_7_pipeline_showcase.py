"""
report_2_7_pipeline_showcase.py  —  Section 2.7: Final Pipeline Showcase

Runs MultiTaskPerceptionModel on:
  (A) 3+ novel "in-the-wild" pet images you supply with --wild_images
  (B) N images from the test set for reference

For each image logs a 3-panel figure:
  [Bounding box overlay] | [Segmentation mask overlay] | [Top-3 breed chart]

Checkpoints are auto-downloaded from Google Drive by MultiTaskPerceptionModel.__init__.

Run from project root:
    python wandb_reports/report_2_7_pipeline_showcase.py \
        --data_root   /path/to/pets \
        --wild_images /path/dog1.jpg /path/cat1.jpg /path/dog2.jpg
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
import torchvision.transforms as T
import wandb
from PIL import Image
from torch.utils.data import DataLoader

from models.multitask import MultiTaskPerceptionModel
from data.pets_dataset import OxfordPetDataset, collate_fn

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
PALETTE = np.array([[255,128,0],[70,130,180],[255,255,0]], dtype=np.uint8)
MEAN    = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
STD     = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
TRANSFORM = T.Compose([
    T.Resize((224,224)), T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])


def denorm(t):
    return (t * STD + MEAN).clamp(0,1).permute(1,2,0).numpy()


@torch.no_grad()
def predict(model, tensor, device):
    out = model(tensor.unsqueeze(0).to(device))
    return (out["classification"][0].cpu(),
            out["localization"][0].cpu(),
            out["segmentation"][0].cpu())


def make_panel(orig_np, cls_logits, bbox_cxcywh, seg_logits, title=""):
    H, W = orig_np.shape[:2]
    seg_rgb  = PALETTE[seg_logits.argmax(0).numpy().clip(0,2)]
    top3     = torch.softmax(cls_logits, 0).topk(3)
    names    = [CLASS_NAMES[i] for i in top3.indices.tolist()]
    probs    = top3.values.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1 — bounding box (cxcywh pixel)
    axes[0].imshow(orig_np)
    cx, cy, bw, bh = bbox_cxcywh.tolist()
    axes[0].add_patch(patches.Rectangle(
        (cx-bw/2, cy-bh/2), bw, bh,
        linewidth=3, edgecolor="red", facecolor="none"))
    axes[0].set_title("Bounding box"); axes[0].axis("off")

    # Panel 2 — segmentation overlay
    overlay = (orig_np * 0.5 + seg_rgb/255.0 * 0.5).clip(0,1)
    axes[1].imshow(overlay)
    axes[1].set_title("Segmentation mask"); axes[1].axis("off")

    # Panel 3 — top-3 breed chart
    colors = ["#378ADD","#D85A30","#1D9E75"]
    bars   = axes[2].barh(names[::-1], probs[::-1], color=colors[::-1])
    axes[2].set_xlim(0,1); axes[2].set_xlabel("Confidence")
    axes[2].set_title("Top-3 breed predictions")
    for bar, p in zip(bars, probs[::-1]):
        axes[2].text(p+0.01, bar.get_y()+bar.get_height()/2,
                     f"{p:.1%}", va="center", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",     default=r"D:\code\repo\DL_Assigment_2\temp")
    ap.add_argument("--wild_images",   nargs="*", default=[],
                    help="Paths to in-the-wild pet images (3 recommended)")
    ap.add_argument("--n_test",        type=int, default=5)
    ap.add_argument("--batch_size",    type=int, default=8)
    ap.add_argument("--num_workers",   type=int, default=4)
    ap.add_argument("--wandb_project", default="da6401-assignment2")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project=args.wandb_project, name="2.7_pipeline_showcase",
               config=vars(args))

    # MultiTaskPerceptionModel auto-downloads checkpoints from Google Drive
    model = MultiTaskPerceptionModel()
    model = model.to(device)
    model.eval()
    print("MultiTaskPerceptionModel ready.")

    # --- A. In-the-wild images ---
    wild_logs = []
    for path in args.wild_images:
        if not os.path.exists(path):
            print(f"  Skipping (not found): {path}"); continue
        pil    = Image.open(path).convert("RGB")
        orig   = np.array(pil.resize((224,224))).astype(np.float32)/255.0
        tensor = TRANSFORM(pil)
        cls_l, bbox, seg_l = predict(model, tensor, device)
        fig = make_panel(orig, cls_l, bbox, seg_l,
                         title=f"Wild — {os.path.basename(path)}")
        wild_logs.append(wandb.Image(fig, caption=os.path.basename(path)))
        plt.close(fig)
        print(f"  Processed: {path}")
    if wild_logs:
        wandb.log({"wild_images": wild_logs})

    # --- B. Test-set reference images ---
    te_dl = DataLoader(
        OxfordPetDataset(args.data_root, partition="test", mode="all"),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn)

    test_logs = []
    collected = 0
    for batch in te_dl:
        imgs   = batch["image"]
        labels = batch["label"]
        for i in range(len(imgs)):
            if collected >= args.n_test: break
            orig = denorm(imgs[i])
            cls_l, bbox, seg_l = predict(model, imgs[i], device)
            fig = make_panel(orig, cls_l, bbox, seg_l,
                             title=f"Test {collected+1} — GT: {CLASS_NAMES[labels[i]]}")
            test_logs.append(wandb.Image(fig, caption=f"Test {collected+1}"))
            plt.close(fig)
            collected += 1
        if collected >= args.n_test: break
    if test_logs:
        wandb.log({"test_set_showcase": test_logs})

    print(f"Logged {len(wild_logs)} wild + {collected} test images.")
    print("Section 2.7 done.")
    wandb.finish()


if __name__ == "__main__":
    main()