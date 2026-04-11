"""
report_2_7_pipeline_showcase.py
=================================
W&B Report Section 2.7 — The Final Pipeline Showcase

Runs the full MultiTaskPerceptionModel on:
  (A) 3 novel "in-the-wild" images you download from the internet
  (B) 5 test-set images as a reference baseline

For each image logs to W&B:
  - Original image
  - Predicted bounding box overlay
  - Predicted segmentation mask overlay
  - Predicted breed label (top-3)

Usage:
  1. Download 3 pet images from the internet, save them anywhere.
  2. Run:

    python report_2_7_pipeline_showcase.py \
        --data_root    /path/to/oxford_pets \
        --model_ckpt   outputs/task4_multitask_best.pt \
        --wild_images  /path/img1.jpg /path/img2.jpg /path/img3.jpg \
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
from PIL import Image
import wandb

import torchvision.transforms as T
from models.multitask import MultiTaskPerceptionModel
from data.dataset     import get_dataloaders

# Oxford Pet class names (1-indexed in dataset; 0-indexed here)
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

PALETTE = np.array([[255,128,0],[70,130,180],[255,255,0]], dtype=np.uint8)

IMG_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])


# ---------------------------------------------------------------------------
def denorm(t):
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    return (t * std + mean).clamp(0,1).permute(1,2,0).numpy()


def run_pipeline(model, img_tensor, device):
    model.eval()
    with torch.no_grad():
        out = model(img_tensor.unsqueeze(0).to(device))
    cls_logits = out['classification'][0].cpu()
    bbox       = out['localization'][0].cpu()
    seg_logits = out['segmentation'][0].cpu()
    return cls_logits, bbox, seg_logits


def make_result_figure(orig_np, cls_logits, bbox, seg_logits, title=''):
    """3-panel figure: bbox overlay | seg mask | top-3 breed bar chart."""
    H, W = orig_np.shape[:2]

    # bbox panel
    pred_mask = seg_logits.argmax(0).numpy().clip(0, 2)
    seg_rgb   = PALETTE[pred_mask]

    top3_probs  = torch.softmax(cls_logits, 0).topk(3)
    top3_labels = [CLASS_NAMES[i] for i in top3_probs.indices.tolist()]
    top3_vals   = top3_probs.values.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: bbox
    ax = axes[0]
    ax.imshow(orig_np)
    cx, cy, bw, bh = bbox.tolist()
    x1 = (cx - bw/2) * W;  y1 = (cy - bh/2) * H
    ax.add_patch(patches.Rectangle(
        (x1, y1), bw*W, bh*H,
        linewidth=3, edgecolor='red', facecolor='none'))
    ax.set_title('Bounding box', fontsize=12); ax.axis('off')

    # Panel 2: seg mask
    ax = axes[1]
    overlay = (orig_np * 0.5 + seg_rgb/255 * 0.5)
    ax.imshow(overlay.clip(0, 1))
    ax.set_title('Segmentation mask', fontsize=12); ax.axis('off')

    # Panel 3: top-3 breeds
    ax = axes[2]
    colors = ['#4C72B0', '#DD8452', '#55A868']
    bars = ax.barh(top3_labels[::-1], top3_vals[::-1], color=colors[::-1])
    ax.set_xlim(0, 1); ax.set_xlabel('Confidence')
    ax.set_title('Top-3 breed predictions', fontsize=12)
    for bar, val in zip(bars, top3_vals[::-1]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.2%}', va='center', fontsize=9)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',     required=True)
    parser.add_argument('--model_ckpt',    required=True)
    parser.add_argument('--wild_images',   nargs='+', default=[],
                        help='Paths to novel in-the-wild pet images')
    parser.add_argument('--n_test',        type=int, default=5,
                        help='How many test-set images to also showcase')
    parser.add_argument('--batch_size',    type=int, default=8)
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--wandb_project', default='da6401_a2_report')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(project=args.wandb_project, name='2.7_pipeline_showcase',
               config=vars(args))

    # --- Load unified model ---
    model = MultiTaskPerceptionModel(num_classes=37, num_seg_classes=3)
    ckpt  = torch.load(args.model_ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    print('MultiTask model loaded.')

    wb_wild, wb_test = [], []

    # ---- A. In-the-wild images ----
    for path in args.wild_images:
        if not os.path.exists(path):
            print(f'  WARNING: {path} not found, skipping.')
            continue
        pil = Image.open(path).convert('RGB')
        orig_np = np.array(pil.resize((224,224))).astype(np.float32) / 255.0
        tensor  = IMG_TRANSFORM(pil)

        cls_l, bbox, seg_l = run_pipeline(model, tensor, device)
        fig = make_result_figure(orig_np, cls_l, bbox, seg_l,
                                 title=f'Wild image: {os.path.basename(path)}')
        wb_wild.append(wandb.Image(fig, caption=os.path.basename(path)))
        plt.close(fig)
        print(f'  Processed wild image: {path}')

    if wb_wild:
        wandb.log({'wild_images_showcase': wb_wild})

    # ---- B. Test-set images ----
    _, _, test_loader = get_dataloaders(
        root=args.data_root, task='multitask',
        batch_size=args.batch_size, num_workers=args.num_workers)

    collected = 0
    for batch in test_loader:
        imgs, labels, bboxes, masks = batch
        for i in range(len(imgs)):
            if collected >= args.n_test: break
            orig_np = denorm(imgs[i])
            cls_l, bbox_pred, seg_l = run_pipeline(model, imgs[i], device)
            fig = make_result_figure(
                orig_np, cls_l, bbox_pred, seg_l,
                title=f'Test image {collected+1} — GT: {CLASS_NAMES[labels[i]]}')
            wb_test.append(wandb.Image(fig, caption=f'Test {collected+1}'))
            plt.close(fig)
            collected += 1
        if collected >= args.n_test: break

    if wb_test:
        wandb.log({'test_set_showcase': wb_test})

    print(f'\nShowcased {len(wb_wild)} wild + {collected} test images.')
    print('Section 2.7 complete.')
    wandb.finish()


if __name__ == '__main__':
    main()
