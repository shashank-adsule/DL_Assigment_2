"""
Weights & Biases logging utilities.

Provides:
  init_wandb          — initialise a W&B run with config dict
  log_metrics         — log a dict of scalar metrics at a given step
  log_images_bbox     — log a W&B table of images with bbox overlays (Task 2)
  log_seg_samples     — log segmentation triples: image / GT / pred (Task 3)
  log_feature_maps    — log feature map grids from a conv layer (Task 4 / report)
  log_activation_hist — log activation histograms (BN ablation study)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import wandb


def init_wandb(project: str, run_name: str, config: dict) -> wandb.run:
    return wandb.init(project=project, name=run_name, config=config)


def log_metrics(metrics: dict, step: int):
    wandb.log(metrics, step=step)


# ---------------------------------------------------------------------------
# Task 2: bounding-box overlay table
# ---------------------------------------------------------------------------
def log_images_bbox(
    images: torch.Tensor,       # (N, 3, H, W) — denormalised to [0,1]
    pred_boxes: torch.Tensor,   # (N, 4) [cx, cy, w, h] norm
    gt_boxes: torch.Tensor,     # (N, 4) [cx, cy, w, h] norm
    iou_scores: torch.Tensor,   # (N,)
    table_name: str = "bbox_predictions",
    n: int = 10,
):
    """
    Log a W&B table with image, predicted bbox (red), GT bbox (green), IoU.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    table = wandb.Table(columns=["image", "IoU", "confidence"])

    for i in range(min(n, len(images))):
        img = images[i].cpu() * std + mean          # denormalise
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        H, W = img.shape[:2]

        def box_to_abs(box):
            cx, cy, bw, bh = box.tolist()
            x1 = int((cx - bw/2) * W)
            y1 = int((cy - bh/2) * H)
            x2 = int((cx + bw/2) * W)
            y2 = int((cy + bh/2) * H)
            return x1, y1, x2, y2

        x1p, y1p, x2p, y2p = box_to_abs(pred_boxes[i])
        x1g, y1g, x2g, y2g = box_to_abs(gt_boxes[i])
        iou = iou_scores[i].item()

        wandb_img = wandb.Image(img, boxes={
            "predictions": {"box_data": [
                {"position": {"minX": x1p, "minY": y1p, "maxX": x2p, "maxY": y2p},
                 "class_id": 1, "box_caption": f"pred IoU={iou:.2f}",
                 "scores": {"iou": iou}}
            ], "class_labels": {1: "pred"}},
            "ground_truth": {"box_data": [
                {"position": {"minX": x1g, "minY": y1g, "maxX": x2g, "maxY": y2g},
                 "class_id": 1, "box_caption": "GT"}
            ], "class_labels": {1: "gt"}},
        })
        table.add_data(wandb_img, round(iou, 3), round(iou, 3))

    wandb.log({table_name: table})


# ---------------------------------------------------------------------------
# Task 3: segmentation triples
# ---------------------------------------------------------------------------
def log_seg_samples(
    images: torch.Tensor,   # (N, 3, H, W)
    gt_masks: torch.Tensor, # (N, H, W) long
    pred_logits: torch.Tensor,  # (N, C, H, W)
    n: int = 5,
    step: int = 0,
):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    pred_masks = pred_logits.argmax(dim=1)  # (N, H, W)
    palette = np.array([[255, 128, 0], [0, 128, 255], [128, 255, 0]])  # fg / bg / border

    log_imgs = []
    for i in range(min(n, len(images))):
        img = (images[i].cpu() * std + mean).clamp(0, 1).permute(1,2,0).numpy()

        def mask_to_rgb(m):
            m = m.cpu().numpy()
            rgb = palette[m.clip(0, 2)]
            return rgb.astype(np.uint8)

        gt_rgb   = mask_to_rgb(gt_masks[i])
        pred_rgb = mask_to_rgb(pred_masks[i])

        log_imgs.append(wandb.Image(img,      caption=f"[{i}] original"))
        log_imgs.append(wandb.Image(gt_rgb,   caption=f"[{i}] GT mask"))
        log_imgs.append(wandb.Image(pred_rgb, caption=f"[{i}] pred mask"))

    wandb.log({"seg_samples": log_imgs}, step=step)


# ---------------------------------------------------------------------------
# Feature map visualisation (W&B report section 2.4)
# ---------------------------------------------------------------------------
def log_feature_maps(
    model: nn.Module,
    image: torch.Tensor,   # (1, 3, H, W)
    layer_indices: list = (0, -3),   # indices into model.features
    step: int = 0,
):
    """
    Extract and log feature maps from specific conv layers.
    layer_indices: indices into model.features (negative = from end).
    """
    activations = {}

    def make_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach().cpu()
        return hook

    hooks = []
    for idx in layer_indices:
        layer = list(model.features.children())[idx]
        h = layer.register_forward_hook(make_hook(f"layer_{idx}"))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        model(image)

    for h in hooks:
        h.remove()

    for name, feat in activations.items():
        # feat shape: (1, C, H, W) — take first 16 channels as a grid
        maps = feat[0, :16].unsqueeze(1)           # (16, 1, H, W)
        maps = (maps - maps.min()) / (maps.max() - maps.min() + 1e-6)
        grid = make_grid(maps, nrow=4, padding=2)
        wandb.log({f"feature_maps/{name}": wandb.Image(grid.permute(1,2,0).numpy())}, step=step)


# ---------------------------------------------------------------------------
# Activation histograms for BN ablation (W&B report section 2.1)
# ---------------------------------------------------------------------------
def log_activation_hist(model: nn.Module, image: torch.Tensor,
                        target_layer_idx: int = 6, step: int = 0, tag: str = ""):
    """
    Log activation distribution at a specific layer (e.g. 3rd conv).
    """
    activations = {}

    def hook(module, inp, out):
        activations["act"] = out.detach().cpu()

    layer = list(model.features.children())[target_layer_idx]
    h = layer.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        model(image)
    h.remove()

    vals = activations["act"].flatten().numpy()
    wandb.log({f"activation_hist/{tag}": wandb.Histogram(vals)}, step=step)
