# DA6401 – Assignment 2: Visual Perception Pipeline

## Links
- **W&B Report:** `https://wandb.ai/YOUR_USERNAME/da6401-assignment2/reports/YOUR_REPORT_LINK`
- **GitHub Repo:** `https://github.com/YOUR_USERNAME/da6401-assignment2`

> Replace both links above with your actual public W&B report URL and GitHub repo URL before submitting.

---

## Project Overview

A multi-stage visual perception pipeline trained on the Oxford-IIIT Pet dataset covering:

- **Task 1** — 37-class breed classification (VGG-11 + FC head)
- **Task 2** — Single-object bounding box localisation (VGG-11 encoder + regression head)
- **Task 3** — Trimap semantic segmentation (U-Net with VGG-11 backbone)
- **Task 4** — Unified multi-task model (single forward pass, three outputs)

---

## Project Structure

```
.
├── checkpoints/
│   └── checkpoints.md        # Checkpoints auto-downloaded from Google Drive
├── data/
│   └── pets_dataset.py       # Dataset loader (albumentations, list.txt split)
├── inference.py              # Evaluate MultiTaskPerceptionModel on test set
├── losses/
│   ├── __init__.py
│   └── iou_loss.py           # Custom IoU loss (inherits nn.Module)
├── models/
│   ├── __init__.py
│   ├── classification.py     # PetClassifier = VGG11Encoder + FCHead
│   ├── layers.py             # CustomDropout (inverted scaling)
│   ├── localization.py       # LocalizationModel = VGG11Encoder + BBoxHead
│   ├── multitask.py          # MultiTaskPerceptionModel (auto-downloads checkpoints)
│   ├── segmentation.py       # UNetVGG11 + DiceCELoss
│   └── vgg11.py              # VGG11Encoder backbone
├── README.md
├── requirements.txt
└── train.py                  # Single entry point for all 3 tasks
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training

Run tasks **in order** — each task loads the encoder from the previous checkpoint.

```bash
# Task 1 — classification (~2-3 hrs on GPU)
python train.py --task classification --data_dir /path/to/oxford_pets --epochs 60

# Task 2 — localisation (~1-1.5 hrs on GPU)
python train.py --task localization --data_dir /path/to/oxford_pets --epochs 40

# Task 3 — segmentation (~1.5-2 hrs on GPU)
python train.py --task segmentation --data_dir /path/to/oxford_pets --epochs 50

# Ablation variants for W&B sections 2.1 and 2.2
python train.py --task classification --data_dir /path/to/oxford_pets --ablation
```

Checkpoints are saved to `checkpoints/` automatically.

---

## Inference

```bash
python inference.py --data_root /path/to/oxford_pets
```

Evaluates the unified `MultiTaskPerceptionModel` and reports:
- Classification Macro F1
- Detection Mean IoU
- Segmentation Dice Score

---

## Architecture Notes

**Custom Dropout** — inverted scaling: keeps units survive with probability `(1-p)` and scales by `1/(1-p)` at train time. Eval mode is a no-op (identity).

**VGG11Encoder** — five conv blocks (`block1`–`block5`) with BatchNorm2d after every convolution. `return_features=True` returns skip maps `{b1..b5}` for U-Net.

**IoULoss** — accepts `reduction='mean'|'sum'|'none'`. Input format: `(cx, cy, w, h)` in pixel coordinates.

**MultiTaskPerceptionModel** — uses three independent encoders (one per task) to avoid the feature-mismatch collapse that occurs with a single shared encoder. Checkpoints are auto-downloaded from Google Drive on init.