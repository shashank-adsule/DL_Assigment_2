# DA6401 — Assignment 2: Visual Perception Pipeline

> **Course:** DA6401 Introduction to Deep Learning — IIT Madras
> **Dataset:** [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

---

## Links

| Resource | URL |
|----------|-----|
| W&B Report | https://wandb.ai/da25m005-indian-institute-of-technology-madras/da6401-assignment2/reports/DL-assigment-2-W-B-report--VmlldzoxNjQ5OTMyMA |
| GitHub Repo | https://github.com/shashank-adsule/DL_Assigment_2 |

---

## Overview

A multi-stage visual perception pipeline that performs three tasks simultaneously on pet images:

- **Task 1 — Classification:** Identify the breed from 37 classes using a VGG-11 backbone with BatchNorm and custom dropout.
- **Task 2 — Localisation:** Predict a bounding box around the pet's head using a regression head on top of the VGG-11 encoder.
- **Task 3 — Segmentation:** Produce a pixel-wise trimap mask (foreground / background / boundary) using a U-Net style decoder.
- **Task 4 — Unified pipeline:** A single `MultiTaskPerceptionModel` that runs all three tasks in one forward pass.

---

## Project Structure

```
.
├── checkpoints/
│   ├── checkpoints.md              # Drive links; .pth files auto-downloaded on init
│   ├── classifier.pth              # Task 1 checkpoint (auto-downloaded, not in repo)
│   ├── localizer.pth               # Task 2 checkpoint (auto-downloaded, not in repo)
│   └── unet.pth                    # Task 3 checkpoint (auto-downloaded, not in repo)
├── data/
│   ├── __init__.py
│   └── dataset.py                  # Dataset loader — albumentations, list.txt split
├── losses/
│   ├── __init__.py
│   └── iou_loss.py                 # Custom IoU loss (nn.Module, reduction kwarg)
├── models/
│   ├── __init__.py
│   ├── classification.py           # FCHead + PetClassifier
│   ├── layers.py                   # CustomDropout (inverted scaling)
│   ├── localization.py             # BBoxHead + LocalizationModel
│   ├── multitask.py                # MultiTaskPerceptionModel
│   ├── segmentation.py             # UpBlock + UNetVGG11 + DiceCELoss
│   └── vgg11.py                    # VGG11Encoder backbone
├── outputs/                        # Training outputs and logs
├── train_tasks/                    # Individual task training scripts
│   ├── train_task1.py
│   ├── train_task2.py
│   ├── train_task3.py
│   └── train_task4.py
├── utils/                          # Shared utilities
│   ├── __init__.py
│   ├── metrics.py                  # Evaluation metrics (F1, IoU, Dice)
│   ├── trainer.py                  # Generic training loop helper
│   └── wandb_logger.py             # W&B logging helpers
├── wandb_test/                     # W&B report scripts (sections 2.1 – 2.8)
│   ├── report_2_1_batchnorm_effect.py
│   ├── report_2_2_dropout_dynamics.py
│   ├── report_2_3_transfer_learning.py
│   ├── report_2_4_feature_maps.py
│   ├── report_2_5_detection_table.py
│   ├── report_2_6_segmentation_eval.py
│   ├── report_2_7_pipeline_showcase.py
│   └── report_2_8_meta_analysis.py
├── .gitignore
├── inference.py                    # Evaluate unified pipeline on test set
├── README.md
├── requirements.txt
├── temp.ipynb                      # Exploration notebook
└── train.py                        # Single entry point for all three tasks
```

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

| Package | Version |
|---------|---------|
| torch | ≥ 2.0.0 |
| torchvision | ≥ 0.15.0 |
| albumentations | ≥ 1.3.0 |
| scikit-learn | ≥ 1.2.0 |
| wandb | ≥ 0.15.0 |
| gdown | ≥ 4.7.0 |
| numpy | ≥ 1.24.0 |
| Pillow | ≥ 9.0.0 |
| matplotlib | ≥ 3.7.0 |

### Dataset

Download the Oxford-IIIT Pet Dataset and extract so the directory looks like:

```
oxford_pets/
├── images/
│   ├── Abyssinian_1.jpg
│   └── ...
└── annotations/
    ├── list.txt
    ├── trimaps/
    └── xmls/
```

---

## Training

Run the three tasks **in order** — each task warm-starts the encoder from the previous checkpoint.

### Task 1 — Classification

```bash
python train.py \
    --task classification \
    --data_dir /path/to/oxford_pets \
    --epochs 100 \
    --lr 3e-4 \
    --batch_size 32
```

Saves `checkpoints/classifier.pth` at best val macro-F1.

### Task 2 — Localisation

Requires `checkpoints/classifier.pth` from Task 1.

```bash
python train.py \
    --task localization \
    --data_dir /path/to/oxford_pets \
    --epochs 40 \
    --lr 5e-4 \
    --batch_size 32
```

Saves `checkpoints/localizer.pth` at best val mean-IoU.

### Task 3 — Segmentation

Requires `checkpoints/classifier.pth` from Task 1.

```bash
python train.py \
    --task segmentation \
    --data_dir /path/to/oxford_pets \
    --epochs 80 \
    --lr 2e-4 \
    --batch_size 16
```

Saves `checkpoints/unet.pth` at best val macro-Dice.

### Ablation variants (W&B sections 2.1 and 2.2)

```bash
python train.py \
    --task classification \
    --data_dir /path/to/oxford_pets \
    --ablation
```

| Run suffix | Change | W&B section |
|------------|--------|-------------|
| `_nobn` | No BatchNorm | 2.1 |
| `_dp02` | Dropout p=0.2 | 2.2 |
| `_nodp` | No Dropout | 2.2 |

### All training arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `classification` | `classification` / `localization` / `segmentation` |
| `--data_dir` | `data/oxford_pet` | Path to Oxford-IIIT Pet root |
| `--ckpt_dir` | `checkpoints` | Where to save `.pth` files |
| `--epochs` | `60` | Number of training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `5e-4` | Initial learning rate |
| `--dropout_p` | `0.5` | Dropout probability |
| `--weight_decay` | `1e-4` | AdamW weight decay |
| `--label_smoothing` | `0.1` | CrossEntropy label smoothing |
| `--mixup_alpha` | `0.4` | Mixup alpha — set 0 to disable |
| `--freeze_encoder` | `False` | Freeze encoder weights (segmentation only) |
| `--num_workers` | `4` | DataLoader workers |
| `--conf_matrix_every` | `5` | Log confusion matrix every N epochs |
| `--wandb_project` | `da6401-assignment2` | W&B project name |
| `--ablation` | `False` | Run BN/dropout ablation variants |

---

## Inference

Checkpoints are auto-downloaded from Google Drive on first run.

```bash
python inference.py \
    --data_root /path/to/oxford_pets \
    --ckpt_dir  checkpoints \
    --batch_size 32
```

```
Test samples: 733
====================================================
  Classification  Macro F1   :  0.XXXX
  Detection       Mean IoU   :  0.XXXX
  Segmentation    Dice Score :  0.XXXX
====================================================
```

---

## W&B Report Scripts

All scripts are in `wandb_test/`. Run from the project root.

| Script | Section | Requires |
|--------|---------|----------|
| `report_2_1_batchnorm_effect.py` | BatchNorm effect | none |
| `report_2_2_dropout_dynamics.py` | Dropout dynamics | none |
| `report_2_3_transfer_learning.py` | Transfer learning showdown | `classifier.pth` |
| `report_2_4_feature_maps.py` | Feature map visualisation | `classifier.pth` |
| `report_2_5_detection_table.py` | Detection table & IoU | `localizer.pth` |
| `report_2_6_segmentation_eval.py` | Dice vs pixel accuracy | `unet.pth` |
| `report_2_7_pipeline_showcase.py` | Pipeline showcase | auto-downloaded |
| `report_2_8_meta_analysis.py` | Meta-analysis | all checkpoints |

Example:

```bash
# No checkpoint needed — trains from scratch
python wandb_test/report_2_1_batchnorm_effect.py --data_root /path/to/oxford_pets

# Needs classifier.pth
python wandb_test/report_2_4_feature_maps.py --data_root /path/to/oxford_pets

# Supply 3 in-the-wild images for section 2.7
python wandb_test/report_2_7_pipeline_showcase.py \
    --data_root   /path/to/oxford_pets \
    --wild_images /path/dog1.jpg /path/cat1.jpg /path/dog2.jpg
```

---

## Architecture

### VGG11Encoder (`models/vgg11.py`)

Standard VGG-11 with BatchNorm2d after every convolution, Kaiming He initialisation.

```
Input (B, 3, 224, 224)
  block1 → 64  filters  → pool → (B,  64, 112, 112)  skip b1
  block2 → 128 filters  → pool → (B, 128,  56,  56)  skip b2
  block3 → 256 filters×2→ pool → (B, 256,  28,  28)  skip b3
  block4 → 512 filters×2→ pool → (B, 512,  14,  14)  skip b4
  block5 → 512 filters×2→ pool → (B, 512,   7,   7)  bottleneck
```

### CustomDropout (`models/layers.py`)

Inverted-scaling dropout — no `nn.Dropout` or `F.dropout` used.

- **Train:** Bernoulli mask at keep prob `(1−p)`, scale survivors by `1/(1−p)`
- **Eval:** identity pass-through

### PetClassifier (`models/classification.py`)

```
VGG11Encoder → Flatten
  → Linear(25088→4096) → BN1d → ReLU → CustomDropout(0.5)
  → Linear(4096→4096)  → BN1d → ReLU → CustomDropout(0.5)
  → Linear(4096→37)
```

### LocalizationModel (`models/localization.py`)

```
VGG11Encoder → Flatten
  → Linear(25088→1024) → BN1d → ReLU → CustomDropout(0.5)
  → Linear(1024→4) → Sigmoid × 224
```

Output: `(cx, cy, w, h)` pixel coordinates bounded to `(0, 224)`.

### IoULoss (`losses/iou_loss.py`)

Custom IoU loss. Input `(cx, cy, w, h)` pixel format → returns `1 − IoU`.
Supports `reduction='mean'|'sum'|'none'`.

### UNetVGG11 (`models/segmentation.py`)

U-Net decoder mirroring VGG-11 encoder. All upsampling via `ConvTranspose2d` only.

```
bottleneck (7×7) → UpBlock×5 → 224×224 → Conv1×1 → 3-class logits
```

Loss: weighted CrossEntropy + soft Dice. Class weights `[1.0, 0.8, 3.0]`.

### MultiTaskPerceptionModel (`models/multitask.py`)

Three independent VGG-11 encoders (one per task). Checkpoints auto-downloaded from Google Drive on init.

```python
from models.multitask import MultiTaskPerceptionModel

model = MultiTaskPerceptionModel()
out   = model(images)

out["classification"]  # (B, 37)
out["localization"]    # (B, 4)            — (cx, cy, w, h) pixels
out["segmentation"]    # (B, 3, 224, 224)
```

---

## Dataset Details

`data/dataset.py` reads `annotations/list.txt`. Stratified 80/10/10 split (`random_state=42`).

| Detail | Value |
|--------|-------|
| Total samples | 7,349 |
| Classes | 37 breeds |
| Input size | 224 × 224 |
| Trimap classes | 0=foreground, 1=background, 2=boundary |
| Bbox format stored | xyxy pixel |
| Bbox format in loss | cxcywh pixel |

**Train augmentation:** `RandomResizedCrop` + `HorizontalFlip` + `ColorJitter` + `CLAHE` + `GaussNoise` + `CoarseDropout`

**Val/test:** `Resize(224)` only.

---

## Training Details

| Component | Choice | Reason |
|-----------|--------|--------|
| Optimiser | AdamW | Decoupled weight decay |
| LR schedule | 5-epoch warm-up → cosine annealing | Prevents cold-start instability |
| Gradient clipping | max norm = 1.0 | Prevents exploding gradients |
| EMA | decay = 0.999 | Better generalisation at checkpoint time |
| Mixup | α = 0.4, starts epoch 6 | After warm-up so BN stats stabilise first |
| Label smoothing | 0.1 | Prevents overconfident predictions |

---

## Checkpoints

Not stored in this repo (excluded via `.gitignore`). Auto-downloaded by `MultiTaskPerceptionModel.__init__`.

| File | Task | Drive link |
|------|------|------------|
| `classifier.pth` | Classification | https://drive.google.com/file/d/10PkzOaIfSnLNwx-j5i4RYXU1vH9bfljd/view?usp=sharing |
| `localizer.pth`  | Localisation  | https://drive.google.com/file/d/1baPZLEV-3ZK9Hcm1ovpSIsKOVX7C2Nkf/view?usp=sharing |
| `unet.pth`       | Segmentation  | https://drive.google.com/file/d/1yPqyIgdMd9ntg3EZ1k34kY0y08ocyxcq/view?usp=sharing |

> ⚠️ Do not delete the Drive folder until assignment marks are released.