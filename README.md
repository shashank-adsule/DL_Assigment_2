# DA6401 — Assignment 2: Visual Perception Pipeline

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
│   └── checkpoints.md          # Drive links; .pth files auto-downloaded on init
├── data/
│   └── pets_dataset.py         # Dataset loader — albumentations, list.txt split
├── inference.py                # Evaluate the unified pipeline on the test set
├── losses/
│   ├── __init__.py
│   └── iou_loss.py             # Custom IoU loss (nn.Module, reduction kwarg)
├── models/
│   ├── __init__.py
│   ├── classification.py       # FCHead + PetClassifier
│   ├── layers.py               # CustomDropout (inverted scaling)
│   ├── localization.py         # BBoxHead + LocalizationModel
│   ├── multitask.py            # MultiTaskPerceptionModel
│   ├── segmentation.py         # UpBlock + UNetVGG11 + DiceCELoss
│   └── vgg11.py                # VGG11Encoder backbone
├── README.md
├── requirements.txt
└── train.py                    # Single entry point for all three tasks
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

Download the Oxford-IIIT Pet Dataset and extract it so the directory looks like:

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

Runs three extra variants after the base run:

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
| `--label_smoothing` | `0.1` | CrossEntropy label smoothing (classification) |
| `--mixup_alpha` | `0.4` | Mixup alpha — set 0 to disable |
| `--freeze_encoder` | `False` | Freeze encoder weights (segmentation only) |
| `--num_workers` | `4` | DataLoader workers |
| `--conf_matrix_every` | `5` | Log confusion matrix every N epochs |
| `--wandb_project` | `da6401-assignment2` | W&B project name |
| `--ablation` | `False` | Run BN/dropout ablation variants |

---

## Inference

Evaluates `MultiTaskPerceptionModel` on the held-out test split. Checkpoints are automatically downloaded from Google Drive on first run.

```bash
python inference.py \
    --data_root /path/to/oxford_pets \
    --ckpt_dir  checkpoints \
    --batch_size 32
```

Sample output:

```
Test samples: 733
====================================================
  Classification  Macro F1   :  0.XXXX
  Detection       Mean IoU   :  0.XXXX
  Segmentation    Dice Score :  0.XXXX
====================================================
```

---

## Architecture

### VGG11Encoder (`models/vgg11.py`)

Standard VGG-11 topology with BatchNorm2d after every convolution. Kaiming He initialisation for all Conv2d layers.

```
Input (B, 3, 224, 224)
  block1 → 64  filters  → pool → (B,  64, 112, 112)  skip b1
  block2 → 128 filters  → pool → (B, 128,  56,  56)  skip b2
  block3 → 256 filters×2→ pool → (B, 256,  28,  28)  skip b3
  block4 → 512 filters×2→ pool → (B, 512,  14,  14)  skip b4
  block5 → 512 filters×2→ pool → (B, 512,   7,   7)  bottleneck / skip b5
```

`forward(x, return_features=True)` returns both the bottleneck tensor and the skip map dict `{"b1"…"b5"}` used by the U-Net decoder.

### CustomDropout (`models/layers.py`)

Inverted-scaling dropout implemented without using `nn.Dropout` or `F.dropout`:

- **Train:** sample a Bernoulli mask at keep probability `(1−p)`, multiply activations by the mask, then scale by `1/(1−p)` so the expected output magnitude is unchanged.
- **Eval:** identity — no masking, no scaling.

```python
from models.layers import CustomDropout
d = CustomDropout(p=0.5)
```

**Design rationale:** BatchNorm1d is placed *before* dropout in the FC head. BN normalises the full feature distribution; dropout then acts as an ensemble regulariser on those normalised features, which is more stable than reversing the order.

### PetClassifier (`models/classification.py`)

```
VGG11Encoder (bottleneck 512×7×7)
  → Flatten → Linear(25088→4096) → BN1d → ReLU → CustomDropout(0.5)
            → Linear(4096→4096)  → BN1d → ReLU → CustomDropout(0.5)
            → Linear(4096→37)    → logits
```

### LocalizationModel (`models/localization.py`)

```
VGG11Encoder (bottleneck 512×7×7)
  → Flatten → Linear(25088→1024) → BN1d → ReLU → CustomDropout(0.5)
            → Linear(1024→4) → Sigmoid × 224
```

Output: `(cx, cy, w, h)` in pixel coordinates, bounded to `(0, 224)` by the sigmoid.

**Justification for fine-tuning vs freezing:** The encoder is warm-started from the classification checkpoint and fine-tuned end-to-end. Freezing the backbone reduces localisation accuracy because the classification features are not optimised for spatial precision; allowing the gradients to flow through the encoder lets the features adapt to the regression objective.

### IoULoss (`losses/iou_loss.py`)

Custom IoU loss inheriting from `nn.Module`. Converts `(cx, cy, w, h)` to corner format internally, computes per-sample IoU, returns `1 − IoU`. Supports `reduction='mean'|'sum'|'none'`.

```python
from losses.iou_loss import IoULoss
criterion = IoULoss(reduction='mean')
```

**Justification:** IoU loss directly optimises the overlap metric used for evaluation. Combined with MSE loss (which provides stable gradients early in training when IoU gradients are near zero), the model converges reliably.

### UNetVGG11 (`models/segmentation.py`)

U-Net style decoder that mirrors the VGG-11 encoder. All upsampling uses `ConvTranspose2d` — no bilinear interpolation.

```
bottleneck (7×7)  → UpBlock(512,512,512) → 14×14   fused with skip b5
                  → UpBlock(512,512,256) → 28×28   fused with skip b4
                  → UpBlock(256,256,128) → 56×56   fused with skip b3
                  → UpBlock(128,128, 64) → 112×112 fused with skip b2
                  → UpBlock( 64, 64, 32) → 224×224 fused with skip b1
                  → Conv1×1(32→3)        → mask logits
```

**Loss:** weighted CrossEntropy (`ignore_index=−1`) + soft Dice with class weights `[1.0, 0.8, 3.0]`. The boundary class (≈10% of pixels) is up-weighted ×3 to prevent the model from ignoring it.

**Justification for Dice loss:** Pixel accuracy is misleading for the Oxford Pet trimap because the background class dominates (~50% of pixels). A trivial "predict everything as background" classifier achieves ~50% pixel accuracy but near-zero Dice. Dice directly measures overlap and is robust to class imbalance.

### MultiTaskPerceptionModel (`models/multitask.py`)

Three independent VGG-11 encoders — one per task. Each encoder is loaded from the checkpoint produced by that task's dedicated training run.

**Why not a single shared encoder?** Each head was optimised against the features from its own encoder. If a shared encoder from one task is substituted, the other two heads receive feature distributions they were never trained on, causing their metrics to collapse. Three separate encoders ensure every head always sees the feature statistics from its training.

```python
from models.multitask import MultiTaskPerceptionModel

model = MultiTaskPerceptionModel()   # auto-downloads checkpoints from Google Drive
out   = model(images)                # single forward pass

out["classification"]  # (B, 37)          class logits
out["localization"]    # (B, 4)            (cx, cy, w, h) pixel space
out["segmentation"]    # (B, 3, 224, 224)  trimap logits
```

---

## Dataset

`data/pets_dataset.py` reads `annotations/list.txt` and applies a stratified 80/10/10 train/val/test split (`random_state=42`).

| Detail | Value |
|--------|-------|
| Total samples | 7,349 |
| Classes | 37 breeds |
| Input size | 224 × 224 |
| Trimap classes | 0=foreground, 1=background, 2=boundary |
| Bbox format (stored) | xyxy pixel |
| Bbox format (in loss) | cxcywh pixel |

**Training augmentation:** `RandomResizedCrop(0.5–1.0)` + `HorizontalFlip` + `ColorJitter` + `CLAHE` + `Sharpen` + `GaussNoise` + `MotionBlur` + `CoarseDropout`

**Val/test:** `Resize(224)` only — no random ops.

---

## Training Details

| Component | Choice | Reason |
|-----------|--------|--------|
| Optimiser | AdamW | Weight decay decoupled from gradient scaling |
| LR schedule | 5-epoch linear warm-up → cosine annealing to 1e-6 | Prevents cold-start instability with BatchNorm |
| Gradient clipping | max norm = 1.0 | Prevents exploding gradients in FC layers |
| EMA | decay = 0.999 | Validation and checkpointing use EMA weights for better generalisation |
| Mixup | α = 0.4, starts at epoch 6 | Activates after warm-up so BatchNorm statistics stabilise first |
| Label smoothing | 0.1 | Prevents overconfident predictions on visually similar breeds |

---

Pipeline metrics on the autograder test set:

| Metric | Value |
|--------|-------|
| Classification Macro-F1 | 0.3571 |
| Localisation Acc@IoU≥0.5 | 70.0% |
| Localisation Acc@IoU≥0.75 | 40.0% |
| Segmentation Macro-Dice | 0.7283 |

---

## Checkpoints

Model checkpoints are not stored in this repository (excluded via `.gitignore`). They are automatically downloaded from Google Drive when `MultiTaskPerceptionModel` is instantiated.

| File | Task | Drive link |
|------|------|------------|
| `classifier.pth` | Classification | https://drive.google.com/file/d/10PkzOaIfSnLNwx-j5i4RYXU1vH9bfljd/view?usp=sharing |
| `localizer.pth`  | Localisation  | https://drive.google.com/file/d/1baPZLEV-3ZK9Hcm1ovpSIsKOVX7C2Nkf/view?usp=sharing |
| `unet.pth`       | Segmentation  | https://drive.google.com/file/d/1yPqyIgdMd9ntg3EZ1k34kY0y08ocyxcq/view?usp=sharing |
