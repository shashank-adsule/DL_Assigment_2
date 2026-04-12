# Checkpoints

Model checkpoints are **not stored in this repository** (excluded via `.gitignore`).

They are automatically downloaded from Google Drive when
`MultiTaskPerceptionModel` is instantiated (see `models/multitask.py`).

## Expected files after download

| File | Task | Saved metric |
|------|------|--------------|
| `classifier.pth` | Task 1 — VGG-11 Classification | Best val macro-F1 |
| `localizer.pth`  | Task 2 — Bounding Box Localisation | Best val mean-IoU |
| `unet.pth`       | Task 3 — U-Net Segmentation | Best val macro-Dice |

## Google Drive links

| File | Link |
|------|------|
| `classifier.pth` | https://drive.google.com/file/d/10PkzOaIfSnLNwx-j5i4RYXU1vH9bfljd/view?usp=sharing |
| `localizer.pth`  | https://drive.google.com/file/d/1baPZLEV-3ZK9Hcm1ovpSIsKOVX7C2Nkf/view?usp=sharing |
| `unet.pth`       | https://drive.google.com/file/d/1yPqyIgdMd9ntg3EZ1k34kY0y08ocyxcq/view?usp=sharing |
