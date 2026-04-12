# Checkpoints

Model checkpoints are **not stored in this repository**.

They are automatically downloaded from Google Drive when
`MultiTaskPerceptionModel` is instantiated (see `models/multitask.py`).

Expected files after download:
- `classifier.pth`  — Task 1 best val macro-F1
- `localizer.pth`   — Task 2 best val mean-IoU
- `unet.pth`        — Task 3 best val macro-Dice
