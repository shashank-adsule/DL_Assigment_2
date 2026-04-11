import os
import warnings
import xml.etree.ElementTree as ET
# Suppress albumentations offline version-check warning
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_IMG_SIZE = 224
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def get_train_transforms() -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(_IMG_SIZE, _IMG_SIZE),
                scale=(0.5, 1.0),
                ratio=(0.75, 1.33),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.4, hue=0.1, p=0.8),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.Sharpen(alpha=(0.1, 0.4), lightness=(0.8, 1.2), p=0.3),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
            A.ToGray(p=0.05),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
            A.MotionBlur(blur_limit=5, p=0.1),
            A.GaussianBlur(blur_limit=3, p=0.15),
            A.CoarseDropout(
                num_holes_range=(6, 12),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                fill=0, p=0.4,
            ),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["bbox_labels"],
            min_visibility=0.2,
        ),
    )


def get_val_transforms() -> A.Compose:
    return A.Compose(
        [
            A.Resize(_IMG_SIZE, _IMG_SIZE),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["bbox_labels"],
            min_visibility=0.1,
        ),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OxfordPetDataset(Dataset):
    def __init__(
        self,
        root: str,
        partition: str = "train",
        mode: str = "cls",
        pipeline: Optional[A.Compose] = None,
        test_frac: float = 0.10,
        val_frac:  float = 0.10,
    ):
        self.root       = Path(root)
        self.partition  = partition
        self.mode       = mode
        self.img_dir    = self.root / "images"
        self.ann_dir    = self.root / "annotations"
        self.trimap_dir = self.ann_dir / "trimaps"
        self.xml_dir    = self.ann_dir / "xmls"

        self.pipeline = (pipeline if pipeline is not None
                         else (get_train_transforms() if partition == "train"
                               else get_val_transforms()))

        self.records = self._prepare_records(test_frac, val_frac)

    # ------------------------------------------------------------------
    def _read_list(self) -> List[dict]:
        entries = []
        with open(self.ann_dir / "list.txt") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                entries.append({"name": parts[0], "label": int(parts[1]) - 1})
        return entries

    def _split(self, entries, test_frac, val_frac):
        names  = [e["name"]  for e in entries]
        labels = [e["label"] for e in entries]
        tr_n, te_n, tr_l, _ = train_test_split(
            names, labels, test_size=test_frac,
            stratify=labels, random_state=42,
        )
        tr_n, va_n, _, _ = train_test_split(
            tr_n, tr_l, test_size=val_frac,
            stratify=tr_l, random_state=42,
        )
        return set(tr_n), set(va_n), set(te_n)

    def _prepare_records(self, test_frac, val_frac) -> List[dict]:
        all_entries = self._read_list()
        tr_s, va_s, te_s = self._split(all_entries, test_frac, val_frac)
        sel = {"train": tr_s, "val": va_s, "test": te_s}[self.partition]

        records = []
        for e in all_entries:
            name = e["name"]
            if name not in sel:
                continue
            img_path    = self.img_dir    / f"{name}.jpg"
            mask_path   = self.trimap_dir / f"{name}.png"
            xml_path    = self.xml_dir    / f"{name}.xml"
            has_mask = mask_path.exists()
            has_xml  = xml_path.exists()
            if not img_path.exists():
                continue
            if   self.mode == "seg" and not has_mask:
                continue
            elif self.mode == "loc" and not has_xml:
                continue
            elif self.mode == "all" and not (has_mask and has_xml):
                continue
            records.append({
                "name": name, "label": e["label"],
                "has_mask": has_mask, "has_xml": has_xml,
            })

        print(f"[Dataset] partition={self.partition} mode={self.mode} "
              f"samples={len(records)}")
        return records

    # ------------------------------------------------------------------
    def _read_trimap(self, name: str) -> Optional[np.ndarray]:
        path = self.trimap_dir / f"{name}.png"
        if not path.exists():
            return None
        raw = np.array(Image.open(path), dtype=np.int32)
        return (raw - 1).clip(0, 2).astype(np.uint8)   # {1,2,3} → {0,1,2}

    def _read_bbox_xyxy(self, name: str) -> Optional[List[float]]:
        path = self.xml_dir / f"{name}.xml"
        if not path.exists():
            return None
        root = ET.parse(path).getroot()
        obj  = root.find("object")
        if obj is None:
            return None
        box = obj.find("bndbox")
        return [
            float(box.find("xmin").text),
            float(box.find("ymin").text),
            float(box.find("xmax").text),
            float(box.find("ymax").text),
        ]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec  = self.records[idx]
        name = rec["name"]

        image  = np.array(Image.open(self.img_dir / f"{name}.jpg").convert("RGB"))
        trimap = self._read_trimap(name)
        if trimap is None:
            trimap = np.zeros(image.shape[:2], dtype=np.uint8)

        raw_bbox = self._read_bbox_xyxy(name)
        bboxes      = [raw_bbox] if raw_bbox is not None else []
        bbox_labels = [0]        if raw_bbox is not None else []

        aug         = self.pipeline(image=image, mask=trimap,
                                    bboxes=bboxes, bbox_labels=bbox_labels)
        img_tensor  = aug["image"]
        mask_tensor = aug["mask"].long()

        if aug["bboxes"]:
            x1, y1, x2, y2 = aug["bboxes"][0]
            bbox_tensor = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
            bbox_valid  = torch.tensor(1.0)
        else:
            bbox_tensor = torch.zeros(4, dtype=torch.float32)
            bbox_valid  = torch.tensor(0.0)

        return {
            "image":     img_tensor,
            "label":     torch.tensor(rec["label"], dtype=torch.long),
            "mask":      mask_tensor if rec["has_mask"] else None,
            "bbox":      bbox_tensor,
            "bbox_mask": bbox_valid,
            "name":      name,
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(batch: List[dict]) -> dict:
    images = torch.stack([s["image"] for s in batch])
    labels = torch.stack([s["label"] for s in batch])
    H, W   = images.shape[2:]

    masks = torch.stack([
        s["mask"] if s["mask"] is not None
        else torch.full((H, W), -1, dtype=torch.long)
        for s in batch
    ])
    bboxes     = torch.stack([s["bbox"]      for s in batch])
    bbox_masks = torch.stack([s["bbox_mask"] for s in batch])

    return {
        "image":     images,
        "label":     labels,
        "mask":      masks,
        "bbox":      bboxes,
        "bbox_mask": bbox_masks,
    }
# ---------------------------------------------------------------------------
# Convenience loader builder (kept for backward compat with train_tasks/)
# ---------------------------------------------------------------------------

def get_dataloaders(
    root: str,
    task: str = "cls",
    batch_size: int = 32,
    num_workers: int = 4,
    test_frac: float = 0.10,
    val_frac:  float = 0.10,
):
    """
    Returns (train_loader, val_loader, test_loader).

    task: 'cls' | 'loc' | 'seg' | 'all'
    """
    kw = dict(root=root, mode=task, test_frac=test_frac, val_frac=val_frac)
    tr = OxfordPetDataset(**kw, partition="train")
    va = OxfordPetDataset(**kw, partition="val")
    te = OxfordPetDataset(**kw, partition="test")

    loader_kw = dict(batch_size=batch_size, num_workers=num_workers,
                     pin_memory=True, collate_fn=collate_fn)
    return (
        DataLoader(tr, shuffle=True,  **loader_kw),
        DataLoader(va, shuffle=False, **loader_kw),
        DataLoader(te, shuffle=False, **loader_kw),
    )

# Aliases
OxfordIIITPetDataset = OxfordPetDataset
PetDataset           = OxfordPetDataset