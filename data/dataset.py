import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Default transforms
# ---------------------------------------------------------------------------
def get_image_transforms(train: bool = True, image_size: int = 224):
    if train:
        return T.Compose([
            T.Resize((image_size + 32, image_size + 32)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])


def get_mask_transforms(image_size: int = 224):
    """Resize mask with nearest-neighbour to avoid interpolating class indices."""
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
    ])


# ---------------------------------------------------------------------------
# Base helper: parse annotation list
# ---------------------------------------------------------------------------
def _parse_annotation_list(annotations_dir: Path, split: str):
    """
    Returns list of (image_stem, class_id_1indexed, species_id, breed_id).
    class_id is 1-indexed in the file; we convert to 0-indexed.
    """
    filepath = annotations_dir / f"{split}.txt"
    entries = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            stem     = parts[0]
            class_id = int(parts[1]) - 1  # 0-indexed
            entries.append((stem, class_id))
    return entries


def _parse_bbox(xml_path: Path, img_w: int, img_h: int) -> Optional[torch.Tensor]:
    """
    Parse the first <object> bounding box from an annotation XML.
    Returns normalised [cx, cy, w, h] tensor, or None if file missing.
    """
    if not xml_path.exists():
        return None
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find(".//bndbox")
        if bndbox is None:
            return None
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        w  = (xmax - xmin) / img_w
        h  = (ymax - ymin) / img_h
        return torch.tensor([cx, cy, w, h], dtype=torch.float32)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Task 1: Classification Dataset
# ---------------------------------------------------------------------------
class PetClassificationDataset(Dataset):
    def __init__(self, root: str, split: str = "trainval", transform=None):
        self.root     = Path(root)
        self.img_dir  = self.root / "images"
        self.ann_dir  = self.root / "annotations"
        self.transform = transform or get_image_transforms(train=(split == "trainval"))

        self.samples = _parse_annotation_list(self.ann_dir, split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem, label = self.samples[idx]
        img_path    = self.img_dir / f"{stem}.jpg"
        image       = Image.open(img_path).convert("RGB")
        image       = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Task 2: Localization Dataset
# ---------------------------------------------------------------------------
class PetLocalizationDataset(Dataset):
    """
    Returns (image, bbox) where bbox = [cx, cy, w, h] normalised to [0,1].
    Samples without a valid XML annotation are skipped.
    """

    def __init__(self, root: str, split: str = "trainval", transform=None):
        self.root    = Path(root)
        self.img_dir = self.root / "images"
        self.xml_dir = self.root / "annotations" / "xmls"
        self.transform = transform or get_image_transforms(train=(split == "trainval"))

        all_samples = _parse_annotation_list(self.root / "annotations", split)

        # Filter: only keep samples that have a corresponding XML
        self.samples = [
            (stem, label) for stem, label in all_samples
            if (self.xml_dir / f"{stem}.xml").exists()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem, label = self.samples[idx]
        img_path = self.img_dir / f"{stem}.jpg"
        image    = Image.open(img_path).convert("RGB")
        w, h     = image.size
        bbox     = _parse_bbox(self.xml_dir / f"{stem}.xml", w, h)

        image = self.transform(image)
        return image, bbox


# ---------------------------------------------------------------------------
# Task 3: Segmentation Dataset
# ---------------------------------------------------------------------------
class PetSegmentationDataset(Dataset):
    """
    Returns (image, mask) where mask is a LongTensor with values 0, 1, 2
    (foreground, background, boundary) — remapped from the trimap 1, 2, 3.
    """

    def __init__(self, root: str, split: str = "trainval",
                 image_transform=None, image_size: int = 224):
        self.root          = Path(root)
        self.img_dir       = self.root / "images"
        self.mask_dir      = self.root / "annotations" / "trimaps"
        self.image_size    = image_size
        self.img_transform = image_transform or get_image_transforms(
            train=(split == "trainval"), image_size=image_size
        )
        self.mask_resize   = get_mask_transforms(image_size)
        self.samples       = _parse_annotation_list(self.root / "annotations", split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem, label = self.samples[idx]
        image = Image.open(self.img_dir / f"{stem}.jpg").convert("RGB")
        mask  = Image.open(self.mask_dir / f"{stem}.png")  # palette PNG

        image = self.img_transform(image)
        mask  = self.mask_resize(mask)
        mask  = torch.from_numpy(np.array(mask)).long() - 1  # 1,2,3 → 0,1,2

        return image, mask


# ---------------------------------------------------------------------------
# Task 4: Multi-task Dataset
# ---------------------------------------------------------------------------
class PetMultiTaskDataset(Dataset):
    """
    Returns (image, class_label, bbox, mask).
    Samples without XML are included; bbox will be zeros tensor.
    """

    def __init__(self, root: str, split: str = "trainval",
                 image_transform=None, image_size: int = 224):
        self.root          = Path(root)
        self.img_dir       = self.root / "images"
        self.xml_dir       = self.root / "annotations" / "xmls"
        self.mask_dir      = self.root / "annotations" / "trimaps"
        self.image_size    = image_size
        self.img_transform = image_transform or get_image_transforms(
            train=(split == "trainval"), image_size=image_size
        )
        self.mask_resize   = get_mask_transforms(image_size)
        self.samples       = _parse_annotation_list(self.root / "annotations", split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem, label = self.samples[idx]

        # Image
        image = Image.open(self.img_dir / f"{stem}.jpg").convert("RGB")
        w, h  = image.size
        image = self.img_transform(image)

        # BBox
        bbox_path = self.xml_dir / f"{stem}.xml"
        bbox = _parse_bbox(bbox_path, w, h)
        if bbox is None:
            bbox = torch.zeros(4, dtype=torch.float32)

        # Mask
        mask_path = self.mask_dir / f"{stem}.png"
        if mask_path.exists():
            mask = Image.open(mask_path)
            mask = self.mask_resize(mask)
            mask = torch.from_numpy(np.array(mask)).long() - 1
        else:
            mask = torch.zeros(self.image_size, self.image_size, dtype=torch.long)

        return image, torch.tensor(label, dtype=torch.long), bbox, mask


# ---------------------------------------------------------------------------
# Convenience: build DataLoaders for any task
# ---------------------------------------------------------------------------
def get_dataloaders(
    root: str,
    task: str = "classification",    # "classification" | "localization" | "segmentation" | "multitask"
    image_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
):
    dataset_cls = {
        "classification": PetClassificationDataset,
        "localization":   PetLocalizationDataset,
        "segmentation":   PetSegmentationDataset,
        "multitask":      PetMultiTaskDataset,
    }[task]

    kwargs = dict(root=root, image_size=image_size) if task in ("segmentation", "multitask") else dict(root=root)

    full_train = dataset_cls(**kwargs, split="trainval")
    test_ds    = dataset_cls(**kwargs, split="test")

    n_val   = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
