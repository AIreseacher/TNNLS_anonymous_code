# data/data.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .utils import read_json, resolve_json_path, to_abs_path


# =============================
# JSON schema dataclasses
# =============================
@dataclass
class JsonRecord:
    """A single sample record parsed from JSON."""
    path: str
    label: int
    h: Optional[Any] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class JsonSplit:
    """All splits loaded from JSON."""
    pretrain: Optional[List[JsonRecord]]
    train: List[JsonRecord]
    validation: List[JsonRecord]
    test: List[JsonRecord]


# =============================
# JSON loading helpers
# =============================
def _coerce_items(seq: List[Dict[str, Any]]) -> List[JsonRecord]:
    """
    Convert a list of raw dict items into a list of JsonRecord.
    Required per item:
      - 'name' (or alias 'file' / 'image'): image path
      - 'label' (int)
    Optional per item:
      - 'h' (any structure for SRA or metadata)
      - 'meta' (free-form dict)
    """
    out: List[JsonRecord] = []
    for it in seq:
        path = it.get("name") or it.get("file") or it.get("image")
        if path is None:
            raise KeyError("Each JSON item must contain 'name' (or 'file'/'image').")
        label = it.get("label")
        if label is None:
            raise KeyError("Each JSON item must contain 'label' (int).")
        out.append(JsonRecord(path=str(path), label=int(label), h=it.get("h"), meta=it.get("meta")))
    return out


def load_json_all(json_dir: str, dataset: str) -> Tuple[JsonSplit, Dict[str, Any]]:
    """
    Load json/<Dataset>.json and return:
      - JsonSplit (pretrain/train/validation/test)
      - the full raw dict (to access optional 'classes' / 'prior' etc.)

    Supported top-level shapes:
      1) Flat:
         {
           "train": [...],
           "validation": [...],
           "test": [...],
           "pretrain": [...],      # optional
           "classes": [...],       # optional
           "prior": {...}          # optional
         }
      2) With "splits" wrapper:
         { "splits": { ... }, "classes": [...], "prior": {...} }
    """
    path = resolve_json_path(dataset, json_dir)
    blob = read_json(path)
    splits = blob.get("splits") or blob

    train = splits.get("train")
    val   = splits.get("validation")
    test  = splits.get("test")
    pre   = splits.get("pretrain", None)

    if train is None or val is None or test is None:
        raise KeyError("JSON must contain 'train', 'validation', and 'test' splits.")

    js = JsonSplit(
        pretrain=_coerce_items(pre) if pre is not None else None,
        train=_coerce_items(train),
        validation=_coerce_items(val),
        test=_coerce_items(test),
    )
    return js, blob


# =============================
# Transforms (defaults in code)
# =============================
def _default_aug(is_train: bool) -> transforms.Compose:
    """
    Default augmentations (no YAML needed):
      - Resize(224, 224)
      - For training: RandomHorizontalFlip()
      - ToTensor + Normalize (ImageNet stats)
    """
    ops: List[Any] = []
    size = (224, 224)
    ops.append(transforms.Resize(size))
    if is_train:
        ops.append(transforms.RandomHorizontalFlip())
        # For stronger aug you could switch to:
        # ops.append(transforms.RandomResizedCrop(size, scale=(0.9, 1.0)))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]))
    return transforms.Compose(ops)


def build_transforms(cfg: Dict[str, Any]) -> Dict[str, transforms.Compose]:
    """
    Build transforms. By default it uses _default_aug().
    If cfg['augment'] is provided (e.g., via command-line overrides),
    it will partially override the defaults for a few common options:
      - resize: [H, W]
      - rand_flip: true/false
      - rand_crop: true/false (RandomResizedCrop)
      - mean/std: lists of 3 floats
    """
    aug = cfg.get("augment", {})

    def adapt(base: transforms.Compose, node: Dict[str, Any], is_train: bool) -> transforms.Compose:
        if not node:
            return base

        size = tuple(node.get("resize", [])) or None
        rand_flip = node.get("rand_flip", None)
        rand_crop = node.get("rand_crop", None)
        mean = node.get("mean", None)
        std  = node.get("std", None)

        ops: List[Any] = []
        # Resize
        if size:
            ops.append(transforms.Resize(size))
        else:
            # reuse the first transform of the base (Resize)
            ops.append(base.transforms[0])

        # Train-time aug
        if is_train:
            if rand_crop:
                ops.append(transforms.RandomResizedCrop(size or (224, 224), scale=(0.9, 1.0)))
            if rand_flip is True:
                ops.append(transforms.RandomHorizontalFlip())

        # To tensor + normalize
        ops.append(transforms.ToTensor())
        if mean is None or std is None:
            ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]))
        else:
            ops.append(transforms.Normalize(mean=mean, std=std))
        return transforms.Compose(ops)

    t_train = adapt(_default_aug(True),  aug.get("train", {}), True)
    t_eval  = adapt(_default_aug(False), aug.get("eval",  {}), False)
    return {"train": t_train, "eval": t_eval}


# =============================
# Auto-infer number of classes
# =============================
def infer_num_classes(blob: Dict[str, Any], splits: JsonSplit) -> Tuple[int, List[str]]:
    """
    Infer (num_classes, class_names) from JSON:
      1) If 'classes' exists and non-empty, use its length and order.
      2) Otherwise compute the union of labels across train/validation/test.
         num_classes = max_label + 1 (robust even if min label > 0).
         class_names are synthesized: ['class_0', 'class_1', ...]
    """
    classes = blob.get("classes", None)
    if isinstance(classes, list) and len(classes) > 0:
        return len(classes), list(classes)

    labels: Set[int] = set()
    for rec in (splits.train + splits.validation + splits.test):
        labels.add(int(rec.label))
    if not labels:
        raise ValueError("Cannot infer num_classes: no labels found in JSON splits.")
    num_classes = max(labels) + 1
    class_names = [f"class_{i}" for i in range(num_classes)]
    return num_classes, class_names


# =============================
# Dataset
# =============================
class MedImageDataset(Dataset):
    """
    Dataset driven by JSON splits.

    __getitem__ returns a dict:
      {
        'image': Tensor[C,H,W],
        'label': int,
        'meta' : {'path': str, 'h': Any|None, ...}
      }
    """
    def __init__(
        self,
        items: List[JsonRecord],
        transform: transforms.Compose,
        root: str,
        class_names: Optional[List[str]] = None,
    ):
        self.items = items
        self.transform = transform
        self.root = root
        self.class_names = class_names or []

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, abs_path: str) -> Image.Image:
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"[MedImageDataset] Image not found: {abs_path}")
        return Image.open(abs_path).convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.items[idx]
        abs_path = to_abs_path(self.root, rec.path)
        img = self._load_image(abs_path)
        img = self.transform(img)
        meta = {"path": abs_path, "h": rec.h}
        if rec.meta:
            meta.update(rec.meta)
        return {"image": img, "label": rec.label, "meta": meta}


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack images/labels and keep metas as a list."""
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
    metas = [b.get("meta", {}) for b in batch]
    return {"image": imgs, "label": labels, "meta": metas}


# =============================
# Public builders (what test.py imports)
# =============================
def build_dataloaders(cfg: Dict[str, Any]) -> Dict[str, DataLoader]:
    """
    Build train/validation/test DataLoaders using YAML fields:
      - cfg['data']['dataset']   : dataset name (JSON file name)
      - cfg['data']['json_dir']  : folder holding <Dataset>.json
      - cfg['data']['root']      : image root to resolve relative paths
      - cfg['data']['batch_size']
      - cfg['data']['num_workers']

    Also:
      - If cfg['model']['num_classes'] is missing, infer it from JSON and write back.
      - We also write class names (if inferred or provided in JSON) to cfg['model']['class_names'].
    """
    data_cfg = cfg.get("data", {})
    dataset_name: str = data_cfg["dataset"]
    json_dir: str = data_cfg["json_dir"]
    root: str = data_cfg.get("root", ".")
    batch_size: int = int(data_cfg.get("batch_size", 64))
    num_workers: int = int(data_cfg.get("num_workers", 8))

    splits, blob = load_json_all(json_dir, dataset_name)

    # Infer number of classes and optional names; write back to cfg for later use.
    num_classes, class_names = infer_num_classes(blob, splits)
    cfg.setdefault("model", {})
    cfg["model"].setdefault("num_classes", num_classes)
    cfg["model"].setdefault("class_names", class_names)

    tfms = build_transforms(cfg)
    ds_train = MedImageDataset(splits.train,      tfms["train"], root, class_names)
    ds_val   = MedImageDataset(splits.validation, tfms["eval"],  root, class_names)
    ds_test  = MedImageDataset(splits.test,       tfms["eval"],  root, class_names)

    loader_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, collate_fn=collate_fn
    )
    loader_val = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False, collate_fn=collate_fn
    )
    loader_test = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False, collate_fn=collate_fn
    )

    # Provide both 'val' and 'validation' keys for convenience.
    return {"train": loader_train, "val": loader_val, "validation": loader_val, "test": loader_test}


# Backward-compat alias in case some scripts still import this old name.
def build_dataloaders_from_json(cfg: Dict[str, Any]) -> Dict[str, DataLoader]:
    return build_dataloaders(cfg)
