#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal pretraining script:
- Robust JSON reader (supports list/tuple or dict layouts)
- Backbones: resnet101, vit-b (vit_b_16), wideresnet101 (wide_resnet101_2)
- Torchvision version compatibility (weights= / pretrained=)
- Saves state_dict to a chosen directory with a consistent filename
"""

import os
import json
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ----------------------------- CLI -----------------------------
def build_args():
    p = argparse.ArgumentParser(description="Backbone pretraining")
    # data & io
    p.add_argument("--json_path", type=str, required=True, help="Path to the dataset JSON")
    p.add_argument("--dataset", type=str, required=True, help="Dataset name (used for naming output)")
    p.add_argument("--subset", type=str, default="pretrain", help="Subset key in JSON (e.g., pretrain/train/all)")
    p.add_argument("--save_dir", type=str, required=True, help="Directory to save pretrained weights")
    # training
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--early_stop_acc", type=float, default=97.0)
    p.add_argument("--num_workers", type=int, default=4)
    # backbone
    p.add_argument("--backbone", type=str, default="resnet101",
                   choices=["resnet101", "vit-b", "wideresnet101"],
                   help="Teacher backbone to pretrain")
    return p.parse_args()

# -------- Torchvision version compatibility helpers -----------
def _tv_create(model_fn, **kwargs):
    """
    Try creating a torchvision model with the new API (weights=...).
    Fallback to the old API (pretrained=...) if the environment is older.
    """
    try:
        return model_fn(**kwargs)  # new API path
    except TypeError:
        if 'weights' in kwargs:
            kwargs.pop('weights', None)
        kwargs.setdefault('pretrained', False)
        return model_fn(**kwargs)

def _extract_vit_head_and_dim(vit):
    """
    Return (container_module_for_classifier, in_features) and set classifier to Identity.
    Works across torchvision versions (heads/head differences).
    """
    # torchvision >= 0.13 often has vit.heads.head
    if hasattr(vit, 'heads') and hasattr(vit.heads, 'head'):
        in_f = vit.heads.head.in_features
        vit.heads.head = nn.Identity()
        return vit.heads, in_f
    # some versions expose vit.heads as a single Linear
    if hasattr(vit, 'heads') and hasattr(vit.heads, 'in_features'):
        in_f = vit.heads.in_features
        vit.heads = nn.Identity()
        return vit, in_f
    raise AttributeError("Unsupported ViT structure; cannot find classifier head.")

# -------------------- JSON reading (robust) --------------------
def _pick_subset(data, subset_key: str):
    """
    Pick a subset from JSON in a robust manner.
    Accepts:
      - dict with a key == subset_key (case-insensitive)
      - nested dicts (e.g., { 'Brain': {'pretrain': [...] }})
      - top-level list (treated as samples)
    """
    if isinstance(data, dict):
        if subset_key in data:
            return data[subset_key]
        for k, v in data.items():
            if isinstance(v, list) and k.lower() == subset_key.lower():
                return v
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if kk.lower() == subset_key.lower():
                        return vv
        # case-insensitive top level
        for k in data.keys():
            if k.lower() == subset_key.lower():
                return data[k]
    elif isinstance(data, list):
        return data
    raise KeyError(f"Cannot find subset '{subset_key}' in JSON.")

def _extract_path_and_label(item) -> Tuple[str, int]:
    """
    Normalize a single sample into (path, label).
    Supports:
      - list/tuple: [path, label]
      - dict: with any of common path/label keys
    """
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return str(item[0]), int(item[1])

    if isinstance(item, dict):
        path_keys = ["name", "image", "img", "path", "file", "file_path", "filepath", "image_path", "im", "fname"]
        label_keys = ["label", "target", "cls", "class", "y", "category_id"]
        path_val, label_val = None, None
        for k in path_keys:
            if k in item:
                path_val = item[k]
                break
        for k in label_keys:
            if k in item:
                label_val = item[k]
                break
        if path_val is None or label_val is None:
            raise KeyError(f"Missing path/label in item keys={list(item.keys())}")
        return str(path_val), int(label_val)

    raise TypeError(f"Unsupported sample type: {type(item)}")

def load_samples_from_json(json_path: str, subset: str) -> Tuple[List[str], List[int]]:
    with open(json_path, "r") as f:
        raw = json.load(f)
    samples = _pick_subset(raw, subset)
    files, labels = [], []
    for it in samples:
        p, y = _extract_path_and_label(it)
        files.append(p)
        labels.append(y)
    if len(files) == 0:
        raise RuntimeError(f"No samples found in {json_path} (subset={subset})")
    return files, labels

# --------------------------- Dataset ---------------------------
class ImgClsDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], transform=None):
        assert len(paths) == len(labels), "Mismatched paths and labels length"
        self.paths = paths
        self.labels = labels
        self.t = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.t:
            img = self.t(img)
        return img, int(self.labels[idx])

def build_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# --------------- Backbones → (features, logits) ----------------
class ForwardWithFeatures(nn.Module):
    """Wrap a backbone so forward(x) returns (features, logits)."""
    def __init__(self, backbone: nn.Module, head: nn.Module, feature_hook):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.feature_hook = feature_hook

    def forward(self, x):
        return self.feature_hook(self.backbone, self.head, x)

def build_resnet101(nc: int) -> nn.Module:
    m = _tv_create(tv.resnet101, weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Identity()
    head = nn.Linear(in_f, nc)
    def hook(bk, hd, x):
        feats = bk(x)      # pooled features
        logits = hd(feats)
        return feats, logits
    return ForwardWithFeatures(m, head, hook)

def build_wideresnet101(nc: int) -> nn.Module:
    m = _tv_create(tv.wide_resnet101_2, weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Identity()
    head = nn.Linear(in_f, nc)
    def hook(bk, hd, x):
        feats = bk(x)
        logits = hd(feats)
        return feats, logits
    return ForwardWithFeatures(m, head, hook)

def build_vit_b(nc: int) -> nn.Module:
    vit = _tv_create(tv.vit_b_16, weights=None)
    container, in_f = _extract_vit_head_and_dim(vit)  # set classifier to Identity
    head = nn.Linear(in_f, nc)
    def hook(bk, hd, x):
        feats = bk(x)      # CLS embedding when classifier is Identity
        logits = hd(feats)
        return feats, logits
    return ForwardWithFeatures(vit, head, hook)

def build_backbone(name: str, nc: int) -> nn.Module:
    name = name.lower()
    if name == "resnet101":
        return build_resnet101(nc)
    if name in ("wideresnet101", "wide_resnet101", "wide-resnet101"):
        return build_wideresnet101(nc)
    if name in ("vit-b", "vit_b", "vit-b-16", "vit_b_16"):
        return build_vit_b(nc)
    raise ValueError(f"Unsupported backbone: {name}")

# --------------------------- Training --------------------------
def main():
    args = build_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files, labels = load_samples_from_json(args.json_path, args.subset)
    num_classes = len(set(labels))

    # basic sanity check
    if not os.path.exists(files[0]):
        raise FileNotFoundError(f"Image path not found: {files[0]}")

    ds = ImgClsDataset(files, labels, build_transform())
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True)

    model = build_backbone(args.backbone, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for ep in range(args.num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for imgs, ys in dl:
            imgs = imgs.to(device, non_blocking=True)
            ys = ys.long().to(device, non_blocking=True)

            optimizer.zero_grad()
            feats, logits = model(imgs)
            loss = criterion(logits, ys)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            correct += (pred == ys).sum().item()
            total += ys.size(0)

        acc = 100.0 * correct / max(1, total)
        print(f"[{args.dataset} | {args.backbone}] "
              f"Epoch {ep+1}/{args.num_epochs}  "
              f"Loss {running_loss/len(dl):.4f}  Acc {acc:.2f}%")

        if acc >= args.early_stop_acc:
            print(f"[Early stop] Acc {acc:.2f}% >= {args.early_stop_acc}%.")
            break

    os.makedirs(args.save_dir, exist_ok=True)
    tag = args.backbone.upper().replace("-", "_")
    save_path = os.path.join(args.save_dir, f"Pretrain_{tag}_{args.dataset}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"[Saved] → {save_path}")

if __name__ == "__main__":
    main()
