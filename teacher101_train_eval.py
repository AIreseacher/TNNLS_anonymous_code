#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json, os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# -------------------------
# Utils: metrics & composite
# -------------------------
def evaluate_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Dict[str, float]:
    # logits: [N,C], targets: [N]
    probs = F.softmax(logits, dim=1).cpu().numpy()
    y_true = targets.cpu().numpy()
    y_pred = probs.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    try:
        auc_ovo = roc_auc_score(y_true, probs, multi_class="ovo")
    except Exception:
        auc_ovo = float("nan")
    try:
        auc_ovr = roc_auc_score(y_true, probs, multi_class="ovr")
    except Exception:
        auc_ovr = float("nan")
    return {"acc": float(acc), "macro_f1": float(macro_f1),
            "auc_ovo": float(auc_ovo), "auc_ovr": float(auc_ovr)}

def composite_score(d: Dict[str, float]) -> float:
    keys = ["acc", "macro_f1", "auc_ovr", "auc_ovo"]
    vals = [d[k] for k in keys if d.get(k) is not None and not np.isnan(d[k])]
    return float(np.mean(vals)) if vals else 0.0

# -------------------------
# JSON dataset
# -------------------------
class JsonClsDataset(Dataset):
    def __init__(self, entries: List[dict], tfm):
        self.items = []
        for it in entries:
            if isinstance(it, dict) and "name" in it and "label" in it:
                self.items.append({"path": it["name"], "label": int(it["label"])})
            elif isinstance(it, (list, tuple)) and len(it) >= 2:
                self.items.append({"path": str(it[0]), "label": int(it[1])})
        self.tfm = tfm

    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        p = self.items[idx]["path"]; y = self.items[idx]["label"]
        img = Image.open(p).convert("RGB")
        x = self.tfm(img)
        return {"image": x, "label": torch.tensor(y, dtype=torch.long)}

def build_transform() -> transforms.Compose:
    # Normalization
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def load_splits(json_path: str) -> Dict[str, List[dict]]:
    with open(json_path, "r") as f:
        raw = json.load(f)
    def pick(key):
        for k,v in raw.items():
            if k.lower() == key.lower():
                return v
        return []
    return {"train": pick("train"), "val": pick("val") or pick("validation"), "test": pick("test")}

def infer_num_classes(train_entries: List[dict]) -> int:
    labels = set(int(it["label"] if isinstance(it, dict) else it[1]) for it in train_entries)
    return int(len(labels))

# -------------------------
# Evaluate one split (no grad)
# -------------------------
@torch.no_grad()
def evaluate_split(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, num_classes: int):
    model.eval()
    loss_sum, n = 0.0, 0
    all_logits, all_targets = [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += float(loss) * y.size(0)
        n += y.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())
    loss = loss_sum / max(n, 1)
    if n == 0:
        return {"loss": 0.0, "acc": 0.0, "macro_f1": 0.0, "auc_ovo": 0.0, "auc_ovr": 0.0}
    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    m = evaluate_metrics(logits_cat, targets_cat, num_classes=num_classes)
    m["loss"] = float(loss)
    return m

# -------------------------
# Main
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser("Teacher(ResNet101) train/eval without YAML")
    ap.add_argument("--json_path", type=str, required=True)
    ap.add_argument("--dataset",   type=str, required=True)
    ap.add_argument("--save_dir",  type=str, required=True)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--epochs",    type=int, default=100)
    ap.add_argument("--lr",        type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size",   type=int, default=256)
    ap.add_argument("--num_workers",  type=int, default=4)
    ap.add_argument("--log_mode",     type=str, default="w", choices=["r","w","b"])
    return ap.parse_args()

def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    set_seed(args.seed)

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    r_log = save_dir / "r_log.log"
    w_log = save_dir / "w_log.log"

    # ----- data -----
    splits = load_splits(args.json_path)
    num_classes = infer_num_classes(splits["train"])
    tfm = build_transform()
    loaders = {}
    for k in ("train","val","test"):
        if splits.get(k):
            ds = JsonClsDataset(splits[k], tfm)
            loaders[k] = DataLoader(ds, batch_size=args.batch_size if k=="train" else 128,
                                    shuffle=(k=="train"), num_workers=args.num_workers, pin_memory=True)
    # ----- model / optim -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models.resnet101(pretrained=False, num_classes=num_classes)
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    best_score = -1e9
    best_path: Optional[Path] = None

    print("\n=== Teacher101 Training Summary ===")
    print(f"Dataset   : {args.dataset}")
    print(f"Save dir  : {save_dir.as_posix()}")
    print(f"Seed      : {args.seed}")
    print(f"Epochs    : {args.epochs} | LR={args.lr} | WD={args.weight_decay} | BS={args.batch_size}")
    print(f"Log mode  : {args.log_mode}  (r=train+val, w=train+val+test, b=both)")  # 与 main.py 对齐:contentReference[oaicite:5]{index=5}

    for epoch in range(1, args.epochs+1):
        # ----- train one epoch -----
        net.train()
        seen = 0
        tr_loss_sum = 0.0
        for batch in loaders["train"]:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = net(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            tr_loss_sum += float(loss) * y.size(0)
            seen += y.size(0)
        tr_loss = tr_loss_sum / max(seen, 1)

        # ----- eval -----
        tr_report = evaluate_split(net, loaders["train"], device, ce, num_classes)
        va_report = evaluate_split(net, loaders.get("val"), device, ce, num_classes) if "val" in loaders else None
        te_report = evaluate_split(net, loaders.get("test"), device, ce, num_classes) if args.log_mode in ("w","b") and "test" in loaders else None

        comp = composite_score({
            "acc": (va_report or {}).get("acc", np.nan),
            "macro_f1": (va_report or {}).get("macro_f1", np.nan),
            "auc_ovr": (va_report or {}).get("auc_ovr", np.nan),
            "auc_ovo": (va_report or {}).get("auc_ovo", np.nan),
        })

        # ----- logging (格式对齐 main.py) -----
        r_line = (f"[Epoch {epoch:03d}][R] "
                  f"train: loss={tr_report['loss']:.4f}, acc={tr_report['acc']:.4f}, "
                  f"f1={tr_report['macro_f1']:.4f}, auc_ovo={tr_report['auc_ovo']:.4f}, auc_ovr={tr_report['auc_ovr']:.4f} | ")
        if va_report:
            r_line += (f"val: loss={va_report['loss']:.4f}, acc={va_report['acc']:.4f}, "
                       f"f1={va_report['macro_f1']:.4f}, auc_ovo={va_report['auc_ovo']:.4f}, auc_ovr={va_report['auc_ovr']:.4f} | "
                       f"composite={comp:.4f}")
        else:
            r_line += "val: NA | composite=0.0000"

        w_line = None
        if te_report is not None:
            w_line = (f"[Epoch {epoch:03d}][W] "
                      f"train: loss={tr_report['loss']:.4f}, acc={tr_report['acc']:.4f}, "
                      f"f1={tr_report['macro_f1']:.4f}, auc_ovo={tr_report['auc_ovo']:.4f}, auc_ovr={tr_report['auc_ovr']:.4f} | ")
            if va_report:
                w_line += (f"val: loss={va_report['loss']:.4f}, acc={va_report['acc']:.4f}, "
                           f"f1={va_report['macro_f1']:.4f}, auc_ovo={va_report['auc_ovo']:.4f}, auc_ovr={va_report['auc_ovr']:.4f} | ")
            w_line += (f"test: loss={te_report['loss']:.4f}, acc={te_report['acc']:.4f}, "
                       f"f1={te_report['macro_f1']:.4f}, auc_ovo={te_report['auc_ovo']:.4f}, auc_ovr={te_report['auc_ovr']:.4f})")

        if args.log_mode == "r":
            print(r_line)
            with open(r_log, "a") as f: f.write(r_line + "\n")
        elif args.log_mode == "w":
            if w_line:
                print(w_line)
                with open(w_log, "a") as f: f.write(w_line + "\n")
        else:  # b
            with open(r_log, "a") as f: f.write(r_line + "\n")
            if w_line:
                print(w_line)
                with open(w_log, "a") as f: f.write(w_line + "\n")

        # ----- checkpoints（字段对齐 main.py 的 CheckpointEngine）:contentReference[oaicite:6]{index=6}
        last_pth = save_dir / "last.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": None,
            "metrics": {"val": va_report, "train": tr_report, "test": te_report},
            "cfg": {
                "data": {"dataset": args.dataset, "json_path": args.json_path},
                "model": {"backbone_teacher": "resnet101", "num_classes": num_classes},
                "train": {"epochs": args.epochs, "base_lr": args.lr, "weight_decay": args.weight_decay},
                "output": {"save_dir": save_dir.as_posix(), "log_mode": args.log_mode},
            },
        }, last_pth.as_posix())

        if va_report:
            score = comp
            if score > best_score:
                best_score = score
                best_pth = save_dir / "best.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": None,
                    "metrics": {"val": va_report, "train": tr_report, "test": te_report},
                    "best_score": best_score,
                }, best_pth.as_posix())
                print(f" New best @ epoch {epoch}: {best_pth.as_posix()} (score={best_score:.4f})")

    # final test: contentReference[oaicite:7]{index=7}
    if "test" in loaders:
        final = evaluate_split(net, loaders["test"], device, ce, num_classes)
        print(f"[Final-Test] loss={final['loss']:.4f}, acc={final['acc']:.4f}, "
              f"f1={final['macro_f1']:.4f}, auc_ovo={final['auc_ovo']:.4f}, auc_ovr={final['auc_ovr']:.4f}")

if __name__ == "__main__":
    main()
