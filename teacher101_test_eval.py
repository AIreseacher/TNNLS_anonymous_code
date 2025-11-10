#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate Teacher101 best.pth for datasets: Chaoyang, Brain, Breast
Append results to /nas/unas15/chentao/TNNLS/checkpoints/teacher101.csv
Columns (strict order):
dataset, pair, method, method_pretty, seed, ckpt_name, acc, macro_f1, auc_ovo, auc_ovr, ckpt_path
"""

import csv
import json
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# --------------------- Settings ---------------------
ROOT = Path("/nas/unas15/chentao/TNNLS/checkpoints")
DATASETS = ("Chaoyang", "Brain", "Breast")
PAIR_DIR = "Teacher101"
CKPT_NAME = "best.pth"
JSON_FALLBACK_DIR = Path("/nas/unas15/chentao/TNNLS/json")
OUT_CSV = ROOT / "teacher101.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_TEST = 128
NUM_WORKERS = 4
COLUMNS = ["dataset", "pair", "method", "method_pretty", "seed",
           "ckpt_name", "acc", "macro_f1", "auc_ovo", "auc_ovr", "ckpt_path"]
# ---------------------------------------------------


# --------------------- Data & Eval ---------------------
class JsonClsDataset(Dataset):
    def __init__(self, entries, tfm):
        items = []
        for it in entries:
            if isinstance(it, dict) and "name" in it and "label" in it:
                items.append({"path": it["name"], "label": int(it["label"])})
            elif isinstance(it, (list, tuple)) and len(it) >= 2:
                items.append({"path": str(it[0]), "label": int(it[1])})
        self.items = items
        self.tfm = tfm

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        p = self.items[idx]["path"]; y = self.items[idx]["label"]
        img = Image.open(p).convert("RGB")
        x = self.tfm(img)
        return {"image": x, "label": torch.tensor(y, dtype=torch.long)}


def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])


def load_splits(json_path: Path):
    with open(json_path, "r") as f:
        raw = json.load(f)
    def pick(key):
        for k, v in raw.items():
            if k.lower() == key.lower(): return v
        return []
    return {"train": pick("train"),
            "val": pick("val") or pick("validation"),
            "test": pick("test")}


def evaluate_logits_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    probs = torch.softmax(logits, dim=1).cpu().numpy()
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


@torch.no_grad()
def evaluate_ckpt(ckpt_path: Path, dataset: str, json_path: Path) -> Dict[str, float]:
    obj = torch.load(ckpt_path.as_posix(), map_location="cpu")
    state = obj.get("model_state_dict") or obj.get("model")
    if state is None:
        raise RuntimeError(f"Bad checkpoint: {ckpt_path}")

    # infer num_classes from fc.weight if possible; else from train labels
    num_classes = None
    for k, v in state.items():
        if k.endswith("fc.weight"):
            num_classes = v.shape[0]
            break

    splits = load_splits(json_path)
    if not splits.get("test"):
        raise FileNotFoundError(f"No 'test' split in {json_path}")
    if num_classes is None:
        labels = {int(it["label"] if isinstance(it, dict) else it[1]) for it in splits["train"]}
        num_classes = len(labels)

    tfm = build_transform()
    ds_test = JsonClsDataset(splits["test"], tfm)
    ld_test = DataLoader(ds_test, batch_size=BATCH_TEST, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

    net = models.resnet101(pretrained=False, num_classes=num_classes).to(DEVICE).eval()
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict missing={len(missing)}, unexpected={len(unexpected)}")

    ce = nn.CrossEntropyLoss()
    all_logits, all_targets = [], []
    total = 0; loss_sum = 0.0
    for batch in ld_test:
        x = batch["image"].to(DEVICE, non_blocking=True)
        y = batch["label"].to(DEVICE, non_blocking=True)
        logits = net(x)
        loss_sum += float(ce(logits, y)) * y.size(0)
        total += y.size(0)
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())
    if total == 0:
        return {"loss": 0.0, "acc": 0.0, "macro_f1": 0.0, "auc_ovo": 0.0, "auc_ovr": 0.0}
    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    m = evaluate_logits_metrics(logits_cat, targets_cat)
    m["loss"] = loss_sum / total
    return m


# --------------------- CSV Utilities ---------------------
def read_done_paths(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        col = None
        for c in df.columns[::-1]:
            if c.lower() == "ckpt_path":
                col = c; break
        if col is None: col = df.columns[-1]
        return set(Path(p).resolve().as_posix() for p in df[col].dropna().astype(str).tolist())
    except Exception:
        done = set()
        with csv_path.open("r", newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
            if header:
                idx = -1
                for i, h in enumerate(header):
                    if h.lower() == "ckpt_path":
                        idx = i; break
                if idx == -1: idx = len(header) - 1
                for row in r:
                    if len(row) > idx:
                        done.add(Path(row[idx]).resolve().as_posix())
        return done


def append_rows(csv_path: Path, rows: List[dict], columns: List[str]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            w.writeheader()
        for r in rows:
            for c in columns:
                r.setdefault(c, "")
            w.writerow(r)


# --------------------- Main ---------------------
def main():
    done = read_done_paths(OUT_CSV)
    print(f"[INFO] Existing entries in CSV: {len(done)} ({OUT_CSV})")

    for ds in DATASETS:
        ds_root = ROOT / ds / PAIR_DIR
        if not ds_root.exists():
            print(f"[WARN] Skip {ds}: directory not found -> {ds_root}")
            continue

        for seed_dir in sorted(p for p in ds_root.glob("seed_*") if p.is_dir()):
            ckpt = seed_dir / CKPT_NAME
            if not ckpt.exists():
                print(f"[INFO] Missing {CKPT_NAME}: {ckpt}")
                continue

            ck_abs = ckpt.resolve().as_posix()
            if ck_abs in done:
                print(f"[SKIP] Already in CSV: {ck_abs}")
                continue

            # choose JSON: prefer merged in seed dir; fallback to global json
            merged_json = seed_dir / f"{ds}_merged_train.json"
            if merged_json.exists():
                json_path = merged_json
            else:
                fallback = JSON_FALLBACK_DIR / f"{ds}.json"
                if not fallback.exists():
                    print(f"[WARN] No JSON found ({merged_json} or {fallback}); skip {ck_abs}")
                    continue
                json_path = fallback

            print(f"[RUN] {ds} | {seed_dir.name} | {ckpt.name} | JSON={json_path}")
            try:
                metrics = evaluate_ckpt(ckpt, ds, json_path)
                row = {
                    "dataset": ds,
                    "pair": PAIR_DIR,
                    "method": PAIR_DIR,
                    "method_pretty": PAIR_DIR,
                    "seed": seed_dir.name.replace("seed_", ""),
                    "ckpt_name": CKPT_NAME,
                    "acc": metrics["acc"],
                    "macro_f1": metrics["macro_f1"],
                    "auc_ovo": metrics["auc_ovo"],
                    "auc_ovr": metrics["auc_ovr"],
                    "ckpt_path": ck_abs,
                }
            except Exception as e:
                print(f"[ERROR] Eval failed: {ck_abs} ({e})")
                row = {
                    "dataset": ds,
                    "pair": PAIR_DIR,
                    "method": PAIR_DIR,
                    "method_pretty": PAIR_DIR,
                    "seed": seed_dir.name.replace("seed_", ""),
                    "ckpt_name": CKPT_NAME,
                    "acc": "", "macro_f1": "", "auc_ovo": "", "auc_ovr": "",
                    "ckpt_path": ck_abs,
                }

            # append immediately (safe for long runs)
            append_rows(OUT_CSV, [row], COLUMNS)

    print(f"[DONE] Appended results to {OUT_CSV}")


if __name__ == "__main__":
    main()
