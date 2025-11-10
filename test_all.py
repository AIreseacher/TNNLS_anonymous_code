# test_all.py  (resume + append mode)
from __future__ import annotations
import csv
import inspect
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import pandas as pd

# project deps
from data import build_dataloaders
from data.utils import set_seed
from models.scope import ScopeModel
from main import evaluate_split  # returns loss/acc/macro_f1/auc_ovo/auc_ovr


# -----------------------
# Path helpers
# -----------------------
def find_checkpoints(root: Path, pattern: str = "*.pth") -> List[Path]:
    """
    Find checkpoints recursively under:
      checkpoints/<Dataset>/<Pair>/<Method>/seed_*/<files>.pth
    pattern: default '*.pth' (use 'best.pth' to restrict)
    """
    return sorted(p for p in root.rglob(pattern) if p.is_file())


def parse_meta(ck: Path) -> Dict[str, str]:
    """Parse dataset/pair/method/seed from canonical layout."""
    seed_dir = ck.parent
    method_dir = seed_dir.parent if seed_dir else None
    pair_dir = method_dir.parent if method_dir else None
    dataset_dir = pair_dir.parent if pair_dir else None
    return {
        "dataset": dataset_dir.name if dataset_dir else "",
        "pair": pair_dir.name if pair_dir else "",
        "method": method_dir.name if method_dir else "",
        "seed": (seed_dir.name.replace("seed_", "") if seed_dir else ""),
        "ckpt_path": ck.as_posix(),
        "ckpt_name": ck.name,
    }


# -----------------------
# Pretty method formatting
# -----------------------
def pretty_method(method_name: str) -> str:
    m = re.fullmatch(r"gcr_tau(-)?(\d+)p(\d+)", method_name)
    if m:
        sign = "-" if m.group(1) else "+"
        a = int(m.group(2)); b = int(m.group(3))
        return f"GCR τ={sign}{a}.{b:02d}"
    low = method_name.lower()
    if low == "scope": return "Scope (default)"
    if low in {"pcgrad", "cagrad"}: return method_name.upper()
    if low.startswith("fixed_alpha_"):
        frac = method_name.split("_")[-1]
        if frac.isdigit(): return f"Fixed-Alpha α=0.{int(frac):02d}"
    mapping = {
        "full_maxprob": "Full-MaxProb",
        "nosra_nogcr": "NoSRA-NoGCR",
        "no_gcr": "NoGCR",
        "teacher_only": "Teacher-Only",
        "label_only": "Label-Only",
        "no_ontology": "No-Ontology",
    }
    return mapping.get(low, method_name)


# -----------------------
# Checkpoint loading
# -----------------------
def load_ckpt(ck: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load checkpoint; tolerate 'model' or 'model_state_dict' keys."""
    obj = torch.load(ck.as_posix(), map_location="cpu")
    cfg = obj.get("cfg", {}) or {}
    state = obj.get("model", None) or obj.get("model_state_dict", None)
    if state is None or not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {ck}")
    return cfg, state


def ensure_data_cfg(cfg: Dict[str, Any], meta: Dict[str, str], default_json_dir: Optional[str]) -> None:
    """
    Ensure cfg['data'] has fields for build_dataloaders:
      - dataset (from meta if missing)
      - json_dir (from cfg or --json_dir or ./json)
    """
    cfg.setdefault("data", {}); cfg.setdefault("model", {})
    data = cfg["data"]
    if not data.get("dataset"):
        data["dataset"] = meta["dataset"]
    if not data.get("json_dir"):
        if default_json_dir:
            data["json_dir"] = default_json_dir
        elif Path("json").is_dir():
            data["json_dir"] = "json"
        else:
            raise FileNotFoundError(
                f"json_dir missing for {meta['dataset']} (not in checkpoint). "
                f"Provide --json_dir or create ./json."
            )
    data.setdefault("root", ".")


def _filter_kwargs_for(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in accepted}


def build_model_from_cfg(cfg: Dict[str, Any], state: Dict[str, Any]) -> ScopeModel:
    mcfg = cfg.get("model", {}) or {}
    scfg = cfg.get("scope", {}) or {}
    base_kwargs = dict(
        num_classes=int(mcfg.get("num_classes", 0)),
        backbone_student=mcfg.get("backbone_student", "resnet18"),
        backbone_teacher=mcfg.get("backbone_teacher", "resnet101"),
        T=float(mcfg.get("T", 4.0)),
        teacher_ckpt=mcfg.get("teacher_ckpt") or None,
        pretrained_backbone=bool(mcfg.get("pretrained_backbone", True)),
        dataset_name=cfg.get("data", {}).get("dataset"),
        pretrain_root=mcfg.get("pretrain_dir", "./Pretrain"),
    )
    merged = {**base_kwargs, **(scfg if isinstance(scfg, dict) else {})}
    filtered = _filter_kwargs_for(ScopeModel, merged)

    model = ScopeModel(**filtered)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] state_dict load: missing={len(missing)}, unexpected={len(unexpected)}")
    return model


@torch.no_grad()
def evaluate_ckpt(ck: Path, device: torch.device, default_json_dir: Optional[str]) -> Dict[str, Any]:
    meta = parse_meta(ck)
    cfg, state = load_ckpt(ck)
    ensure_data_cfg(cfg, meta, default_json_dir)

    loaders = build_dataloaders(cfg)
    num_classes = int(cfg["model"]["num_classes"])
    set_seed(int(cfg.get("output", {}).get("seed", 42)))

    model = build_model_from_cfg(cfg, state).to(device).eval()
    crit = nn.CrossEntropyLoss()
    metrics = evaluate_split(model, loaders["test"], device, crit, num_classes)

    return {
        "dataset": meta["dataset"],
        "pair": meta["pair"],
        "method": meta["method"],
        "method_pretty": pretty_method(meta["method"]),
        "seed": meta["seed"],
        "ckpt_name": meta["ckpt_name"],
        "acc": metrics.get("acc"),
        "macro_f1": metrics.get("macro_f1"),
        "auc_ovo": metrics.get("auc_ovo"),
        "auc_ovr": metrics.get("auc_ovr"),
        "ckpt_path": meta["ckpt_path"],
    }


# -----------------------
# CSV helpers (resume)
# -----------------------
def load_done_paths(csv_path: Path) -> set[str]:
    """
    Read existing eval.csv and return a set of absolute ckpt paths already evaluated.
    Prefer a column named 'ckpt_path'; otherwise use the last column.
    """
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    col = None
    for c in df.columns[::-1]:
        if c.lower() == "ckpt_path":
            col = c; break
    if col is None:
        col = df.columns[-1]
    # normalize to absolute posix for robust matching
    return set(Path(p).resolve().as_posix() for p in df[col].dropna().astype(str).tolist())


def append_rows(csv_path: Path, rows: List[Dict[str, Any]], header: List[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        for r in rows:
            # fill missing keys
            for h in header:
                r.setdefault(h, "")
            w.writerow(r)


# -----------------------
# Main
# -----------------------
def main():
    import argparse
    ap = argparse.ArgumentParser("Resume eval: only run for pth not yet in CSV; append results")
    ap.add_argument("--ckpt_root", type=str, default="checkpoints",
                    help="Root: checkpoints/<Dataset>/<Pair>/<Method>/seed_*/{*.pth|best.pth}")
    ap.add_argument("--pattern", type=str, default="*.pth",
                    help="Glob pattern for files to evaluate (e.g., '*.pth' or 'best.pth').")
    ap.add_argument("--out_csv", type=str, default="results/eval.csv",
                    help="Summary CSV to append to (created if missing).")
    ap.add_argument("--json_dir", type=str, default=None,
                    help="Fallback json_dir if checkpoint cfg misses it (e.g., ./json)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to run evaluation")
    args = ap.parse_args()

    root = Path(args.ckpt_root)
    out_csv = Path(args.out_csv)
    device = torch.device(args.device)

    all_ckpts = find_checkpoints(root, args.pattern)
    if not all_ckpts:
        print(f"[WARN] No checkpoints found by pattern '{args.pattern}' under {root.resolve()}")
        return

    done = load_done_paths(out_csv)
    print(f"[INFO] Found {len(all_ckpts)} ckpt(s); already done in CSV: {len(done)}")

    # filter: only new ckpts
    todo: List[Path] = []
    for ck in all_ckpts:
        if ck.resolve().as_posix() not in done:
            todo.append(ck)

    if not todo:
        print("[INFO] Nothing new to evaluate. Exit.")
        return

    print(f"[INFO] To evaluate: {len(todo)}")
    header = ["dataset", "pair", "method", "method_pretty", "seed",
              "ckpt_name", "acc", "macro_f1", "auc_ovo", "auc_ovr", "ckpt_path"]

    # evaluate and append progressively (so即使中断也能续跑)
    for i, ck in enumerate(sorted(todo), 1):
        try:
            row = evaluate_ckpt(ck, device, args.json_dir)
            print(f"[{i:03d}/{len(todo)}] {row['dataset']} | {row['pair']} | {row['method_pretty']} "
                  f"| seed={row['seed']} | {row['ckpt_name']} -> "
                  f"ACC={row['acc']:.4f} F1={row['macro_f1']:.4f} OVO={row['auc_ovo']:.4f} OVR={row['auc_ovr']:.4f}")
        except Exception as e:
            meta = parse_meta(ck)
            row = {
                "dataset": meta["dataset"], "pair": meta["pair"],
                "method": meta["method"], "method_pretty": pretty_method(meta["method"]),
                "seed": meta["seed"],
                "ckpt_name": meta["ckpt_name"],
                "acc": None, "macro_f1": None, "auc_ovo": None, "auc_ovr": None,
                "ckpt_path": meta["ckpt_path"],
            }
            print(f"[ERROR] Failed on {ck}: {e}")

        # append immediately (safer for long runs)
        append_rows(out_csv, [row], header)

    print(f"[DONE] Appended new results to {out_csv.resolve()}")


if __name__ == "__main__":
    main()
