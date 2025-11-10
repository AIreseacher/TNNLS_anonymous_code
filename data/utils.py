# data/utils.py
from __future__ import annotations
import os, re, json, yaml, random, warnings
from typing import Any, Dict, Iterable, Optional
import numpy as np
import torch

# ---------------- Reproducibility ----------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------- YAML / JSON IO ----------------
def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[load_yaml] File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def read_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[read_json] File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ---------------- Overrides ----------------
_BOOL_TRUE = {"true","yes","y","1","on"}
_BOOL_FALSE= {"false","no","n","0","off"}

def _parse_scalar(text: str) -> Any:
    s = text.strip(); low = s.lower()
    if low in _BOOL_TRUE: return True
    if low in _BOOL_FALSE: return False
    if re.fullmatch(r"[+-]?\d+", s):
        try: return int(s)
        except: pass
    if re.fullmatch(r"[+-]?\d+(\.\d+)?([eE][+-]?\d+)?", s):
        try: return float(s)
        except: pass
    try: return json.loads(s)
    except: return s

def merge_override(cfg: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    if not overrides: return cfg
    for item in overrides:
        if not item or "=" not in item:
            warnings.warn(f"[merge_override] skip invalid override: {item}"); continue
        key, val_str = item.split("=", 1)
        keys = key.strip().split(".")
        val = _parse_scalar(val_str)
        d = cfg
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict): d[k] = {}
            d = d[k]
        d[keys[-1]] = val
    return cfg

# ---------------- Path helpers ----------------
def resolve_json_path(dataset: str, json_dir: str) -> str:
    if not os.path.isdir(json_dir):
        raise FileNotFoundError(f"[resolve_json_path] json_dir not found: {json_dir}")
    cand = os.path.join(json_dir, f"{dataset}.json")
    if os.path.isfile(cand): return os.path.abspath(cand)
    lower_target = f"{dataset}.json".lower()
    for fn in os.listdir(json_dir):
        if fn.lower() == lower_target:
            return os.path.abspath(os.path.join(json_dir, fn))
    raise FileNotFoundError(f"[resolve_json_path] {dataset}.json not found under {json_dir}")

def to_abs_path(root: str, maybe_rel: str) -> str:
    if not maybe_rel: return maybe_rel
    p = maybe_rel.replace("\\", "/")
    if os.path.isabs(p): return p
    return os.path.abspath(os.path.join(root, p))
