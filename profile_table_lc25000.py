
from __future__ import annotations
import csv
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import re

import torch
import torch.nn as nn
from thop import profile as thop_profile

# 复用工程内模块
from data import build_dataloaders
from data.utils import set_seed
from models.scope import ScopeModel
from main import evaluate_split  # 评测 AUC/ACC/F1/OVO/OVR

# ---------------------- 搜索 & 元信息 ---------------------- #
def find_all_best(search_root: Path) -> List[Path]:
    """
    递归从 search_root 查找所有 seed_*/best*.pt(pth)（大小写不敏感）
    不再强制固定层级或方法名大小写（scope/SCOPE 均可）。
    """
    all_ckpts: List[Path] = []
    for p in search_root.rglob("*"):
        if p.is_file():
            nm = p.name.lower()
            if nm.startswith("best") and (nm.endswith(".pt") or nm.endswith(".pth")):
                if re.search(r"(?:^|/|\\)seed_\d+(?:/|\\)", p.as_posix(), flags=re.IGNORECASE):
                    all_ckpts.append(p)
    return sorted(all_ckpts)

def parse_meta(ck: Path) -> Dict[str, str]:
    """
    从路径解析 dataset/pair/method/seed，按常见结构：
    <dataset>/<pair>/<method>/seed_XXX/best*.pt
    若层级不足，尽力解析。
    """
    parts = ck.parts
    # 通常 .../checkpoints/<dataset>/<pair>/<method>/seed_xxx/best.pth
    #                  -5        -4        -3        -2        -1
    dataset = parts[-5] if len(parts) >= 5 else "UnknownDS"
    pair    = parts[-4] if len(parts) >= 4 else "UnknownPair"
    method  = parts[-3] if len(parts) >= 3 else "UnknownMethod"
    seed    = "unknown"
    for prt in parts:
        if prt.lower().startswith("seed_"):
            seed = prt.split("_", 1)[-1]
            break
    return {
        "dataset": dataset,
        "pair": pair,
        "method": method,
        "seed": seed,
        "ckpt_path": ck.as_posix(),
    }

# ---------------------- 模型重建 ---------------------- #
def load_ckpt(ck: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    obj = torch.load(ck.as_posix(), map_location="cpu")
    cfg = obj.get("cfg", {}) or {}
    state = obj.get("model", None) or obj.get("model_state_dict", None)
    if state is None or not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {ck}")
    return cfg, state

def ensure_data_cfg(cfg: Dict[str, Any], meta: Dict[str, str], default_json_dir: Optional[str]) -> None:
    cfg.setdefault("data", {}); cfg.setdefault("model", {}); cfg.setdefault("scope", {})
    if not cfg["data"].get("dataset"):
        cfg["data"]["dataset"] = meta["dataset"]
    if not cfg["data"].get("json_dir"):
        if default_json_dir:
            cfg["data"]["json_dir"] = default_json_dir
        elif Path("json").is_dir():
            cfg["data"]["json_dir"] = "json"
        else:
            raise FileNotFoundError(
                f"json_dir missing for {meta['dataset']}; provide --json_dir or create ./json."
            )
    cfg["data"].setdefault("root", ".")

def _filter_kwargs_for(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    import inspect
    sig = inspect.signature(cls.__init__)
    keep = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in keep}

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

# ---------------------- 统计量 ---------------------- #
def count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

def compute_flops_g(model: nn.Module, sample_x: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        flops, _ = thop_profile(model, inputs=(sample_x,), verbose=False)
    return float(flops) / 1e9

@torch.no_grad()
def measure_infer_time_per_image(model: nn.Module, device: torch.device,
                                 sample_x: torch.Tensor, warmup: int = 10, runs: int = 50) -> float:
    model = model.to(device).eval()
    x = sample_x.to(device, non_blocking=True)
    for _ in range(warmup):
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / runs

# ---------------------- 单个 ckpt 评测 ---------------------- #
def evaluate_and_profile_one(ck: Path, device_gpu: torch.device,
                             default_json_dir: Optional[str]) -> Dict[str, Any]:
    meta = parse_meta(ck)
    cfg, state = load_ckpt(ck)
    ensure_data_cfg(cfg, meta, default_json_dir)

    loaders = build_dataloaders(cfg)
    num_classes = int(cfg["model"]["num_classes"])

    set_seed(int(cfg.get("output", {}).get("seed", 42)))
    base_model = build_model_from_cfg(cfg, state)

    sample = next(iter(loaders["test"]))
    sample_x = sample["image"][0:1].contiguous()

    params_m = count_params_m(base_model)
    flops_g = compute_flops_g(base_model, sample_x)

    gpu_device = device_gpu if torch.cuda.is_available() else torch.device("cpu")
    gpu_time_s = measure_infer_time_per_image(base_model, gpu_device, sample_x)
    cpu_time_s = measure_infer_time_per_image(base_model, torch.device("cpu"), sample_x)

    crit = nn.CrossEntropyLoss()
    metrics = evaluate_split(base_model.to(gpu_device), loaders["test"], gpu_device, crit, num_classes)
    auc_ovr = metrics.get("auc_ovr")

    return {
        "dataset": meta["dataset"],
        "pair": meta["pair"],
        "method": meta["method"],
        "seed": meta["seed"],
        "gpu_time_s": gpu_time_s,
        "cpu_time_s": cpu_time_s,
        "params_m": params_m,
        "flops_g": flops_g,
        "auc_ovr": auc_ovr,
        "ckpt_path": meta["ckpt_path"],
    }

# ---------------------- 入口 ---------------------- #
def main():
    import argparse
    ap = argparse.ArgumentParser("Profile GPU/CPU time, Params, FLOPs, and AUC-OVR on LUNG")
    ap.add_argument("--ckpt_root", type=str, default="checkpoints",
                    help="Root dir: checkpoints/<Dataset>/<Pair>/<Method>/seed_*/best*.pt(h)")
    ap.add_argument("--dataset", type=str, default="LUNG",
                    help="Dataset folder name under ckpt_root (default: LUNG)")
    ap.add_argument("--out_csv", type=str, default="results/table_iv_lc25000.csv",
                    help="Output CSV path")
    ap.add_argument("--json_dir", type=str, default=None,
                    help="Fallback json_dir if checkpoints miss it (e.g., ./json)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device for evaluation (GPU timing will use this if cuda)")
    args = ap.parse_args()

    search_root = Path(args.ckpt_root) / args.dataset
    if not search_root.exists():
        print(f"[WARN] Search root not exists: {search_root.resolve()}")
        return

    ckpts = find_all_best(search_root)
    if not ckpts:
        print(f"[WARN] No checkpoints found under {search_root.resolve()}")
        return

    # ✅ 只保留 ResNet101_ResNet18 这一组
    ckpts = [p for p in ckpts if "ResNet101_ResNet18" in str(p)]
    if not ckpts:
        print(f"[WARN] No checkpoints for ResNet101_ResNet18 under {search_root.resolve()}")
        return

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    device_gpu = torch.device(args.device)

    rows: List[Dict[str, Any]] = []
    print(f"[INFO] Found {len(ckpts)} checkpoints under {search_root.resolve()}")
    for i, ck in enumerate(ckpts, 1):
        try:
            row = evaluate_and_profile_one(ck, device_gpu, args.json_dir)
            print(f"[{i:03d}/{len(ckpts)}] {row['pair']} | {row['method']} | seed={row['seed']} "
                  f"-> GPU={row['gpu_time_s']:.3f}s CPU={row['cpu_time_s']:.2f}s "
                  f"Params={row['params_m']:.2f}M FLOPs={row['flops_g']:.2f}G AUC-OVR={row['auc_ovr']:.2%}")
            rows.append(row)
        except Exception as e:
            meta = parse_meta(ck)
            print(f"[ERROR] {ck}: {e}")
            rows.append({
                "dataset": meta["dataset"], "pair": meta["pair"], "method": meta["method"], "seed": meta["seed"],
                "gpu_time_s": None, "cpu_time_s": None, "params_m": None, "flops_g": None, "auc_ovr": None,
                "ckpt_path": meta["ckpt_path"],
            })

    header = ["dataset", "pair", "method", "seed",
              "gpu_time_s", "cpu_time_s", "params_m", "flops_g", "auc_ovr", "ckpt_path"]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Saved CSV to {out.resolve()}")

if __name__ == "__main__":
    main()
