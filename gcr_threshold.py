#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thresholded-GCR launcher with:
  (a) Early stop when ACC > 0.76  —— 通过自动注入: --override "train.acc_stop=0.76"
  (b) Save a checkpoint for EVERY epoch as epoch_XXX.pth
  (c) Thresholded projection for GCR (cos/dot) without editing main.py or models/gcr.py
"""

from __future__ import annotations
import os
import sys
import argparse
import importlib
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Wrapper for thresholded GCR ablations (+ early-stop + per-epoch ckpt)",
        add_help=False
    )
    p.add_argument("--gcr_tau", type=float, default=0.0,
                   help="Threshold; cos-mode: project if cos(g_kd,g_ce) < tau; dot-mode: dot < tau.")
    p.add_argument("--gcr_mode", type=str, choices=["cos", "dot"], default="cos",
                   help="Conflict metric: 'cos' or 'dot'.")
    return p


@torch.cuda.amp.autocast(enabled=False)  # keep FP32 for algebra
def _backward_with_gcr_threshold(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ce: torch.Tensor,
    kd: torch.Tensor,
    rel: torch.Tensor,
    beta: float = 1.0,
    gamma: float = 1.0,
    step: bool = True,
    zero_grad_each_step: bool = True,
    conflict_counter: Optional[dict] = None,
    grad_clip_norm: Optional[float] = None,
    eps: float = 1e-12,
) -> None:
    mode = os.getenv("GCR_MODE", "cos").strip().lower()
    try:
        tau = float(os.getenv("GCR_TAU", "0.0"))
    except Exception:
        tau = 0.0

    gcr_mod = importlib.import_module("models.gcr")
    _flat   = gcr_mod._flatten_param_grads
    _assign = gcr_mod._assign_grads_from_flat
    _logcos = gcr_mod._maybe_log_cos

    student = model
    params: List[nn.Parameter] = [p for p in student.parameters() if p.requires_grad]

    # REL
    optimizer.zero_grad(set_to_none=True)
    rel_val = float(rel.detach()) if rel is not None else 0.0
    if (rel is not None) and (rel_val != 0.0):
        rel.backward(retain_graph=True)
        g_rel_flat, shapes, device = _flat(params)
    else:
        g_rel_flat, shapes, device = _flat(params)
        if g_rel_flat.numel() > 0:
            g_rel_flat.zero_()

    # CE
    optimizer.zero_grad(set_to_none=True)
    if ce is None:
        raise RuntimeError("CE must not be None for GCR.")
    ce.backward(retain_graph=True)
    g_ce_flat, shapes_ce, _ = _flat(params)
    assert g_ce_flat.numel() == g_rel_flat.numel()

    # KD
    optimizer.zero_grad(set_to_none=True)
    kd.backward(retain_graph=True)
    g_kd_flat, shapes_kd, _ = _flat(params)
    assert g_kd_flat.numel() == g_ce_flat.numel()

    # log cosine before projection
    _logcos(g_kd_flat, g_ce_flat, conflict_counter, eps=eps)

    # thresholded conflict check
    dot = torch.dot(g_kd_flat, g_ce_flat)
    if mode == "dot":
        do_project = (dot.item() < tau)
    else:
        nkd = g_kd_flat.norm().clamp_min(eps)
        nce = g_ce_flat.norm().clamp_min(eps)
        cos = (dot / (nkd * nce)).item()
        do_project = (cos < tau)

    if conflict_counter is not None:
        conflict_counter["total"] = conflict_counter.get("total", 0) + 1
        if do_project:
            conflict_counter["conflict"] = conflict_counter.get("conflict", 0) + 1

    if do_project:
        denom = g_ce_flat.norm().pow(2) + eps
        proj = (dot / denom) * g_ce_flat
        g_kd_flat = g_kd_flat - proj

    g_total = beta * g_ce_flat + gamma * g_kd_flat
    if g_rel_flat.numel() > 0:
        g_total = g_total + g_rel_flat

    _assign(params, g_total, shapes)

    if grad_clip_norm is not None and grad_clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip_norm)

    if step:
        optimizer.step()
        if zero_grad_each_step:
            optimizer.zero_grad(set_to_none=True)


def main() -> None:
    # parse only wrapper flags; leave the rest to main.py
    p = _build_parser()
    known, remaining = p.parse_known_args()

    os.environ["GCR_TAU"]  = str(known.gcr_tau)
    os.environ["GCR_MODE"] = known.gcr_mode

    # patch GCR
    gcr_mod = importlib.import_module("models.gcr")
    gcr_mod.backward_with_gcr = _backward_with_gcr_threshold

    # patch engine: also dump epoch_XXX.pth every epoch
    eng_mod = importlib.import_module("modules.engine")
    _orig_save_last = eng_mod.CheckpointEngine.save_last

    def _save_last_with_epoch_dump(self, *, epoch, model_state_dict, optimizer_state_dict,
                                   scheduler_state_dict, metrics, cfg, filename="last.pth"):
        out = _orig_save_last(self,
                              epoch=epoch,
                              model_state_dict=model_state_dict,
                              optimizer_state_dict=optimizer_state_dict,
                              scheduler_state_dict=scheduler_state_dict,
                              metrics=metrics,
                              cfg=cfg,
                              filename=filename)
        save_dir = Path(getattr(self, "save_dir", "."))
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": scheduler_state_dict,
            "metrics": metrics,
            "cfg": cfg,
        }, save_dir / f"epoch_{int(epoch):03d}.pth")
        return out

    eng_mod.CheckpointEngine.save_last = _save_last_with_epoch_dump

    # inject early-stop via override (NOT --acc_stop)
    # if no override for train.acc_stop, append ours
    has_acc_override = any(
        (arg == "--override" and (i + 1) < len(remaining) and remaining[i + 1].startswith("train.acc_stop="))
        for i, arg in enumerate(remaining)
    )
    if not has_acc_override:
        remaining += ["--override", "train.acc_stop=0.76"]

    # forward remaining args to main.py
    sys.argv = [sys.argv[0]] + remaining
    main_module = importlib.import_module("main")
    if not hasattr(main_module, "main"):
        raise RuntimeError("Expected function main() in main.py but not found.")
    main_module.main()


if __name__ == "__main__":
    main()
