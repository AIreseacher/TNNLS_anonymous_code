# train_scope.py
from __future__ import annotations
import argparse, os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import build_dataloaders
from data.utils import load_yaml, merge_override, set_seed

# use your uploaded modules (placed under models/)
from models.scope import ScopeModel                              # :contentReference[oaicite:7]{index=7}
from models.gcr import backward_with_gcr                         # :contentReference[oaicite:8]{index=8}


def kd_loss(student_logits: torch.Tensor, teacher_logits_T: torch.Tensor, T: float) -> torch.Tensor:
    """Standard logit-based KD (KL with temperature)."""
    # student: log_softmax; teacher: softmax  (both already /T inside ScopeModel output)
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    p_t = F.softmax(teacher_logits_T, dim=-1)  # teacher_logits_T is already divided by T in ScopeModel
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


def parse_args():
    ap = argparse.ArgumentParser("SRA + KD + GCR training (YAML-driven)")
    ap.add_argument("--config", type=str, required=True, help="e.g., ./configs/Brain.yaml")
    ap.add_argument("--json_dir", type=str, default=None, help="override data.json_dir")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--override", action="append", default=None,
                    help="KEY=VALUE overrides. e.g. train.epochs=2 model.backbone_teacher=resnet101")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # 1) load cfg
    cfg = load_yaml(args.config)
    cfg.setdefault("data", {}); cfg.setdefault("model", {}); cfg.setdefault("train", {}); cfg.setdefault("scope", {})
    if args.json_dir is not None:
        cfg["data"]["json_dir"] = args.json_dir
    cfg = merge_override(cfg, args.override)

    # 2) dataloaders (also infers num_classes)
    loaders = build_dataloaders(cfg)
    num_classes = int(cfg["model"]["num_classes"])
    dataset_name = cfg["data"]["dataset"]

    # 3) read model/train/scope hyper-params
    mcfg = cfg["model"]
    tcfg = cfg["train"]
    scfg = cfg["scope"]

    backbone_student = mcfg.get("backbone_student", "resnet18")
    backbone_teacher = mcfg.get("backbone_teacher", "resnet101")
    teacher_ckpt     = mcfg.get("teacher_ckpt", "")
    use_pretrained   = bool(mcfg.get("pretrained_backbone", True))
    T                = float(mcfg.get("T", 4.0))

    epochs           = int(tcfg.get("epochs", 100))
    lr               = float(tcfg.get("base_lr", 1e-3) or tcfg.get("lr", 1e-3))
    weight_decay     = float(tcfg.get("weight_decay", 1e-4))

    beta             = float(scfg.get("beta", 1.0))    # CE weight
    gamma            = float(scfg.get("gamma", 1.0))   # KD weight
    alpha_mode       = scfg.get("alpha_mode", "entropy")
    with_relations   = bool(scfg.get("with_relations", True))
    # You can also add tau/prior if needed later.

    save_dir = Path(cfg.get("output", {}).get("save_dir", f"./runs/{dataset_name}_scope"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # 4) build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScopeModel(
        num_classes=num_classes,
        backbone_student=backbone_student,
        backbone_teacher=backbone_teacher,
        T=T,
        teacher_ckpt=teacher_ckpt if teacher_ckpt else None,
        pretrained_backbone=use_pretrained,
        dataset_name=dataset_name,
        pretrain_root=cfg.get("model", {}).get("pretrain_dir", "./Pretrain"),
    ).to(device)  # :contentReference[oaicite:9]{index=9}

    optimizer = torch.optim.AdamW(model.student.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_ce = nn.CrossEntropyLoss()

    print("\n=== Scope Training Summary ===")
    print(f"Dataset       : {dataset_name}")
    print(f"Teacher→Student: {backbone_teacher} → {backbone_student}")
    print(f"T (KD temp)   : {T} | beta (CE)={beta} | gamma (KD)={gamma}")
    print(f"Pretrained    : {use_pretrained} | Teacher ckpt: {teacher_ckpt or '(auto-match)'}")
    print(f"Device        : {device} | Epochs: {epochs} | LR: {lr} | WD: {weight_decay}")
    print(f"Save dir      : {save_dir.as_posix()}")

    best_val = -1.0
    conflict_counter: Dict[str, int] = {}

    # 5) training loop
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss_ce = tr_loss_kd = tr_loss_rel = 0.0
        n_train = 0

        for batch in loaders["train"]:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            # forward: student + teacher + SRA relations (no prior vector for now)
            out = model(x, labels=y, h_vec=None, return_feat=False, with_relations=with_relations)  # :contentReference[oaicite:10]{index=10}

            # CE
            loss_ce = criterion_ce(out["student_logits"], y)

            # KD
            loss_kd = kd_loss(out["student_logits"], out["teacher_logits_T"], T)

            # REL (SRA)
            # R_s keeps grad (student), R_t'/R_y are detached constants from build_relations
            if with_relations:
                R_s = out["R_s"]; R_t_prime = out["R_t_prime"]                          # :contentReference[oaicite:11]{index=11}
                loss_rel = F.mse_loss(R_s, R_t_prime)                                   # a simple L2 relation loss
            else:
                loss_rel = torch.tensor(0.0, device=device, requires_grad=True)

            # GCR backward/step (projection if <g_kd, g_ce> < 0)
            backward_with_gcr(
                model=model.student,
                optimizer=optimizer,
                ce=loss_ce,
                kd=loss_kd,
                rel=loss_rel,
                beta=beta,
                gamma=gamma,
                conflict_counter=conflict_counter,
            )  # :contentReference[oaicite:12]{index=12}

            bs = y.size(0)
            tr_loss_ce  += float(loss_ce.detach())  * bs
            tr_loss_kd  += float(loss_kd.detach())  * bs
            tr_loss_rel += float(loss_rel.detach()) * bs
            n_train     += bs

        tr_loss_ce  /= max(n_train, 1)
        tr_loss_kd  /= max(n_train, 1)
        tr_loss_rel /= max(n_train, 1)

        # validation (student only)
        model.eval()
        correct = total = 0
        val_loss = 0.0
        with torch.no_grad():
            for batch in loaders["validation"]:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                s_logits = model(x)["student_logits"]
                val_loss += float(criterion_ce(s_logits, y)) * y.size(0)
                pred = s_logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total   += y.size(0)
        val_loss /= max(total, 1)
        val_acc = correct / max(total, 1)

        # logging
        conf_r = 0.0
        if conflict_counter.get("total", 0) > 0:
            conf_r = conflict_counter.get("conflict", 0) / conflict_counter["total"]
        print(f"[Epoch {epoch:03d}] "
              f"train: CE={tr_loss_ce:.4f} KD={tr_loss_kd:.4f} REL={tr_loss_rel:.4f} "
              f"| val: loss={val_loss:.4f} acc={val_acc*100:6.2f}% "
              f"| GCR conflicts ratio={conf_r*100:5.1f}%")

        # save best
        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {"model": model.state_dict(), "val_acc": best_val, "cfg": cfg, "epoch": epoch},
                (save_dir / f"best_{backbone_teacher}_to_{backbone_student}.pth").as_posix(),
            )

    # test
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loaders["test"]:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            s_logits = model(x)["student_logits"]
            pred = s_logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    print(f"[Test] acc={correct/max(total,1)*100:6.2f}%")

if __name__ == "__main__":
    main()
