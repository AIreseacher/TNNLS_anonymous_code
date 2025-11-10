# main.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import build_dataloaders
from data.utils import load_yaml, merge_override, set_seed

from models.scope import ScopeModel
from models.gcr import backward_with_gcr

from modules.metrics import evaluate_metrics, composite_score
from modules.engine import CheckpointEngine


def kd_loss(student_logits: torch.Tensor, teacher_logits_T: torch.Tensor, T: float) -> torch.Tensor:
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    p_t = F.softmax(teacher_logits_T, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


@torch.no_grad()
def evaluate_split(
    model: ScopeModel, loader, device: torch.device, criterion_ce: nn.Module, num_classes: int
) -> Dict[str, float | None]:
    model.eval()
    loss_sum, n = 0.0, 0
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits = model(x)["student_logits"]
        loss_sum += float(criterion_ce(logits, y)) * y.size(0)
        n += y.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())
    loss = loss_sum / max(n, 1)
    if n == 0:
        return {"loss": 0.0, "acc": 0.0, "macro_f1": 0.0, "auc_ovo": 0.0, "auc_ovr": 0.0}
    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    m = evaluate_metrics(logits_cat, targets_cat, num_classes=num_classes)
    return {"loss": loss, "acc": m["acc"], "macro_f1": m["macro_f1"], "auc_ovo": m["auc_ovo"], "auc_ovr": m["auc_ovr"]}


def parse_args():
    ap = argparse.ArgumentParser("SRA + KD + GCR training (YAML-driven) with logging modes")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML, e.g. ./configs/Brain.yaml")
    ap.add_argument("--json_dir", type=str, default=None, help="Override data.json_dir")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for this run")
    ap.add_argument("--log_mode", type=str, default=None, choices=["r", "w", "b"],
                    help="Override output.log_mode from YAML. r/w/b")
    ap.add_argument("--override", action="append", default=None,
                    help="KEY=VALUE overrides. e.g. train.epochs=2 model.backbone_teacher=resnet101")
    # ---- Early stop threshold by ACC (validation) ----
    ap.add_argument("--acc_stop", type=float, default=None,
                    help="Early stop if validation ACC >= this value (e.g., 0.9). If None, use YAML train.acc_stop or default 0.90.")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load config
    cfg = load_yaml(args.config)
    cfg.setdefault("data", {}); cfg.setdefault("model", {}); cfg.setdefault("train", {}); cfg.setdefault("scope", {}); cfg.setdefault("output", {})
    if args.json_dir is not None:
        cfg["data"]["json_dir"] = args.json_dir
    cfg = merge_override(cfg, args.override)

    # Resolve log_mode: CLI > YAML > default "r"
    log_mode = args.log_mode if args.log_mode is not None else cfg["output"].get("log_mode", "r")
    assert log_mode in ("r", "w", "b"), "log_mode must be one of: r, w, b"

    # Data (infers num_classes into cfg['model'])
    loaders = build_dataloaders(cfg)
    dataset_name: str = cfg["data"]["dataset"]
    num_classes: int = int(cfg["model"]["num_classes"])

    # Hyper-params
    mcfg, tcfg, scfg = cfg["model"], cfg["train"], cfg["scope"]
    backbone_student = mcfg.get("backbone_student", "resnet18")
    backbone_teacher = mcfg.get("backbone_teacher", "resnet101")
    teacher_ckpt     = mcfg.get("teacher_ckpt", "")
    use_pretrained   = bool(mcfg.get("pretrained_backbone", True))
    T                = float(mcfg.get("T", 4.0))
    epochs           = int(tcfg.get("epochs", 100))
    lr               = float(tcfg.get("base_lr", tcfg.get("lr", 1e-3)))
    weight_decay     = float(tcfg.get("weight_decay", 1e-4))
    beta             = float(scfg.get("beta", 1.0))
    gamma            = float(scfg.get("gamma", 1.0))
    with_relations   = bool(scfg.get("with_relations", True))

    # Early-stop threshold (validation ACC)
    acc_stop = args.acc_stop
    if acc_stop is None:
        acc_stop = float(tcfg.get("acc_stop", 0.76))  # default 0.90

    # Save dir: if YAML didn't set output.save_dir, auto-construct
    teacher_student = f"{backbone_teacher}_to_{backbone_student}"
    default_save_dir = Path("runs") / dataset_name / teacher_student / f"seed_{args.seed}"
    save_dir = Path(cfg["output"].get("save_dir", default_save_dir.as_posix()))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Model / optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScopeModel(
        num_classes=num_classes,
        backbone_student=backbone_student,
        backbone_teacher=backbone_teacher,
        T=T,
        teacher_ckpt=teacher_ckpt if teacher_ckpt else None,
        pretrained_backbone=use_pretrained,
        dataset_name=dataset_name,
        pretrain_root=mcfg.get("pretrain_dir", "./Pretrain"),
    ).to(device)
    optimizer = torch.optim.AdamW(model.student.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_ce = nn.CrossEntropyLoss()

    engine = CheckpointEngine(save_dir=save_dir.as_posix(), monitor="composite")

    print("\n=== Scope Training Summary ===")
    print(f"Dataset         : {dataset_name}")
    print(f"Teacher_Student : {backbone_teacher} $\\righrarrow$ {backbone_student}")
    print(f"T (KD temp)     : {T} | beta (CE)={beta} | gamma (KD)={gamma}")
    print(f"Pretrained      : {use_pretrained} | Teacher ckpt: {teacher_ckpt or '(auto-match)'}")
    print(f"Device          : {device} | Epochs: {epochs} | LR: {lr} | WD: {weight_decay}")
    print(f"Seed            : {args.seed}")
    print(f"Log mode        : {log_mode}  (r=train+val, w=train+val+test, b=both)")
    print(f"Save dir        : {save_dir.as_posix()}")
    print(f"Early Stop ACC  : {acc_stop:.4f} (validation)")

    conflict_counter: Dict[str, int] = {}

    for epoch in range(1, epochs + 1):
        # ---------------- TRAIN ----------------
        model.train()
        tr_loss_ce = tr_loss_kd = tr_loss_rel = 0.0
        seen = 0
        for batch in loaders["train"]:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            out = model(x, labels=y, h_vec=None, return_feat=False, with_relations=with_relations)
            loss_ce = criterion_ce(out["student_logits"], y)
            loss_kd = kd_loss(out["student_logits"], out["teacher_logits_T"], T)
            loss_rel = F.mse_loss(out["R_s"], out["R_t_prime"]) if with_relations else torch.zeros((), device=device)

            backward_with_gcr(model=model.student, optimizer=optimizer,
                              ce=loss_ce, kd=loss_kd, rel=loss_rel,
                              beta=beta, gamma=gamma, conflict_counter=conflict_counter)

            bs = y.size(0)
            tr_loss_ce  += float(loss_ce.detach())  * bs
            tr_loss_kd  += float(loss_kd.detach())  * bs
            tr_loss_rel += float(loss_rel.detach()) * bs
            seen += bs

        tr_loss_ce  /= max(seen, 1)
        tr_loss_kd  /= max(seen, 1)
        tr_loss_rel /= max(seen, 1)

        # ---------------- EVAL ----------------
        tr_report = evaluate_split(model, loaders["train"], device, criterion_ce, num_classes)
        va_report = evaluate_split(model, loaders["validation"], device, criterion_ce, num_classes)
        comp = composite_score({
            "acc": va_report["acc"], "macro_f1": va_report["macro_f1"],
            "auc_ovr": va_report["auc_ovr"], "auc_ovo": va_report["auc_ovo"],
        })
        te_report: Optional[Dict[str, float | None]] = None
        if log_mode in ("w", "b"):
            te_report = evaluate_split(model, loaders["test"], device, criterion_ce, num_classes)

        conf_ratio = (conflict_counter.get("conflict", 0) / max(1, conflict_counter.get("total", 0))) if conflict_counter else 0.0

        # ---------------- LOG ----------------
        r_line = (f"[Epoch {epoch:03d}][R] "
                  f"train: loss={tr_report['loss']:.4f}, acc={tr_report['acc']:.4f}, "
                  f"f1={tr_report['macro_f1'] if tr_report['macro_f1'] is not None else 'NA'}, "
                  f"auc_ovo={tr_report['auc_ovo'] if tr_report['auc_ovo'] is not None else 'NA'}, "
                  f"auc_ovr={tr_report['auc_ovr'] if tr_report['auc_ovr'] is not None else 'NA'} | "
                  f"val: loss={va_report['loss']:.4f}, acc={va_report['acc']:.4f}, "
                  f"f1={va_report['macro_f1'] if va_report['macro_f1'] is not None else 'NA'}, "
                  f"auc_ovo={va_report['auc_ovo'] if va_report['auc_ovo'] is not None else 'NA'}, "
                  f"auc_ovr={va_report['auc_ovr'] if va_report['auc_ovr'] is not None else 'NA'} | "
                  f"composite={comp:.4f} | GCR={conf_ratio*100:5.1f}%")

        w_line = None
        if te_report is not None:
            w_line = (f"[Epoch {epoch:03d}][W] "
                      f"train: loss={tr_report['loss']:.4f}, acc={tr_report['acc']:.4f}, "
                      f"f1={tr_report['macro_f1'] if tr_report['macro_f1'] is not None else 'NA'}, "
                      f"auc_ovo={tr_report['auc_ovo'] if tr_report['auc_ovo'] is not None else 'NA'}, "
                      f"auc_ovr={tr_report['auc_ovr'] if tr_report['auc_ovr'] is not None else 'NA'} | "
                      f"val: loss={va_report['loss']:.4f}, acc={va_report['acc']:.4f}, "
                      f"f1={va_report['macro_f1'] if va_report['macro_f1'] is not None else 'NA'}, "
                      f"auc_ovo={va_report['auc_ovo'] if va_report['auc_ovo'] is not None else 'NA'}, "
                      f"auc_ovr={va_report['auc_ovr'] if va_report['auc_ovr'] is not None else 'NA'} | "
                      f"test: loss={te_report['loss']:.4f}, acc={te_report['acc']:.4f}, "
                      f"f1={te_report['macro_f1'] if te_report['macro_f1'] is not None else 'NA'}, "
                      f"auc_ovo={te_report['auc_ovo'] if te_report['auc_ovo'] is not None else 'NA'}, "
                      f"auc_ovr={te_report['auc_ovr'] if te_report['auc_ovr'] is not None else 'NA'})")

        r_log_path = save_dir / "r_log.log"
        w_log_path = save_dir / "w_log.log"

        if log_mode == "r":
            print(r_line)
            with open(r_log_path, "a") as f:
                f.write(r_line + "\n")
        elif log_mode == "w":
            if w_line:
                print(w_line)
                with open(w_log_path, "a") as f:
                    f.write(w_line + "\n")
        else:  # "b"
            with open(r_log_path, "a") as f:
                f.write(r_line + "\n")
            if w_line:
                print(w_line)
                with open(w_log_path, "a") as f:
                    f.write(w_line + "\n")

        # ---------------- CHECKPOINTS ----------------
        # 1) 仍然保存 last.pth （覆盖式）
        engine.save_last(
            epoch=epoch,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=None,
            metrics={"val": va_report, "train": tr_report, "test": te_report},
            cfg=cfg,
            filename="last.pth",
        )
        # 2) 如更好则保存 best.pth（由 engine 监控）
        saved = engine.save_if_best(
            epoch=epoch,
            metrics={"acc": va_report["acc"], "macro_f1": va_report["macro_f1"],
                     "auc_ovr": va_report["auc_ovr"], "auc_ovo": va_report["auc_ovo"]},
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=None,
            cfg=cfg,
        )
        if saved:
            print(f" New best @ epoch {epoch}: {saved.as_posix()} (score={engine.best_score:.4f})")

        # 3) 追加：为该 epoch 单独落盘（epoch_XXX.pth）
        epoch_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "metrics": {"val": va_report, "train": tr_report, "test": te_report},
            "cfg": cfg,
        }
        torch.save(epoch_ckpt, (save_dir / f"epoch_{epoch:03d}.pth"))

        # ---------------- EARLY-STOP ----------------
        if va_report["acc"] is not None and va_report["acc"] >= acc_stop:
            print(f"[EARLY-STOP] Validation ACC {va_report['acc']:.4f} >= {acc_stop:.4f} at epoch {epoch}. Stop training.")
            break

    # Final test (always print once)
    final_test = evaluate_split(model, loaders["test"], device, criterion_ce, num_classes)
    print(f"[Final-Test] loss={final_test['loss']:.4f}, acc={final_test['acc']:.4f}, "
          f"f1={final_test['macro_f1'] if final_test['macro_f1'] is not None else 'NA'}, "
          f"auc_ovo={final_test['auc_ovo'] if final_test['auc_ovo'] is not None else 'NA'}, "
          f"auc_ovr={final_test['auc_ovr'] if final_test['auc_ovr'] is not None else 'NA'}")


if __name__ == "__main__":
    main()
