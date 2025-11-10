# modules/metrics.py
from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from sklearn.metrics import f1_score, roc_auc_score
    _SK_OK = True
except Exception:
    _SK_OK = False


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy in [0,1]."""
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        return float((preds == targets).float().mean().item())


def macro_f1_score(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Optional[float]:
    """Macro F1 in [0,1]. Returns None if sklearn not available."""
    if not _SK_OK:
        return None
    with torch.no_grad():
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        y = targets.detach().cpu().numpy()
    try:
        return float(f1_score(y, preds, average="macro", labels=list(range(num_classes))))
    except Exception:
        return None


def _probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits to probabilities (softmax or sigmoid)."""
    if logits.ndim != 2:
        raise ValueError("logits must be 2D [B, C]")
    C = logits.size(1)
    if C == 1:
        # Binary case with a single logit: use sigmoid and stack p0=1-p, p1=p
        p1 = torch.sigmoid(logits)
        probs = torch.cat([1.0 - p1, p1], dim=1)
    else:
        probs = F.softmax(logits, dim=1)
    return probs


def auc_scores(
    logits: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute AUC-OVR and AUC-OVO in [0,1] for multi-class (or binary) problems.
    Returns (auc_ovr, auc_ovo). If sklearn is not available or computation fails,
    the corresponding value is None.
    """
    if not _SK_OK:
        return None, None

    with torch.no_grad():
        probs = _probs_from_logits(logits).detach().cpu().numpy()
        y = targets.detach().cpu().numpy()

    try:
        if num_classes <= 2:
            # Binary AUC is the same for OVR/OVO; use positive class column (index 1)
            p_pos = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            auc_bin = roc_auc_score(y, p_pos)
            return float(auc_bin), float(auc_bin)
        else:
            auc_ovr = roc_auc_score(y, probs, multi_class="ovr", labels=list(range(num_classes)))
            auc_ovo = roc_auc_score(y, probs, multi_class="ovo", labels=list(range(num_classes)))
            return float(auc_ovr), float(auc_ovo)
    except Exception:
        return None, None


def evaluate_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> Dict[str, Optional[float]]:
    """
    Compute the 4 validation metrics:
      - acc        : top-1 accuracy
      - macro_f1   : macro-averaged F1
      - auc_ovr    : one-vs-rest AUC
      - auc_ovo    : one-vs-one  AUC
    Any metric that cannot be computed will be set to None.
    """
    acc = accuracy_top1(logits, targets)
    f1  = macro_f1_score(logits, targets, num_classes)
    auc_ovr, auc_ovo = auc_scores(logits, targets, num_classes)
    return {"acc": acc, "macro_f1": f1, "auc_ovr": auc_ovr, "auc_ovo": auc_ovo}


def composite_score(metrics: Dict[str, Optional[float]]) -> float:
    """
    Average of the available metrics in [0,1]. None values are ignored.
    If all metrics are None (should not happen), returns 0.0.
    """
    vals = []
    for k in ("acc", "macro_f1", "auc_ovr", "auc_ovo"):
        v = metrics.get(k, None)
        if v is None:
            continue
        # clip to [0,1] for safety
        vals.append(float(np.clip(v, 0.0, 1.0)))
    if not vals:
        return 0.0
    return float(np.mean(vals))
