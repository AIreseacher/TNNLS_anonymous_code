# gcr.py
from __future__ import annotations
from typing import List, Tuple, Optional
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _flatten_param_grads(params: List[nn.Parameter]) -> Tuple[torch.Tensor, List[torch.Size], torch.device]:
    """
    Flatten per-parameter grads into a single 1-D vector, and return:
      - flat_vec: concatenated grad vector
      - shapes  : original grad shapes (None if the param does not require grad)
      - device  : common device
    If a parameter's grad is None, we insert a zero vector with the SAME length
    as that parameter to keep vector lengths consistent across CE/KD/REL passes.
    """
    flat_parts: List[torch.Tensor] = []
    shapes: List[Optional[torch.Size]] = []
    device: Optional[torch.device] = None

    for p in params:
        if not p.requires_grad:
            shapes.append(None)
            continue
        if device is None:
            device = p.device
        if p.grad is None:
            shapes.append(p.shape)
            flat_parts.append(torch.zeros(p.numel(), device=device))
        else:
            shapes.append(p.grad.shape)
            flat_parts.append(p.grad.detach().view(-1).to(device))

    if not flat_parts:
        device = device or torch.device("cpu")
        return torch.zeros(0, device=device), shapes, device
    return torch.cat(flat_parts, dim=0), shapes, (device or torch.device("cpu"))


def _assign_grads_from_flat(params: List[nn.Parameter], flat_vec: torch.Tensor, shapes: List[Optional[torch.Size]]) -> None:
    """
    Scatter a flat grad vector back to each parameter's .grad with original shapes.
    Only parameters with requires_grad=True will receive slices.
    """
    offset = 0
    for p, shp in zip(params, shapes):
        if (not p.requires_grad) or (shp is None):
            continue
        numel = 1
        for s in shp:
            numel *= s
        g_slice = flat_vec[offset:offset + numel].view(shp)
        offset += numel
        if p.grad is None:
            p.grad = g_slice.clone()
        else:
            p.grad.copy_(g_slice)


def _maybe_log_cos(g_kd_flat: torch.Tensor,
                   g_ce_flat: torch.Tensor,
                   counter: Optional[dict],
                   eps: float = 1e-12) -> None:
    """
    Log raw cos<gKD, gCE> BEFORE any projection if the sampling gate is on.
    This records the ORIGINAL angle for fair comparison across methods.
    """
    if counter is None or not counter.get("collect_cos", False):
        return
    cos_list = counter.setdefault("cos", [])
    if len(cos_list) >= int(counter.get("collect_limit", 0)):
        return
    nkd = g_kd_flat.norm()
    nce = g_ce_flat.norm()
    # skip degenerate cases
    if nkd.item() == 0.0 or nce.item() == 0.0:
        return
    dot = torch.dot(g_kd_flat, g_ce_flat)
    cos = (dot / (nkd * nce + eps)).item()
    cos_list.append(float(cos))


# -----------------------------------------------------------------------------
# GCR backward
# -----------------------------------------------------------------------------
@torch.cuda.amp.autocast(enabled=False)  # keep projection/algebra in FP32 for stability
def backward_with_gcr(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ce: torch.Tensor,                 # CE loss (primary task)
    kd: torch.Tensor,                 # KD loss (distillation term)
    rel: torch.Tensor,                # any extra regularizer (e.g., SRA)
    beta: float = 1.0,                # CE coefficient
    gamma: float = 1.0,               # KD coefficient
    step: bool = True,                # whether to call optimizer.step()
    zero_grad_each_step: bool = True, # set_to_none=True between passes
    conflict_counter: Optional[dict] = None,
    grad_clip_norm: Optional[float] = None,
    eps: float = 1e-12,
) -> None:
    """
    Execute one backward+step for (CE, KD, REL) with Gradient-Conflict Resolution:
      - Compute grads for REL, CE, KD separately on the student parameters.
      - If <gKD, gCE> < 0, project gKD onto the orthogonal complement of gCE:
            gKD <- gKD - <gKD,gCE> / (||gCE||^2 + eps) * gCE
      - Compose the total grad:
            g_total = beta * gCE + gamma * gKD (+ gREL)
      - Assign g_total back to model parameters and step the optimizer.

    Notes:
      * We log the ORIGINAL cos<gKD, gCE> BEFORE projection via `conflict_counter`.
      * If some params have no grad in a pass, zero vectors with matching lengths
        are inserted to keep lengths aligned across passes.
    """
    student = model  # by convention, caller passes model.student
    params = [p for p in student.parameters() if p.requires_grad]

    # ---------------------------
    # 1) REL pass (optional)
    # ---------------------------
    optimizer.zero_grad(set_to_none=True)
    rel_val = float(rel.detach()) if rel is not None else 0.0
    if (rel is not None) and (rel_val != 0.0):
        rel.backward(retain_graph=True)
        g_rel_flat, shapes, device = _flatten_param_grads(params)
    else:
        # still need shapes/device for consistency
        # do a harmless scan to infer shapes/device
        g_rel_flat, shapes, device = _flatten_param_grads(params)
        # zero it explicitly
        if g_rel_flat.numel() > 0:
            g_rel_flat.zero_()

    # ---------------------------
    # 2) CE pass (primary)
    # ---------------------------
    optimizer.zero_grad(set_to_none=True)
    if ce is None:
        raise RuntimeError("CE must not be None for GCR.")
    ce.backward(retain_graph=True)
    g_ce_flat, shapes_ce, _ = _flatten_param_grads(params)
    # shapes must be the same across passes by construction
    assert g_ce_flat.numel() == g_rel_flat.numel(), "CE and REL grad vector sizes mismatch."

    # ---------------------------
    # 3) KD pass
    # ---------------------------
    optimizer.zero_grad(set_to_none=True)
    kd.backward(retain_graph=True)
    g_kd_flat, shapes_kd, _ = _flatten_param_grads(params)
    assert g_kd_flat.numel() == g_ce_flat.numel(), "KD and CE grad vector sizes mismatch."

    # ---------------------------
    # 4) Log raw cosine BEFORE projection (for analysis)
    # ---------------------------
    _maybe_log_cos(g_kd_flat, g_ce_flat, conflict_counter, eps=eps)

    # ---------------------------
    # 5) Conflict check & projection
    # ---------------------------
    dot = torch.dot(g_kd_flat, g_ce_flat)
    if dot.item() < 0.0:
        denom = g_ce_flat.norm().pow(2) + eps
        proj = (dot / denom) * g_ce_flat
        g_kd_flat = g_kd_flat - proj  # remove the conflicting component

    # ---------------------------
    # 6) Compose total gradient
    # ---------------------------
    g_total = beta * g_ce_flat + gamma * g_kd_flat
    if g_rel_flat.numel() > 0:
        g_total = g_total + g_rel_flat

    # ---------------------------
    # 7) Scatter back & step
    # ---------------------------
    _assign_grads_from_flat(params, g_total, shapes)

    if grad_clip_norm is not None and grad_clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip_norm)

    if step:
        optimizer.step()
        if zero_grad_each_step:
            optimizer.zero_grad(set_to_none=True)
