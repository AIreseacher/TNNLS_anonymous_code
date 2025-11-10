# pcgrad.py
from __future__ import annotations
import random
from typing import Optional, List, Tuple
import torch
from torch import nn


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _flatten_param_grads(params: List[nn.Parameter]) -> Tuple[torch.Tensor, List[Optional[torch.Size]], torch.device]:
    """
    Flatten per-parameter grads into a single 1-D vector, and also return
    the original grad shapes and a common device. If a parameter's grad is
    None, we insert a zero vector with the SAME length as that parameter,
    so that different passes (CE/KD/REL) have consistent vector lengths.
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


def _maybe_log_cos_raw(g_ce: torch.Tensor, g_kd: torch.Tensor, counter: Optional[dict], eps: float = 1e-12) -> None:
    """
    Log cos<gKD, gCE> BEFORE any PCGrad pairwise projection (original angle).
    """
    if counter is None or not counter.get("collect_cos", False):
        return
    cos_list = counter.setdefault("cos", [])
    if len(cos_list) >= int(counter.get("collect_limit", 0)):
        return
    n1 = g_ce.norm()
    n2 = g_kd.norm()
    if n1.item() == 0.0 or n2.item() == 0.0:
        return
    cos = (torch.dot(g_kd, g_ce) / (n1 * n2 + eps)).item()
    cos_list.append(float(cos))


# -----------------------------------------------------------------------------
# PCGrad backward
# -----------------------------------------------------------------------------
@torch.cuda.amp.autocast(enabled=False)  # keep projection in FP32 for stability
def backward_with_pcgrad(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ce: torch.Tensor,
    kd: torch.Tensor,
    rel: torch.Tensor,
    beta: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-12,
    scheduler=None,
    step_scheduler_per_batch: bool = False,
    conflict_counter: Optional[dict] = None,
) -> None:
    """
    PCGrad backward pass:
      - Treat CE / KD / REL as independent task gradients.
      - Perform pairwise projection to mitigate conflicts.
      - Weighted-sum the post-projection grads using (1.0, beta, gamma).
    We also log the ORIGINAL cos<gKD, gCE> BEFORE any projection if requested.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    def _grad_from_loss(loss: torch.Tensor) -> Tuple[torch.Tensor, List[Optional[torch.Size]], torch.device]:
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        g_flat, shapes, device = _flatten_param_grads(params)
        optimizer.zero_grad(set_to_none=True)
        return g_flat, shapes, device

    grads_info: List[Tuple[torch.Tensor, float, str]] = []  # (flat_grad, weight, name)
    shapes: List[Optional[torch.Size]] = []
    device: torch.device = params[0].device if params else torch.device("cpu")

    # REL (optional)
    if rel is not None and rel.requires_grad and (float(rel.detach()) != 0.0):
        g_rel, shapes, device = _grad_from_loss(rel)
        if g_rel.numel() > 0:
            grads_info.append((g_rel.clone(), 1.0, "REL"))
    else:
        # still infer shapes/device for later use
        g_rel, shapes, device = _flatten_param_grads(params)
        if g_rel.numel() > 0:
            g_rel.zero_()

    # CE (primary)
    g_ce, _, _ = _grad_from_loss(ce)
    if g_ce.numel() > 0:
        grads_info.append((g_ce.clone(), beta, "CE"))

    # KD
    if kd is not None and kd.requires_grad and (float(kd.detach()) != 0.0):
        g_kd, _, _ = _grad_from_loss(kd)
        if g_kd.numel() > 0:
            grads_info.append((g_kd.clone(), gamma, "KD"))
    else:
        # keep alignment with shapes by adding explicit zeros if needed
        g_kd = torch.zeros_like(g_ce)

    if not grads_info:
        return

    # --- Log ORIGINAL cos before any PCGrad projections (NEW) ---
    g_ce_raw = None
    g_kd_raw = None
    for g, _, name in grads_info:
        if name == "CE":
            g_ce_raw = g
        elif name == "KD":
            g_kd_raw = g
    if (g_ce_raw is not None) and (g_kd_raw is not None):
        _maybe_log_cos_raw(g_ce_raw, g_kd_raw, conflict_counter)

    # Prepare a local copy for pairwise projections
    grads = [g.clone() for g, _, _ in grads_info]
    order = list(range(len(grads)))
    random.shuffle(order)

    total_pairs = 0
    total_conflicts = 0
    for i in order:
        gi = grads[i]
        for j in order:
            if i == j:
                continue
            gj = grads[j]
            dot = torch.dot(gi, gj)
            if dot.item() < 0.0:
                gi = gi - (dot / (gj.norm().pow(2) + eps)) * gj
                total_conflicts += 1
            total_pairs += 1
        grads[i] = gi

    if conflict_counter is not None:
        conflict_counter["pcgrad_pairs"] = conflict_counter.get("pcgrad_pairs", 0) + total_pairs
        conflict_counter["pcgrad_conflicts"] = conflict_counter.get("pcgrad_conflicts", 0) + total_conflicts

    # Weighted sum after PCGrad
    g_total = torch.zeros_like(grads[0])
    for gi, (_, w, _) in zip(grads, grads_info):
        g_total = g_total + w * gi

    _assign_grads_from_flat(params, g_total, shapes)
    optimizer.step()
    if scheduler is not None and step_scheduler_per_batch:
        try:
            scheduler.step()
        except TypeError:
            pass
