# cagrad.py
from __future__ import annotations
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
    flat_parts = []
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
    Log cos<gKD, gCE> BEFORE any CAGrad convex combination (original angle).
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
# CAGrad backward
# -----------------------------------------------------------------------------
@torch.cuda.amp.autocast(enabled=False)  # keep algebra in FP32 for stability
def backward_with_cagrad(
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
    CAGrad backward pass (2-task closed-form for CE vs KD) with optional REL:
      - Compute flat grads for REL, CE, KD separately.
      - Log ORIGINAL cos<gKD, gCE> before any convex combination (for analysis).
      - Compute closed-form weight w* to minimize gradient norm of convex combo.
      - Compose total grad = g_rel + [w* * (beta*g_ce) + (1-w*) * (gamma*g_kd)].
    """
    params = [p for p in model.parameters() if p.requires_grad]

    def _grad_from_loss(loss: torch.Tensor) -> Tuple[torch.Tensor, List[Optional[torch.Size]], torch.device]:
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        g_flat, shapes, device = _flatten_param_grads(params)
        optimizer.zero_grad(set_to_none=True)
        return g_flat, shapes, device

    shapes: List[Optional[torch.Size]] = []
    device: torch.device = params[0].device if params else torch.device("cpu")

    # REL (optional)
    g_rel = torch.zeros(0, device=device)
    if rel is not None and rel.requires_grad and (float(rel.detach()) != 0.0):
        g_rel, shapes, device = _grad_from_loss(rel)
    else:
        g_rel, shapes, device = _flatten_param_grads(params)
        if g_rel.numel() > 0:
            g_rel.zero_()

    # CE (primary)
    g_ce, _, _ = _grad_from_loss(ce)

    # KD
    if kd is not None and kd.requires_grad and (float(kd.detach()) != 0.0):
        g_kd, _, _ = _grad_from_loss(kd)
    else:
        g_kd = torch.zeros_like(g_ce)

    # Handle degenerate cases (rare)
    if g_ce.numel() == 0 or g_kd.numel() == 0:
        g_total = g_rel + beta * g_ce + gamma * g_kd
        _assign_grads_from_flat(params, g_total, shapes)
        optimizer.step()
        if scheduler is not None and step_scheduler_per_batch:
            try:
                scheduler.step()
            except TypeError:
                pass
        return

    # --- Log ORIGINAL cos before convex combination (NEW) ---
    _maybe_log_cos_raw(g_ce, g_kd, conflict_counter)

    # Closed-form convex combination between g1 and g2
    g1 = beta * g_ce
    g2 = gamma * g_kd
    diff = g1 - g2
    denom = diff.norm().pow(2) + eps
    num = g2.norm().pow(2) - torch.dot(g1, g2)
    w_star = torch.clamp(num / denom, 0.0, 1.0)

    if conflict_counter is not None:
        conflict_counter["cagrad_w"] = float(w_star)

    g_ck = w_star * g1 + (1.0 - w_star) * g2
    g_total = g_rel + g_ck

    _assign_grads_from_flat(params, g_total, shapes)
    optimizer.step()
    if scheduler is not None and step_scheduler_per_batch:
        try:
            scheduler.step()
        except TypeError:
            pass
