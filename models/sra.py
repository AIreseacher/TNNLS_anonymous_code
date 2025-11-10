from __future__ import annotations
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def softmax_with_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    return F.softmax(logits / T, dim=-1)


def pairwise_cosine(mat: torch.Tensor) -> torch.Tensor:
    """
    Pairwise cosine similarity for row vectors.
    mat: [B, D] -> [B, B]
    NOTE: This must remain differentiable w.r.t. `mat` because REL needs
    gradient to flow back to the student.
    """
    mat = F.normalize(mat, p=2, dim=-1)
    return mat @ mat.t()


def entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Shannon entropy for categorical distribution per sample."""
    p = p.clamp(min=eps)
    return -(p * p.log()).sum(dim=-1)


def build_relations(
    *,
    student_probs: torch.Tensor,          # [B, C], must require grad
    teacher_probs_T: torch.Tensor,        # [B, C], temperature-softmaxed; no grad is OK
    num_classes: int,
    labels: Optional[torch.Tensor] = None,  # [B]
    h_vec: Optional[torch.Tensor] = None,   # [B, H] prior vector (optional)
    alpha_mode: str = "entropy",
    variant: str = "full",                  # NEW: full|teacher_only|label_only|fixed_alpha|no_ontology
    fixed_alpha: float = 0.5               # NEW: used when variant=fixed_alpha
) -> Dict[str, torch.Tensor]:
    """
    Return SRA matrices:
      - R_s: student relation (B,B)           (keeps grad for student)
      - R_t: teacher relation (B,B)           (no grad needed)
      - R_y: label/prior relation (B,B)       (no grad)
      - R_t_prime: alpha-weighted fusion of R_t and R_y (B,B)  (no grad)
      - alpha: pairwise confidence weights (B,B)               (no grad)
    """
    B = student_probs.size(0)
    teacher_probs_T = teacher_probs_T.detach()

    # Student relation (keeps grad)
    R_s = pairwise_cosine(student_probs)

    with torch.no_grad():
        # Teacher relation
        R_t = pairwise_cosine(teacher_probs_T)

        # R_y depends on variant
        if variant == "no_ontology":
            R_y = torch.zeros_like(R_t)
        else:
            if labels is not None:
                R_y = (labels.view(B, 1) == labels.view(1, B)).float()
            elif h_vec is not None and h_vec.numel() > 0:
                R_y = pairwise_cosine(h_vec)
            else:
                R_y = torch.zeros_like(R_t)

        # alpha weights
        if variant == "fixed_alpha":
            a = float(max(0.0, min(1.0, fixed_alpha)))
            alpha = torch.full_like(R_t, a)
        else:
            if alpha_mode == "entropy":
                H = entropy(teacher_probs_T)  # [B]
                H_max = torch.log(torch.tensor(float(num_classes), device=H.device, dtype=H.dtype))
                conf = 1.0 - (H / (H_max + 1e-8))
                conf = conf.clamp(0.0, 1.0)
            else:
                conf = teacher_probs_T.max(dim=-1).values
            alpha = conf.view(B, 1) * conf.view(1, B)  # [B,B]

        # R_t_prime depends on variant
        if variant == "teacher_only":
            R_t_prime = R_t
        elif variant == "label_only":
            R_t_prime = R_y
        else:  # full, fixed_alpha, no_ontology
            R_t_prime = alpha * R_t + (1.0 - alpha) * R_y

    return {"R_s": R_s, "R_t": R_t, "R_y": R_y, "R_t_prime": R_t_prime, "alpha": alpha}
