# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Optional, Tuple

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm

from .sra import build_relations, softmax_with_temperature  # <-- 拆分后的SRA工具


# ---------------------------
# 名称归一化与预训练TAG映射
# ---------------------------
def _norm_name(name: str) -> str:
    return (name or "").strip().lower().replace("_", "-")

def _pretrain_tag(name: str) -> str:
    n = _norm_name(name)
    if n == "resnet101":
        return "Resnet101"
    if n == "resnet18":
        return "Resnet18"
    if n == "wideresnet101":
        return "Wideresnet101"
    if n == "shufflenetv2":
        return "ShuffleNetV2"
    if n in ("vit-b", "vit-b-16", "vitb", "vit-b16"):
        return "VIT_B"
    return name


# ---------------------------
# Backbones
# ---------------------------

def _build_backbone(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build a classifier backbone from torchvision with a linear head.
    兼容新旧API（weights / pretrained）。
    """
    raw_name = name
    name = _norm_name(name)

    def _ctor(model_name: str, pretrained_flag: bool):
        ctor = getattr(tvm, model_name)
        try:
            weights = tvm.get_model_weights(model_name).DEFAULT if pretrained_flag else None
            return ctor(weights=weights)
        except Exception:
            return ctor(pretrained=pretrained_flag)

    # --- ResNet 家族（18/34/50/101） ---
    if name in {"resnet18", "resnet34", "resnet50", "resnet101"}:
        net = _ctor(name, pretrained)
        in_feat = net.fc.in_features
        net.fc = nn.Linear(in_feat, num_classes)
        return net

    # --- WideResNet101-2 ---
    if name == "wideresnet101":
        # torchvision 名称：wide_resnet101_2
        net = _ctor("wide_resnet101_2", pretrained)
        in_feat = net.fc.in_features
        net.fc = nn.Linear(in_feat, num_classes)
        return net

    # --- ShuffleNetV2 x1.0 ---
    if name == "shufflenetv2":
        net = _ctor("shufflenet_v2_x1_0", pretrained)
        # torchvision 标准实现：使用 fc
        if hasattr(net, "fc") and isinstance(net.fc, nn.Linear):
            in_feat = net.fc.in_features
            net.fc = nn.Linear(in_feat, num_classes)
            return net
        # 罕见旧分支：存在 classifier
        if hasattr(net, "classifier"):
            head = net.classifier
            if isinstance(head, nn.Sequential) and len(head) > 0 and isinstance(head[-1], nn.Linear):
                in_feat = head[-1].in_features
                head[-1] = nn.Linear(in_feat, num_classes)
            elif isinstance(head, nn.Linear):
                in_feat = head.in_features
                head = nn.Linear(in_feat, num_classes)
            else:
                raise AttributeError("Unsupported ShuffleNetV2 classifier head structure.")
            net.classifier = head
            return net
        raise AttributeError("ShuffleNetV2 head not found: expecting .fc or .classifier")

    # --- EfficientNet-B0（你原来就有，保留） ---
    if name == "efficientnet-b0":
        try:
            weights = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net = tvm.efficientnet_b0(weights=weights)
        except Exception:
            net = tvm.efficientnet_b0(pretrained=pretrained)
        in_feat = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_feat, num_classes)
        return net

    # --- ViT-B/16 ---
    if name in {"vit-b", "vitb", "vit-b-16", "vit-b16"}:
        # torchvision 名称：vit_b_16
        net = _ctor("vit_b_16", pretrained)
        # 兼容不同版本的 heads 定义
        if hasattr(net, "heads") and hasattr(net.heads, "head"):
            in_feat = net.heads.head.in_features
            net.heads.head = nn.Linear(in_feat, num_classes)
        elif hasattr(net, "heads") and isinstance(net.heads, nn.Sequential):
            # 少见版本：heads 是一个顺序容器，末端是线性层
            if isinstance(net.heads[-1], nn.Linear):
                in_feat = net.heads[-1].in_features
                net.heads[-1] = nn.Linear(in_feat, num_classes)
            else:
                raise AttributeError("Unsupported ViT structure: heads[-1] is not Linear")
        else:
            raise AttributeError("Unsupported ViT structure: cannot locate classifier head")
        return net

    # 其他直接报错
    raise ValueError(f"Unsupported backbone: {raw_name}")


def _extract_features_and_logits(backbone: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    提取 penultimate feature + logits。
    - ResNet-like: 手工走干路，fc 前的 pooled 向量；
    - EfficientNet: 用 features + GAP；
    - ViT: 用 transformer 的分类头前一层；
    """
    # ResNet-like
    # ResNet-like: must have fc AND the standard residual stages
    if (
        hasattr(backbone, "fc") and isinstance(backbone.fc, nn.Linear)
        and all(hasattr(backbone, k) for k in ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "avgpool"])
    ):
        m = backbone
        x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
        x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
        x = m.avgpool(x)
        feat = torch.flatten(x, 1)
        logit = m.fc(feat)
        return feat, logit

    # EfficientNet-like
    if hasattr(backbone, "features") and hasattr(backbone, "classifier"):
        f = backbone.features(x)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        if isinstance(backbone.classifier, nn.Sequential):
            logit = backbone.classifier[-1](f)
        else:
            logit = backbone.classifier(f)
        return f, logit

    # ViT
    if hasattr(backbone, "heads"):
        # torchvision ViT 的 forward 直接返回 logits；提取 head 前特征相对繁琐，
        # 这里用 logits 前的输入张量替代（很多 KD 只需要 logits）。
        logit = backbone(x)
        # 用空特征占位，或也可以输出 cls token（需要深入到 encoder）
        feat = torch.empty(x.size(0), 0, device=x.device)
        return feat, logit

    # Fallback
    logit = backbone(x)
    feat = torch.empty(x.size(0), 0, device=x.device)
    return feat, logit


# ---------------------------
# Scope model (Student-Teacher wrapper)
# ---------------------------
class ScopeModel(nn.Module):
    """
    Student-Teacher model wrapper.
    SRA 的关系构建已拆到 models/sra.py；GCR 在 losses.py。
    """
    def __init__(
        self,
        num_classes: int,
        backbone_student: str = "resnet18",
        backbone_teacher: str = "resnet50",
        T: float = 4.0,
        teacher_ckpt: Optional[str] = None,
        pretrained_backbone: bool = True,
        dataset_name: Optional[str] = None,   # NEW: 用于自动拼教师预训练路径
        pretrain_root: str = "/nas/unas15/username/TNNLS/Pretrain",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.T = float(T)

        # student
        self.student = _build_backbone(backbone_student, num_classes, pretrained=pretrained_backbone)

        # teacher (frozen)
        self.teacher = _build_backbone(backbone_teacher, num_classes, pretrained=pretrained_backbone)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # 1) 优先使用显式 teacher_ckpt
        loaded = False
        if teacher_ckpt:
            loaded = self._load_teacher_weights(teacher_ckpt)

        # 2) 否则按命名规则自动查找：/nas/.../Pretrain_XX_<Dataset>.pth
        if (not loaded) and dataset_name:
            tag = _pretrain_tag(backbone_teacher)
            guess = os.path.join(pretrain_root, f"Pretrain_{tag}_{dataset_name}.pth")
            if os.path.isfile(guess):
                self._load_teacher_weights(guess)

    @torch.no_grad()
    def _load_teacher_weights(self, path: str) -> bool:
        try:
            sd = torch.load(path, map_location="cpu")
            # 兼容 state_dict 或完整对象
            state = sd.get("model", sd) if isinstance(sd, dict) else sd.state_dict()
            # 兼容 DDP
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
            missing, unexpected = self.teacher.load_state_dict(state, strict=False)
            if missing:
                print(f"[Scope] Teacher missing keys: {missing[:8]}{'...' if len(missing)>8 else ''}")
            if unexpected:
                print(f"[Scope] Teacher unexpected keys: {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")
            print(f"[Scope] Loaded teacher weights from: {path}")
            return True
        except Exception as e:
            print(f"[Scope] WARNING: failed to load teacher weights from {path}: {e}")
            return False

    # --------- forward ----------
    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        h_vec: Optional[torch.Tensor] = None,
        return_feat: bool = False,
        with_relations: bool = False,
        # ==== SRA (NEW) ====
        sra_variant: str = "full",        # full|teacher_only|label_only|fixed_alpha|no_ontology
        alpha_mode: str = "entropy",      # entropy|max
        fixed_alpha: float = 0.5,
    ) -> Dict[str, torch.Tensor]:

        """
        Returns:
          - student_logits / student_probs
          - teacher_logits_T / teacher_probs_T  (already temperature-scaled for KD)
          - (optional) student_feat / teacher_feat
          - (optional) SRA relations from models/sra.build_relations(...)
        """
        # Student
        s_feat, s_logit = _extract_features_and_logits(self.student, images)
        s_prob = F.softmax(s_logit, dim=-1)

        # Teacher (no grad)
        with torch.no_grad():
            t_feat, t_logit = _extract_features_and_logits(self.teacher, images)
        t_prob_T = softmax_with_temperature(t_logit, self.T)

        out = {
            "student_logits": s_logit,             # [B, C]
            "student_probs": s_prob,               # [B, C]

            # Teacher two flavors:
            "teacher_logits": t_logit,             # [B, C]  (UNSCALED, raw logits)   # NEW: alias
            "teacher_logits_T": t_logit / self.T,  # [B, C]  (scaled for convenience)
            "teacher_probs_T": t_prob_T,           # [B, C]
        }

        # Backward-compat aliases so other parts of code don't need edits
        out["logits"] = s_logit                    # NEW: alias for student logits

        if return_feat:
            out["student_feat"] = s_feat
            out["teacher_feat"] = t_feat

        if with_relations:
            rel = build_relations(
                student_probs=s_prob,
                teacher_probs_T=t_prob_T,
                num_classes=self.num_classes,
                labels=labels,
                h_vec=h_vec,
                # ==== SRA (NEW) ====
                alpha_mode=alpha_mode,
                variant=sra_variant,
                fixed_alpha=fixed_alpha,
            )
            out.update(rel)


        return out
