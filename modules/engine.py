# modules/engine.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .metrics import composite_score


class CheckpointEngine:
    """
    Handles:
      - Saving 'last.pth' at the end of every epoch
      - Tracking the best epoch by a monitored score (default: composite average of 4 metrics)
      - Saving 'best.pth' when the monitored score improves
    """

    def __init__(
        self,
        save_dir: str,
        monitor: str = "composite",   # "composite" or a specific key: "acc" / "macro_f1" / "auc_ovr" / "auc_ovo"
        higher_is_better: bool = True,
        keep_last: bool = True,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.higher_is_better = higher_is_better
        self.keep_last = keep_last

        self.best_score: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.best_path = self.save_dir / "best.pth"

    # --------------------------
    # Score helpers
    # --------------------------
    def _get_score(self, metrics: Dict[str, Any]) -> float:
        """
        Get the monitored score from a metrics dictionary.
        If monitor="composite", average the 4 metrics; otherwise use metrics[monitor].
        """
        if self.monitor == "composite":
            return composite_score(metrics)
        val = metrics.get(self.monitor, None)
        if val is None:
            # fall back to composite when target key not present
            return composite_score(metrics)
        return float(val)

    def _is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        return (score > self.best_score) if self.higher_is_better else (score < self.best_score)

    # --------------------------
    # Save helpers
    # --------------------------
    @staticmethod
    def _to_cpu_state(state: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure tensors are moved to CPU before serialization."""
        def _move(v):
            if isinstance(v, torch.Tensor):
                return v.detach().cpu()
            return v
        return {k: _move(v) for k, v in state.items()}

    def save_last(
        self,
        epoch: int,
        model_state_dict: Dict[str, Any],
        optimizer_state_dict: Optional[Dict[str, Any]] = None,
        scheduler_state_dict: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        cfg: Optional[Dict[str, Any]] = None,
        filename: str = "last.pth",
    ) -> Path:
        """Save the 'last' checkpoint (always)."""
        path = self.save_dir / filename
        state = {
            "epoch": int(epoch),
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "scheduler": scheduler_state_dict,
            "metrics": metrics,
            "cfg": cfg,
        }
        torch.save(self._to_cpu_state(state), path.as_posix())
        return path

    def save_if_best(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        model_state_dict: Dict[str, Any],
        optimizer_state_dict: Optional[Dict[str, Any]] = None,
        scheduler_state_dict: Optional[Dict[str, Any]] = None,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """
        Save 'best.pth' if the monitored score improves.
        Returns the path if saved, otherwise None.
        """
        score = self._get_score(metrics)
        if self._is_better(score):
            self.best_score = score
            self.best_epoch = int(epoch)
            state = {
                "epoch": int(epoch),
                "best_score": float(score),
                "monitor": self.monitor,
                "model": model_state_dict,
                "optimizer": optimizer_state_dict,
                "scheduler": scheduler_state_dict,
                "metrics": metrics,
                "cfg": cfg,
            }
            torch.save(self._to_cpu_state(state), self.best_path.as_posix())
            return self.best_path
        return None
