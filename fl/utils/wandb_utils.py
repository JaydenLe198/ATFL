"""Weights & Biases helpers (optional)."""

from __future__ import annotations

from typing import Any, Dict, Optional

_WANDB_AVAILABLE = False
try:
    import wandb  # type: ignore

    _WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore

_RUN = None  # type: ignore


def maybe_init_wandb(cfg: Dict[str, Any]):
    global _RUN
    wb = cfg.get("wandb", {}) if isinstance(cfg, dict) else {}
    if not wb or not bool(wb.get("enabled", False)):
        return None
    if not _WANDB_AVAILABLE:
        print("[wandb] Not installed; skipping logging.")
        return None
    _RUN = wandb.init(
        project=wb.get("project", "ATFL"),
        entity=wb.get("entity"),
        name=wb.get("run_name"),
        tags=wb.get("tags"),
        notes=wb.get("notes"),
        config={
            "simulation": cfg.get("simulation", {}),
            "training": cfg.get("training", {}),
            "model": cfg.get("model", {}),
            "data": cfg.get("data", {}),
        },
    )
    return _RUN


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    if _RUN is None or not _WANDB_AVAILABLE:
        return
    wandb.log(metrics, step=step)


def log_artifact(path: str, name: Optional[str] = None, type_: str = "artifact") -> None:
    if _RUN is None or not _WANDB_AVAILABLE:
        return
    artifact = wandb.Artifact(name or os.path.basename(path), type=type_)
    artifact.add_file(path)
    _RUN.log_artifact(artifact)


def finish() -> None:
    global _RUN
    if _RUN is None or not _WANDB_AVAILABLE:
        return
    try:
        _RUN.finish()
    except Exception:
        pass
    _RUN = None
