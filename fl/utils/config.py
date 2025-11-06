"""Configuration loader for ATFL."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to load configs.") from exc


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_config_path() -> Path:
    return _project_root() / "fl" / "configs" / "default.yaml"


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in upd.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def default_config() -> Dict[str, Any]:
    return {
        "ray": {"num_cpus": 32, "num_gpus": 4, "ignore_reinit_error": True},
        "resources": {"client": {"num_cpus": 4, "num_gpus": 1}},
        "simulation": {
            "num_clients": 24,
            "total_rounds": 10,
            "fraction_fit": 0.25,
            "fraction_evaluate": 0.0,
            "min_available_clients": 6,
            "min_fit_clients": 6,
            "strict_num_clients": False,
        },
        "training": {"local_epochs": 5, "lr": 1e-4, "proximal_mu": 0.0},
        "model": {
            "seq_len": 100,
            "enc_in": 25,
            "c_out": 25,
            "d_model": 512,
            "n_heads": 8,
            "e_layers": 3,
        },
        "data": {
            "dataset": "PSM",
            "partition_mode": "sequential",
            "batch_size": 64,
            "seq_length": 100,
            "step": 1,
            "centralized_cache": False,
            "cache_dir": ".cache/dataset",
            "cache_clients": None,
            "client_load_test": False,
        },
        "evaluation": {
            "enabled": False,
            "seq_length": 50,
            "step": 5,
            "num_thresholds": 256,
            "min_consecutive": 5,
            "threshold_quantile_range": [0.01, 0.99],
        },
        "post_training": {
            "generate_arrays": True,
            "dataset": "PSM",
            "seq_length": 100,
        },
        "wandb": {
            "enabled": False,
            "project": "ATFL",
            "entity": None,
            "run_name": None,
        },
    }


def load_config(path: Optional[Union[str, os.PathLike]] = None) -> Dict[str, Any]:
    defaults = default_config()

    env_path = os.environ.get("ATFL_CONFIG")
    candidate = Path(path) if path else (Path(env_path) if env_path else _default_config_path())
    if candidate and candidate.exists():
        try:
            with open(candidate, "r", encoding="utf-8") as fin:
                loaded = yaml.safe_load(fin) or {}
            if not isinstance(loaded, dict):
                raise ValueError("Configuration must be a mapping.")
            return _deep_update(defaults, loaded)
        except Exception as exc:
            print(f"[Config] Failed to load {candidate}: {exc}. Using defaults.")
            return defaults
    return defaults


__all__ = ["load_config", "default_config"]
