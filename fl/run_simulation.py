"""Flower simulation entry point for ATFL."""

from __future__ import annotations

import argparse
import io
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import ray
import torch
import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.server import ServerConfig
from flwr.server.strategy import FedProx
from flwr.server.strategy.aggregate import aggregate

from fl.client_app import client_fn
from fl.task import (
    build_model,
    get_weights,
    set_weights,
    load_centralized_test_data,
    get_anomaly_scores,
)
from fl.utils.config import load_config
from fl.utils.wandb_utils import maybe_init_wandb, log_metrics, finish as wandb_finish
from fl.utils.evaluation import get_threshold_tranad, save_tranad_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ATFL Flower simulation.")
    parser.add_argument("config", nargs="?", help="Path to YAML config.")
    parser.add_argument(
        "--results-dir",
        dest="results_dir",
        default=None,
        help="Override evaluation results directory (relative or absolute path).",
    )
    return parser.parse_args()


def fit_config(server_round: int, training_cfg: Dict[str, float]) -> Dict[str, fl.common.Scalar]:
    return {
        "server_round": server_round,
        "local_epochs": training_cfg.get("local_epochs", 1),
        "proximal_mu": training_cfg.get("proximal_mu", 0.0),
        "lr": float(training_cfg.get("lr", 1e-4)),
    }


class SaveResultsStrategy(FedProx):
    """FedProx strategy storing latest model for post-training inference."""

    def __init__(self, *args, cfg_ref: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_parameters: Optional[fl.common.Parameters] = None
        self._cfg_ref = cfg_ref or {}
        self._wb_cfg = self._cfg_ref.get("wandb", {}) if isinstance(self._cfg_ref, dict) else {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[fl.common.Parameters, Dict[str, fl.common.Scalar]]:
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters

        try:
            wb_cfg = self._wb_cfg
            per_client = bool(wb_cfg.get("log_per_client", True))
            max_logged = int(wb_cfg.get("max_logged_clients", 32))
            client_times = []
            per_client_logs: Dict[str, float] = {}
            logged_clients = 0
            for cp, fr in results:
                if isinstance(fr.metrics.get("train_time_s"), (int, float)):
                    client_times.append(float(fr.metrics["train_time_s"]))
                if (
                    per_client
                    and logged_clients < max_logged
                    and isinstance(fr.metrics.get("train_loss"), (int, float))
                ):
                    cid = getattr(cp, "cid", None)
                    if cid is not None:
                        per_client_logs[f"client/{cid}/loss"] = float(fr.metrics["train_loss"])
                        logged_clients += 1
            log_dict: Dict[str, float] = {}
            if client_times:
                log_dict["clients/time_mean_s"] = float(np.mean(client_times))
                log_dict["clients/time_std_s"] = float(np.std(client_times))
            log_dict.update(per_client_logs)
            if log_dict:
                log_metrics(log_dict, step=server_round)
        except Exception:
            pass
        return aggregated_parameters, metrics


class ScaffoldStrategy(SaveResultsStrategy):
    """Minimal SCAFFOLD extension of FedProx."""

    def __init__(self, *args, param_templates: List[np.ndarray], damping: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_c = [np.zeros_like(p) for p in param_templates]
        self.client_ci: Dict[str, List[np.ndarray]] = {}
        self.damping = float(damping)

    def configure_fit(self, server_round: int, parameters: fl.common.Parameters, client_manager):
        fit_cfg = super().configure_fit(server_round, parameters, client_manager)
        for client, fit_ins in fit_cfg:
            cid = getattr(client, "cid", "unknown")
            if cid not in self.client_ci:
                self.client_ci[cid] = [np.zeros_like(p) for p in self.global_c]

            def _pack(arrs: List[np.ndarray]) -> bytes:
                buf = io.BytesIO()
                np.savez_compressed(buf, **{f"a{i}": a for i, a in enumerate(arrs)})
                return buf.getvalue()

            fit_ins.config["c_bytes"] = _pack(self.global_c)
            fit_ins.config["ci_bytes"] = _pack(self.client_ci[cid])
            fit_ins.config["proximal_mu"] = 0.0
        return fit_cfg

    def aggregate_fit(self, server_round, results, failures):
        filtered = [
            (cp, fr)
            for cp, fr in results
            if fr is not None and fr.parameters is not None
        ]
        if not filtered:
            return super().aggregate_fit(server_round, results, failures)

        weights_results = [
            (parameters_to_ndarrays(fr.parameters), fr.num_examples)
            for _, fr in filtered
        ]
        new_weights = aggregate(weights_results)
        aggregated_parameters = ndarrays_to_parameters(new_weights)
        metrics: Dict[str, fl.common.Scalar] = {}

        deltas = []
        cids = []
        for cp, fr in filtered:
            cid = getattr(cp, "cid", "unknown")
            bytes_blob = fr.metrics.get("delta_ci")
            if isinstance(bytes_blob, (bytes, bytearray)):
                try:
                    buf = io.BytesIO(bytes_blob)
                    with np.load(buf) as npz:
                        deltas.append([npz[k] for k in sorted(npz.files, key=lambda x: int(x[1:]))])
                        cids.append(cid)
                except Exception:
                    continue
        if deltas:
            for cid, delta in zip(cids, deltas):
                if cid not in self.client_ci:
                    self.client_ci[cid] = [np.zeros_like(p) for p in self.global_c]
                for idx, arr in enumerate(delta):
                    self.client_ci[cid][idx] = self.client_ci[cid][idx] + self.damping * arr
            mean_delta = [np.zeros_like(p) for p in self.global_c]
            for delta in deltas:
                for idx, arr in enumerate(delta):
                    mean_delta[idx] += self.damping * arr
            for idx, arr in enumerate(mean_delta):
                self.global_c[idx] += arr / float(len(deltas))

        self.latest_parameters = aggregated_parameters

        try:
            wb_cfg = getattr(self, "_wb_cfg", {})
            per_client = bool(wb_cfg.get("log_per_client", True))
            max_logged = int(wb_cfg.get("max_logged_clients", 32))
            per_client_logs: Dict[str, float] = {}
            client_times: List[float] = []
            logged_clients = 0
            for cp, fr in filtered:
                cid = getattr(cp, "cid", None)
                m = getattr(fr, "metrics", {}) or {}
                if isinstance(m.get("train_time_s"), (int, float)):
                    client_times.append(float(m["train_time_s"]))
                if (
                    per_client
                    and cid is not None
                    and logged_clients < max_logged
                    and isinstance(m.get("train_loss"), (int, float))
                ):
                    per_client_logs[f"client/{cid}/loss"] = float(m["train_loss"])
                    logged_clients += 1
            log_dict: Dict[str, float] = {}
            if client_times:
                log_dict["clients/time_mean_s"] = float(np.mean(client_times))
                log_dict["clients/time_std_s"] = float(np.std(client_times))
            log_dict.update(per_client_logs)
            if log_dict:
                log_metrics(log_dict, step=server_round)
        except Exception:
            pass

        return aggregated_parameters, metrics


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    if args.config:
        os.environ["ATFL_CONFIG"] = args.config
    elif "ATFL_CONFIG" in os.environ:
        pass
    else:
        os.environ["ATFL_CONFIG"] = ""

    run = maybe_init_wandb(cfg)
    sim_start = time.time()

    training_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    seq_len = int(model_cfg.get("seq_len", 100))
    model_kwargs = {
        "win_size": seq_len,
        "enc_in": int(model_cfg.get("enc_in", 25)),
        "c_out": int(model_cfg.get("c_out", 25)),
        "d_model": int(model_cfg.get("d_model", 512)),
        "n_heads": int(model_cfg.get("n_heads", 8)),
        "e_layers": int(model_cfg.get("e_layers", 3)),
    }

    base_model = build_model(model_kwargs)
    ndarrays = get_weights(base_model)
    initial_parameters = ndarrays_to_parameters(ndarrays)

    ray_cfg = cfg.get("ray", {})
    ray.init(
        num_cpus=int(ray_cfg.get("num_cpus", 32)),
        num_gpus=float(ray_cfg.get("num_gpus", 4)),
        ignore_reinit_error=bool(ray_cfg.get("ignore_reinit_error", True)),
    )

    client_resources = cfg.get("resources", {}).get("client", {"num_cpus": 4, "num_gpus": 1})

    strategy_cfg = cfg.get("strategy", {})
    strategy_type = str(strategy_cfg.get("type", "fedprox")).lower()

    if strategy_type == "scaffold":
        strategy = ScaffoldStrategy(
            fraction_fit=float(cfg["simulation"].get("fraction_fit", 0.25)),
            fraction_evaluate=0.0,
            min_available_clients=int(cfg["simulation"].get("min_available_clients", 1)),
            min_fit_clients=int(cfg["simulation"].get("min_fit_clients", 1)),
            initial_parameters=initial_parameters,
            on_fit_config_fn=lambda rnd: fit_config(rnd, training_cfg),
            proximal_mu=0.0,
            cfg_ref=cfg,
            param_templates=[p.copy() for p in ndarrays],
            damping=float(cfg.get("scaffold", {}).get("damping", 0.1)),
        )
    else:
        strategy = SaveResultsStrategy(
            fraction_fit=float(cfg["simulation"].get("fraction_fit", 0.25)),
            fraction_evaluate=0.0,
            min_available_clients=int(cfg["simulation"].get("min_available_clients", 1)),
            min_fit_clients=int(cfg["simulation"].get("min_fit_clients", 1)),
            initial_parameters=initial_parameters,
            on_fit_config_fn=lambda rnd: fit_config(rnd, training_cfg),
            proximal_mu=float(training_cfg.get("proximal_mu", 0.0)),
            cfg_ref=cfg,
        )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=int(cfg["simulation"].get("num_clients", 1)),
        client_resources=client_resources,
        config=ServerConfig(num_rounds=int(cfg["simulation"].get("total_rounds", 1))),
        ray_init_args={
            "num_cpus": int(ray_cfg.get("num_cpus", 32)),
            "num_gpus": float(ray_cfg.get("num_gpus", 4)),
            "ignore_reinit_error": bool(ray_cfg.get("ignore_reinit_error", True)),
        },
        strategy=strategy,
    )

    sim_duration = time.time() - sim_start
    print(f"Simulation finished in {sim_duration:.2f}s")

    final_parameters = getattr(strategy, "latest_parameters", None)
    if final_parameters is None:
        print("[Post-Training] No aggregated parameters found, using initial weights.")
        final_parameters = initial_parameters

    final_weights = parameters_to_ndarrays(final_parameters)
    final_model = build_model(model_kwargs)
    set_weights(final_model, final_weights)

    EVAL_CFG = cfg.get("evaluation", {})
    if bool(EVAL_CFG.get("enabled", False)):
        dataset_name = str(cfg.get("data", {}).get("dataset", "PSM"))
        seq_length_eval = int(cfg.get("data", {}).get("seq_length", seq_len))
        step_eval = int(cfg.get("data", {}).get("step", 1))
        batch_size_eval = int(EVAL_CFG.get("batch_size", cfg.get("data", {}).get("batch_size", 64)))
        if args.results_dir:
            results_dir = Path(args.results_dir)
        else:
            results_dir = Path(EVAL_CFG.get("results_dir", "eval_results"))
        results_dir.mkdir(parents=True, exist_ok=True)

        try:
            testloader = load_centralized_test_data(
                dataset_name=dataset_name,
                batch_size=batch_size_eval,
                seq_length=seq_length_eval,
                step=step_eval,
            )
            if len(testloader.dataset) == 0:
                print("[Evaluation] Centralized test loader is empty; skipping TranAD metrics.")
            else:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                eval_t0 = time.time()
                scores, labels = get_anomaly_scores(final_model, testloader, device=device)
                eval_dt = time.time() - eval_t0
                metrics = get_threshold_tranad(labels, scores, verbose=True)
                (
                    auc_val,
                    precision,
                    recall,
                    f1,
                    precision_adj,
                    recall_adj,
                    f1_adj,
                    threshold,
                ) = metrics
                metric_items = [
                    ("auc", auc_val),
                    ("precision", precision),
                    ("recall", recall),
                    ("f1", f1),
                    ("precision_adjusted", precision_adj),
                    ("recall_adjusted", recall_adj),
                    ("f1_adjusted", f1_adj),
                    ("threshold", threshold),
                ]
                save_path = save_tranad_metrics(
                    results_dir / f"{dataset_name}_tranad_metrics.csv",
                    metric_items,
                )
                print(f"[Evaluation] Metrics saved to {save_path}")
                try:
                    log_metrics(
                        {f"eval/{k}": float(v) for k, v in metric_items},
                        step=int(cfg.get("simulation", {}).get("total_rounds", 0)),
                    )
                    log_metrics(
                        {"post/infer_time_s": float(eval_dt)},
                        step=int(cfg.get("simulation", {}).get("total_rounds", 0)),
                    )
                except Exception:
                    pass
        except Exception as exc:
            print(f"[Evaluation] Failed to compute metrics: {exc}")

    final_model.to("cpu")

    wandb_finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
