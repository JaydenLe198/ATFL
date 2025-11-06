"""Flower client application for ATFL.

This module wraps the Anomaly Transformer in a Flower `NumPyClient`.
Implementation mirrors the DualTF-FLsim client but keeps only the
time-domain model. FedProx and SCAFFOLD hooks are prepared but require
the training routine in ``task.py`` to be filled in.
"""

from __future__ import annotations

import gc
import io
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from fl.task import (
    build_model,
    get_weights,
    set_weights,
    train,
    load_data,
)
from fl.utils.config import load_config


def _unpack_bytes_to_ndarrays(data: Optional[bytes]) -> Optional[List[np.ndarray]]:
    if not isinstance(data, (bytes, bytearray)):
        return None
    try:
        buf = io.BytesIO(data)
        with np.load(buf) as npz:
            return [npz[k] for k in sorted(npz.files, key=lambda x: int(x[1:]))]
    except Exception:
        return None


class ATFLClient(NumPyClient):
    """Federated client for the Anomaly Transformer."""

    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: torch.utils.data.DataLoader,
        valloader: Optional[torch.utils.data.DataLoader],
        partition_id: int,
    ) -> None:
        super().__init__()
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.partition_id = partition_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            current_gpu = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_gpu)
            print(f"[Client {partition_id}] using GPU cuda:{current_gpu} ({device_name})")
        else:
            print(f"[Client {partition_id}] using CPU")

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        cfg = load_config(os.environ.get("ATFL_CONFIG"))
        train_cfg = cfg.get("training", {})

        local_epochs = int(config.get("local_epochs", train_cfg.get("local_epochs", 1)))
        proximal_mu = float(config.get("proximal_mu", train_cfg.get("proximal_mu", 0.0)))
        lr = float(config.get("lr", train_cfg.get("lr", 1e-4)))

        control_c = _unpack_bytes_to_ndarrays(config.get("c_bytes"))
        control_ci = _unpack_bytes_to_ndarrays(config.get("ci_bytes"))

        set_weights(self.net, parameters)

        start_time = time.time()
        start_mem = None
        peak_mem = None
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
                start_mem = torch.cuda.memory_allocated(self.device)
            except Exception:
                start_mem = None

        result = train(
            net=self.net,
            trainloader=self.trainloader,
            valloader=self.valloader,
            epochs=local_epochs,
            device=self.device,
            proximal_mu=proximal_mu,
            lr=lr,
            control_c=control_c,
            control_ci=control_ci,
        )
        train_loss, delta_ci = result.train_loss, result.delta_ci

        duration = time.time() - start_time
        if torch.cuda.is_available():
            try:
                peak_mem = torch.cuda.max_memory_allocated(self.device)
            except Exception:
                peak_mem = None

        # Move model back to CPU and free VRAM.
        try:
            self.net.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()

        metrics: Dict[str, Any] = {
            "train_loss": float(train_loss),
            "train_time_s": float(duration),
        }
        if start_mem is not None and peak_mem is not None:
            metrics["gpu_mem_alloc_start"] = float(start_mem)
            metrics["gpu_mem_alloc_peak"] = float(peak_mem)

        if delta_ci is not None:
            buf = io.BytesIO()
            np.savez_compressed(buf, **{f"a{i}": a for i, a in enumerate(delta_ci)})
            metrics["delta_ci"] = buf.getvalue()

        num_examples = len(self.trainloader.dataset)
        return get_weights(self.net), num_examples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        # Evaluation happens centrally after the federated rounds.
        return 0.0, 0, {"eval": "client_eval_disabled"}


def client_fn(context: Context):
    """Flower client factory."""
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

    cfg = load_config(os.environ.get("ATFL_CONFIG"))
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    seq_len = int(model_cfg.get("seq_len", 100))
    model_kwargs = {
        "win_size": seq_len,
        "enc_in": int(model_cfg.get("enc_in", 25)),
        "c_out": int(model_cfg.get("c_out", 25)),
        "d_model": int(model_cfg.get("d_model", 512)),
        "n_heads": int(model_cfg.get("n_heads", 8)),
        "e_layers": int(model_cfg.get("e_layers", 3)),
    }

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    loaders = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=int(data_cfg.get("batch_size", 64)),
        seq_length=int(data_cfg.get("seq_length", seq_len)),
    )

    trainloader = loaders.train
    valloader = loaders.val

    net = build_model(model_kwargs)

    return ATFLClient(net, trainloader, valloader, partition_id).to_client()


app = ClientApp(client_fn)
