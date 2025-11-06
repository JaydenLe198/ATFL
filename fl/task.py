"""Core training utilities for ATFL.

This file should host the adaptations of the centralized Anomaly Transformer
training loop to the federated setting (FedProx/SCAFFOLD).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from fl.model.TimeTransformer import AnomalyTransformer
from fl.utils.data_loader import (
    FederatedLoaders,
    WindowDataset,
    load_client_data,
    load_dataset_by_name,
    _create_sequences,
)


@dataclass
class TrainResult:
    train_loss: float
    delta_ci: Optional[List[np.ndarray]]


def build_model(model_kwargs: Dict[str, Any]) -> torch.nn.Module:
    """Instantiate the Anomaly Transformer with the provided hyper-parameters."""
    net = AnomalyTransformer(**model_kwargs)
    return net


def get_weights(net: torch.nn.Module) -> List[np.ndarray]:
    """Return model parameters as numpy arrays."""
    return [p.detach().cpu().numpy() for _, p in net.state_dict().items()]


def set_weights(net: torch.nn.Module, parameters: Iterable[np.ndarray]) -> None:
    """Load numpy weights into the model."""
    state_dict = net.state_dict()
    for key, array in zip(state_dict.keys(), parameters):
        state_dict[key] = torch.tensor(array)
    net.load_state_dict(state_dict, strict=True)


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    seq_length: int,
) -> FederatedLoaders:
    """Load train/val/test loaders for a specific client."""
    return load_client_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        seq_length=seq_length,
    )


def train(
    net: torch.nn.Module,
    trainloader: DataLoader,
    valloader: Optional[DataLoader],
    epochs: int,
    device: torch.device,
    proximal_mu: float,
    lr: float,
    control_c: Optional[List[np.ndarray]],
    control_ci: Optional[List[np.ndarray]],
) -> TrainResult:
    """Train the model locally with optional FedProx and SCAFFOLD."""
    if epochs <= 0 or len(trainloader.dataset) == 0:
        weights = [p.detach().cpu().numpy().copy() for p in net.parameters()]
        return TrainResult(train_loss=0.0, delta_ci=None)

    net.to(device)
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    global_params = [p.detach().clone().to(device) for p in net.parameters()]
    params = list(net.parameters())

    if control_c is not None and control_ci is not None:
        c_tensors = [torch.tensor(a, device=device, dtype=torch.float32) for a in control_c]
        ci_tensors = [torch.tensor(a, device=device, dtype=torch.float32) for a in control_ci]
    else:
        c_tensors = ci_tensors = None

    w_before = [p.detach().cpu().numpy().copy() for p in net.parameters()]

    total_loss_acc = 0.0
    steps = 0
    successful_steps = 0
    k = 3.0

    for epoch in range(epochs):
        for batch in trainloader:
            if isinstance(batch, (list, tuple)):
                input_data = batch[0]
            else:
                input_data = batch
            input_data = input_data.float().to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                output, series, prior, _ = net(input_data)
                rec_loss = criterion(output, input_data)
                batch_size = input_data.shape[0]
                series_acc = torch.zeros(batch_size, device=device)
                prior_acc = torch.zeros(batch_size, device=device)
                for u in range(len(prior)):
                    den = torch.sum(prior[u], dim=-1, keepdim=True) + 1e-12
                    prior_norm = prior[u] / den
                    prior_norm = torch.nan_to_num(prior_norm, nan=0.0, posinf=0.0, neginf=0.0)
                    series_u = torch.nan_to_num(series[u], nan=0.0, posinf=0.0, neginf=0.0)
                    series_acc += my_kl_loss(series_u, prior_norm.detach()) + my_kl_loss(prior_norm.detach(), series_u)
                    prior_acc += my_kl_loss(prior_norm, series_u.detach()) + my_kl_loss(series_u.detach(), prior_norm)
                if len(prior) > 0:
                    series_loss = series_acc.mean() / len(prior)
                    prior_loss = prior_acc.mean() / len(prior)
                else:
                    series_loss = torch.tensor(0.0, device=device)
                    prior_loss = torch.tensor(0.0, device=device)

                loss1 = rec_loss - k * series_loss
                loss2 = rec_loss + k * prior_loss
                total_loss = (loss1 + loss2).mean()

                if proximal_mu > 0:
                    proximal_term = 0.0
                    for local_param, global_param in zip(params, global_params):
                        proximal_term += (local_param - global_param).pow(2).sum()
                    total_loss = total_loss + (proximal_mu / 2.0) * proximal_term

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)

            # Skip step if gradients contain NaNs/Infs
            if any(p.grad is not None and (not torch.isfinite(p.grad).all()) for p in params):
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                steps += 1
                continue

            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

            if c_tensors is not None and ci_tensors is not None:
                for p, c_k, ci_k in zip(params, c_tensors, ci_tensors):
                    if p.grad is not None:
                        p.grad.add_(c_k - ci_k)

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            if scaler.get_scale() >= scale_before:
                successful_steps += 1
                total_loss_acc += float(total_loss.detach().cpu())
            optimizer.zero_grad(set_to_none=True)

    train_loss = total_loss_acc / max(1, successful_steps)

    delta_ci = None
    if c_tensors is not None and ci_tensors is not None and successful_steps > 0:
        w_after = [p.detach().cpu().numpy().copy() for p in net.parameters()]
        inv = 1.0 / (float(successful_steps) * float(lr))
        deltas: List[np.ndarray] = []
        for wb, wa, c_np, ci_np in zip(w_before, w_after, control_c, control_ci):
            delta = (-c_np + inv * (wb - wa)).astype(wb.dtype, copy=False)
            deltas.append(delta)
        delta_ci = deltas

    return TrainResult(train_loss=float(train_loss), delta_ci=delta_ci)


def load_centralized_test_data(
    dataset_name: str,
    batch_size: int,
    seq_length: int,
    step: int,
) -> DataLoader:
    """Build a DataLoader covering the full test set for centralized evaluation."""
    data_dict = load_dataset_by_name(dataset_name)
    test_segments = data_dict["test_segments"]
    label_segments = data_dict["label_segments"]
    feature_dim = int(test_segments[0].shape[1]) if test_segments else 0

    test_windows_list = [_create_sequences(seg, seq_length, step) for seg in test_segments]
    label_windows_list = []
    for lbl_seg in label_segments:
        lbl = lbl_seg
        if lbl.ndim == 1:
            lbl = lbl.reshape(-1, 1)
        label_windows_list.append(_create_sequences(lbl, seq_length, step))

    non_empty_test = [w for w in test_windows_list if w.shape[0] > 0]
    if non_empty_test:
        test_concat = np.concatenate(non_empty_test, axis=0)
    else:
        test_concat = np.empty((0, seq_length, feature_dim), dtype=np.float32)

    non_empty_label = [w for w in label_windows_list if w.shape[0] > 0]
    if non_empty_label:
        label_concat = np.concatenate(non_empty_label, axis=0)
    else:
        label_concat = np.zeros((test_concat.shape[0], seq_length, 1), dtype=np.float32)

    dataset = WindowDataset(test_concat, labels=label_concat)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )


def get_anomaly_scores(
    net: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    k: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute anomaly scores and labels over a dataset."""
    net.to(device)
    net.eval()

    criterion = nn.MSELoss(reduction="none")
    scores: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    use_amp = device.type == "cuda"
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                input_data = batch[0]
                targets = batch[1]
            else:
                input_data = batch
                targets = torch.zeros((input_data.shape[0], input_data.shape[1]), dtype=torch.float32)
            input_data = input_data.float().to(device)
            with autocast(enabled=use_amp):
                output, series, prior, _ = net(input_data)

            rec_loss = criterion(output, input_data).mean(dim=(1, 2))
            series_loss = torch.zeros(rec_loss.shape[0], device=device)
            for u in range(len(prior)):
                den = torch.sum(prior[u], dim=-1, keepdim=True) + 1e-12
                prior_norm = prior[u] / den
                prior_norm = torch.nan_to_num(prior_norm, nan=0.0, posinf=0.0, neginf=0.0)
                series_u = torch.nan_to_num(series[u], nan=0.0, posinf=0.0, neginf=0.0)
                series_loss += my_kl_loss(series_u, prior_norm.detach()) + my_kl_loss(prior_norm.detach(), series_u)
            if len(prior) > 0:
                series_loss /= len(prior)
            anomaly_score = rec_loss + k * series_loss
            scores.append(anomaly_score.detach().cpu().numpy())

            if isinstance(targets, torch.Tensor):
                t = targets
            else:
                t = torch.from_numpy(targets)
            if t.dim() >= 3:
                t = t.squeeze(-1)
            if t.dim() >= 2:
                lbl = (t.to(dtype=torch.float32).max(dim=1).values > 0.5).to(dtype=torch.int64)
            else:
                lbl = (t.to(dtype=torch.float32) > 0.5).to(dtype=torch.int64)
            labels.append(lbl.cpu().numpy())

    if not scores:
        raise RuntimeError("No scores were collected from the dataloader.")

    return np.concatenate(scores), np.concatenate(labels)


def my_kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    p = p.to(dtype=torch.float32)
    q = q.to(dtype=torch.float32)
    p_sum = torch.sum(p, dim=-1, keepdim=True)
    q_sum = torch.sum(q, dim=-1, keepdim=True)
    p = p / (p_sum + eps)
    q = q / (q_sum + eps)
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    res = p * (torch.log(p) - torch.log(q))
    res = torch.sum(res, dim=-1)      # (B, H, L)
    res = torch.mean(res, dim=1)      # (B, L)
    return torch.mean(res, dim=1)     # (B)


__all__ = [
    "TrainResult",
    "build_model",
    "get_weights",
    "set_weights",
    "load_data",
    "train",
    "load_centralized_test_data",
    "get_anomaly_scores",
    "my_kl_loss",
]
