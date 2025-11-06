"""Dataset loaders and partitioning helpers for ATFL."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import ast
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from fl.utils.config import load_config


@dataclass
class FederatedLoaders:
    train: DataLoader
    val: Optional[DataLoader]
    test: Optional[DataLoader]


class WindowDataset(Dataset):
    """Dataset returning fixed-length windows (optionally with labels)."""

    def __init__(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> None:
        self.x = np.asarray(data, dtype=np.float32)
        if labels is not None:
            labels_arr = np.asarray(labels, dtype=np.float32)
            if labels_arr.ndim == 2:
                labels_arr = labels_arr[:, :, np.newaxis]
            self.y = labels_arr.astype(np.float32)
        else:
            self.y = None

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int):
        x = torch.from_numpy(self.x[index])
        if self.y is None:
            return x
        y = torch.from_numpy(self.y[index])
        return x, y


def _create_sequences(values: np.ndarray, seq_length: int, step: int) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    windows = []
    for start in range(0, values.shape[0] - seq_length + 1, step):
        windows.append(values[start : start + seq_length])
    if not windows:
        feature_dim = values.shape[1]
        return np.empty((0, seq_length, feature_dim), dtype=values.dtype)
    return np.stack(windows)


def sequential_partition(lengths: List[int], num_clients: int) -> List[slice]:
    """Partition concatenated segments sequentially across clients."""
    total = sum(lengths)
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    base = total // num_clients
    rem = total % num_clients
    slices: List[slice] = []
    cursor = 0
    for cid in range(num_clients):
        take = base + (1 if cid < rem else 0)
        start = cursor
        end = min(total, start + take)
        slices.append(slice(start, end))
        cursor = end
    return slices


def _concat_files_in_dir(dir_path: Path, loader: str = "csv") -> np.ndarray:
    if not dir_path.exists():
        raise FileNotFoundError(f"Missing dataset directory: {dir_path}")
    files = sorted([f for f in dir_path.iterdir() if f.is_file()])
    if not files:
        raise RuntimeError(f"No data files found in {dir_path}")
    arrays = []
    for fp in files:
        if loader == "csv":
            try:
                df = pd.read_csv(fp, header=None)
            except Exception:
                df = pd.read_csv(fp, header=None, delim_whitespace=True)
            arr = df.values.astype(np.float32)
        elif loader == "npy":
            arr = np.load(fp).astype(np.float32)
        else:
            raise ValueError(f"Unsupported loader type: {loader}")
        arrays.append(arr)
    return np.concatenate(arrays, axis=0)


def load_PSM() -> Dict[str, Any]:
    base = Path(__file__).resolve().parents[2] / "fl" / "datasets" / "PSM"
    train_df = pd.read_csv(base / "train.csv").iloc[:, 1:].fillna(method="ffill").values.astype(np.float32)
    test_df = pd.read_csv(base / "test.csv").iloc[:, 1:].fillna(method="ffill").values.astype(np.float32)
    labels = pd.read_csv(base / "test_label.csv")["label"].values.astype(int)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    return {
        "train_segments": [train_scaled],
        "test_segments": [test_scaled],
        "label_segments": [labels.reshape(-1, 1)],
        "scalers": [scaler],
    }


def load_SMD_raw() -> Dict[str, Any]:
    base = Path(__file__).resolve().parents[2] / "fl" / "datasets" / "SMD"
    train_dir = base / "train"
    test_dir = base / "test"
    label_dir = base / "test_label"

    train_segments = []
    test_segments = []
    label_segments = []
    scalers = []

    train_files = sorted(train_dir.iterdir())
    test_files = sorted(test_dir.iterdir())
    label_files = sorted(label_dir.iterdir())

    for trf, tef, lbf in zip(train_files, test_files, label_files):
        train_df = pd.read_csv(trf, header=None).values.astype(np.float32)
        test_df = pd.read_csv(tef, header=None).values.astype(np.float32)
        labels = np.loadtxt(lbf, delimiter=",").astype(int)
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_df)
        test_scaled = scaler.transform(test_df)

        train_segments.append(train_scaled)
        test_segments.append(test_scaled)
        label_segments.append(labels.reshape(-1, 1))
        scalers.append(scaler)

    return {
        "train_segments": train_segments,
        "test_segments": test_segments,
        "label_segments": label_segments,
        "scalers": scalers,
    }


def _load_smap_msl_combined(dataset_name: str = "SMAP") -> Dict[str, Any]:
    dataset_name = dataset_name.upper()
    base = Path(__file__).resolve().parents[2] / "fl" / "datasets" / "SMAP+MSL"
    data_dir = base / "data"
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    anomalies_csv = base / "labeled_anomalies.csv"

    meta = pd.read_csv(anomalies_csv)
    meta = meta[meta["spacecraft"] == dataset_name]

    train_segments = []
    test_segments = []
    label_segments = []
    scalers = []

    for _, row in meta.iterrows():
        chan_id = row["chan_id"]
        train_fp = train_dir / f"{chan_id}.csv"
        test_fp = test_dir / f"{chan_id}.csv"
        if not train_fp.exists() or not test_fp.exists():
            continue
        train_df = pd.read_csv(train_fp).values.astype(np.float32)
        test_df = pd.read_csv(test_fp).values.astype(np.float32)

        # Build label vector from anomaly ranges
        labels = np.zeros(test_df.shape[0], dtype=int)
        label_ranges = row["anomaly_sequences"]
        if isinstance(label_ranges, str) and label_ranges.strip():
            sequences = ast.literal_eval(label_ranges)
            for seq in sequences:
                start, end = seq
                labels[start:end + 1] = 1

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_df)
        test_scaled = scaler.transform(test_df)

        train_segments.append(train_scaled)
        test_segments.append(test_scaled)
        label_segments.append(labels.reshape(-1, 1))
        scalers.append(scaler)

    if not train_segments:
        raise RuntimeError(f"No segments loaded for dataset {dataset_name}")

    return {
        "train_segments": train_segments,
        "test_segments": test_segments,
        "label_segments": label_segments,
        "scalers": scalers,
    }


def load_dataset_by_name(name: str) -> Dict[str, Any]:
    name_u = name.upper()
    if name_u == "PSM":
        return load_PSM()
    if name_u == "SMD":
        return load_SMD_raw()
    if name_u in {"SMAP", "MSL"}:
        return _load_smap_msl_combined(name_u)
    raise ValueError(f"Unsupported dataset: {name}")


def _build_client_windows(
    data_dict: Dict[str, Any],
    partition_id: int,
    num_partitions: int,
    seq_length: int,
    step: int,
    partition_mode: str,
) -> Dict[str, np.ndarray]:
    train_segments = data_dict["train_segments"]
    test_segments = data_dict["test_segments"]
    label_segments = data_dict["label_segments"]

    if partition_mode == "by_machine":
        if partition_id >= len(train_segments):
            raise IndexError(f"partition_id {partition_id} out of range for dataset.")
        train_segment = train_segments[partition_id]
        test_segment = test_segments[partition_id] if partition_id < len(test_segments) else test_segments[0]
        label_segment = label_segments[partition_id] if partition_id < len(label_segments) else label_segments[0]

        train_windows = _create_sequences(train_segment, seq_length, step)
        test_windows = _create_sequences(test_segment, seq_length, step)
        label_windows = _create_sequences(label_segment, seq_length, step)
    else:
        train_windows_list = [_create_sequences(seg, seq_length, step) for seg in train_segments]
        lengths = [w.shape[0] for w in train_windows_list]
        non_empty_train = [w for w in train_windows_list if w.shape[0] > 0]
        if non_empty_train:
            concat_train = np.concatenate(non_empty_train, axis=0)
        else:
            concat_train = np.empty((0, seq_length, train_segments[0].shape[1]), dtype=np.float32)
        slices = sequential_partition(lengths, num_partitions)
        current_slice = slices[partition_id]
        train_windows = concat_train[current_slice]

        test_windows_list = [_create_sequences(seg, seq_length, step) for seg in test_segments]
        label_windows_list = [_create_sequences(seg, seq_length, step) for seg in label_segments]
        non_empty_test = [w for w in test_windows_list if w.shape[0] > 0]
        non_empty_label = [w for w in label_windows_list if w.shape[0] > 0]
        if non_empty_test:
            test_concat = np.concatenate(non_empty_test, axis=0)
        else:
            test_concat = np.empty((0, seq_length, train_segments[0].shape[1]), dtype=np.float32)
        if non_empty_label:
            label_concat = np.concatenate(non_empty_label, axis=0)
        else:
            label_concat = np.empty((0, seq_length, 1), dtype=np.float32)
        test_windows = test_concat
        label_windows = label_concat

    return {
        "train": np.asarray(train_windows, dtype=np.float32),
        "test": np.asarray(test_windows, dtype=np.float32),
        "labels": np.asarray(label_windows, dtype=np.float32),
    }


def load_client_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    seq_length: int,
) -> FederatedLoaders:
    cfg = load_config()
    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("dataset", "PSM")
    partition_mode = str(data_cfg.get("partition_mode", "sequential")).lower()
    step = int(data_cfg.get("step", 1))
    client_load_test = bool(data_cfg.get("client_load_test", False))

    data_dict = load_dataset_by_name(dataset_name)

    feature_dim = data_dict["train_segments"][0].shape[1]
    model_cfg = cfg.get("model", {})
    expected = (int(model_cfg.get("enc_in", feature_dim)), int(model_cfg.get("c_out", feature_dim)))
    if expected[0] != feature_dim or expected[1] != feature_dim:
        raise ValueError(
            f"Model enc_in/c_out ({expected}) mismatch dataset feature dim {feature_dim}. "
            "Update config.model enc_in/c_out to match dataset."
        )
    model_seq_len = int(model_cfg.get("seq_len", seq_length))
    if model_seq_len != seq_length:
        raise ValueError(
            f"Model seq_len ({model_seq_len}) must equal data.seq_length ({seq_length}). "
            "Align config.model.seq_len with data.seq_length."
        )

    window_dict = _build_client_windows(
        data_dict,
        partition_id=partition_id,
        num_partitions=num_partitions,
        seq_length=seq_length,
        step=step,
        partition_mode=partition_mode,
    )

    train_dataset = WindowDataset(window_dict["train"])
    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    if client_load_test and window_dict["test"].size > 0:
        val_dataset = WindowDataset(window_dict["test"], labels=window_dict["labels"])
        valloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )
    else:
        valloader = None

    return FederatedLoaders(train=trainloader, val=valloader, test=None)


__all__ = [
    "FederatedLoaders",
    "WindowDataset",
    "_create_sequences",
    "load_client_data",
    "load_dataset_by_name",
]
