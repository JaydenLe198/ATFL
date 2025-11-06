"""Centralized dataset cache utilities (optional)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

CACHE_TRAIN_SEGMENTS = "train_segments.npy"
CACHE_TEST_SEGMENTS = "test_segments.npy"
CACHE_LABEL_SEGMENTS = "label_segments.npy"
CACHE_TRAIN_WINDOWS = "train_windows.npy"
CACHE_TEST_WINDOWS = "test_windows.npy"
CACHE_LABEL_WINDOWS = "label_windows.npy"
CACHE_SLICES = "client_slices.npy"
CACHE_META = "cache_meta.json"

from fl.utils.data_loader import (
    _create_sequences,
    load_dataset_by_name,
    sequential_partition,
)


def build_central_cache(
    dataset: str,
    seq_length: int,
    stride: int,
    cache_dir: str,
    num_clients: int,
) -> Dict[str, Any]:
    """Persist raw dataset slices for reuse across simulations."""
    os.makedirs(cache_dir, exist_ok=True)
    data_dict = load_dataset_by_name(dataset)
    train_segments = data_dict["train_segments"]
    test_segments = data_dict["test_segments"]
    label_segments = data_dict["label_segments"]

    np.save(
        os.path.join(cache_dir, CACHE_TRAIN_SEGMENTS),
        np.array(train_segments, dtype=object),
        allow_pickle=True,
    )
    np.save(
        os.path.join(cache_dir, CACHE_TEST_SEGMENTS),
        np.array(test_segments, dtype=object),
        allow_pickle=True,
    )
    np.save(
        os.path.join(cache_dir, CACHE_LABEL_SEGMENTS),
        np.array(label_segments, dtype=object),
        allow_pickle=True,
    )

    train_windows_list = [_create_sequences(seg, seq_length, stride) for seg in train_segments]
    test_windows_list = [_create_sequences(seg, seq_length, stride) for seg in test_segments]
    label_windows_list = [_create_sequences(seg, seq_length, stride) for seg in label_segments]

    def _concat_or_empty(arrs: List[np.ndarray], feature_dim: int) -> np.ndarray:
        non_empty = [a for a in arrs if a.shape[0] > 0]
        if non_empty:
            return np.concatenate(non_empty, axis=0)
        return np.empty((0, seq_length, feature_dim), dtype=np.float32)

    feature_dim = int(train_segments[0].shape[1]) if train_segments else 0
    train_concat = _concat_or_empty(train_windows_list, feature_dim)
    test_concat = _concat_or_empty(test_windows_list, feature_dim)
    label_feature_dim = 1 if label_windows_list and label_windows_list[0].ndim == 3 else feature_dim
    label_concat = (
        np.concatenate([a for a in label_windows_list if a.shape[0] > 0], axis=0)
        if any(a.shape[0] > 0 for a in label_windows_list)
        else np.empty((0, seq_length, label_feature_dim), dtype=np.float32)
    )

    np.save(os.path.join(cache_dir, CACHE_TRAIN_WINDOWS), train_concat)
    np.save(os.path.join(cache_dir, CACHE_TEST_WINDOWS), test_concat)
    np.save(os.path.join(cache_dir, CACHE_LABEL_WINDOWS), label_concat)

    client_slices: List[Tuple[int, int]] = []
    if num_clients and num_clients > 0 and train_concat.shape[0] > 0:
        lengths = [arr.shape[0] for arr in train_windows_list]
        slices = sequential_partition(lengths, num_clients)
        for sl in slices:
            start = int(sl.start or 0)
            stop = int(sl.stop or 0)
            client_slices.append((start, stop))
        np.save(os.path.join(cache_dir, CACHE_SLICES), np.array(client_slices, dtype=np.int64))

    meta = {
        "dataset": dataset,
        "seq_length": seq_length,
        "stride": stride,
        "num_train_segments": len(train_segments),
        "train_segment_lengths": [int(seg.shape[0]) for seg in train_segments],
        "num_test_segments": len(test_segments),
        "test_segment_lengths": [int(seg.shape[0]) for seg in test_segments],
        "feature_dim": feature_dim,
        "num_clients": int(num_clients),
        "has_client_slices": bool(client_slices),
        "train_windows": int(train_concat.shape[0]),
        "test_windows": int(test_concat.shape[0]),
    }

    with open(os.path.join(cache_dir, CACHE_META), "w", encoding="utf-8") as fout:
        json.dump(meta, fout, indent=2)

    return meta


def load_cache(cache_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(cache_dir, CACHE_META)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Cache metadata not found at {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as fin:
        return json.load(fin)
