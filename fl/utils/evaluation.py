"""TranAD-style evaluation helpers shared by training/evaluation scripts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import sklearn.metrics


def adjust_predicts_from_tranad(label, score, threshold=None, pred=None, calc_latency=False):
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred.copy()
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    return predict


def get_threshold_tranad(labels, scores, verbose: bool = True):
    auc_val = sklearn.metrics.roc_auc_score(labels, scores)
    thresholds_sorted = np.asarray(scores).copy()
    thresholds_sorted.sort()

    thresholds = []
    for i in range(len(thresholds_sorted)):
        if i % 1000 == 0 or i == len(thresholds_sorted) - 1:
            thresholds.append(thresholds_sorted[i])

    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_threshold = math.inf
    best_f1_adjusted = 0.0
    best_precision_adjusted = 0.0
    best_recall_adjusted = 0.0

    labels_np = np.asarray(labels).astype(int)
    scores_np = np.asarray(scores).astype(float)

    for threshold in thresholds:
        y_pred = (scores_np >= threshold).astype(int)
        precision = sklearn.metrics.precision_score(labels_np, y_pred, zero_division=0)
        recall = sklearn.metrics.recall_score(labels_np, y_pred, zero_division=0)
        f1 = sklearn.metrics.f1_score(labels_np, y_pred, zero_division=0)

        y_pred_adjusted = adjust_predicts_from_tranad(labels_np, scores_np, pred=y_pred, threshold=threshold)
        precision_adjusted = sklearn.metrics.precision_score(labels_np, y_pred_adjusted, zero_division=0)
        recall_adjusted = sklearn.metrics.recall_score(labels_np, y_pred_adjusted, zero_division=0)
        f1_adjusted = sklearn.metrics.f1_score(labels_np, y_pred_adjusted, zero_division=0)

        if f1_adjusted > best_f1_adjusted:
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            best_f1_adjusted = f1_adjusted
            best_precision_adjusted = precision_adjusted
            best_recall_adjusted = recall_adjusted
            best_threshold = threshold

    if verbose:
        print("auc:", auc_val)
        print("precision_adjusted:", best_precision_adjusted)
        print("recall_adjusted:", best_recall_adjusted)
        print("f1:", best_f1)
        print("f1_adjusted:", best_f1_adjusted)
        print("threshold:", best_threshold)

    return (
        auc_val,
        best_precision,
        best_recall,
        best_f1,
        best_precision_adjusted,
        best_recall_adjusted,
        best_f1_adjusted,
        best_threshold,
    )


def save_tranad_metrics(
    path: Path,
    metrics: Iterable[Tuple[str, float]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([dict(metrics)])
    df.to_csv(path, index=False)
    return path


__all__ = ["adjust_predicts_from_tranad", "get_threshold_tranad", "save_tranad_metrics"]
