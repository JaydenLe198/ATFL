"""TranAD-style evaluation for ATFL time arrays."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from fl.utils.evaluation import get_threshold_tranad, save_tranad_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ATFL results (TranAD-style).")
    parser.add_argument("--dataset", type=str, default="PSM")
    parser.add_argument("--data_num", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="eval_results")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_time_array(dataset: str, data_num: int) -> pd.DataFrame:
    path = Path("time_arrays") / f"{dataset}_{data_num}_time_evaluation_array.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Time array not found at {path}")
    return pd.read_pickle(path)


def main() -> None:
    args = _parse_args()
    time_array = load_time_array(args.dataset, args.data_num)
    if args.verbose:
        print("Time array head:")
        print(time_array.iloc[:, :10])

    scores = np.asarray(time_array.loc["Avg(RE)", :]).astype(float)
    labels = np.asarray(time_array.loc["GT", :]).astype(int)

    results = get_threshold_tranad(labels, scores, verbose=True)
    (
        auc_val,
        precision,
        recall,
        f1,
        precision_adjusted,
        recall_adjusted,
        f1_adjusted,
        threshold,
    ) = results

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("auc", auc_val),
        ("precision", precision),
        ("recall", recall),
        ("f1", f1),
        ("precision_adjusted", precision_adjusted),
        ("recall_adjusted", recall_adjusted),
        ("f1_adjusted", f1_adjusted),
        ("threshold", threshold),
    ]
    save_tranad_metrics(out_dir / f"{args.dataset}_{args.data_num}_tranad_metrics.csv", metrics)
    print(f"Saved evaluation metrics to {out_dir}/{args.dataset}_{args.data_num}_tranad_metrics.csv")


if __name__ == "__main__":
    main()
