"""Produce the per-controller, per-seed results used in the defense."""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


def summarize(episodes):
    rows = []
    keys = ["namespace", "environment_group", "suite", "method", "seed_train"]
    if episodes.duplicated(keys + ["cond_seed"]).any():
        raise ValueError("Duplicate episode seeds within a controller/suite")
    for group, frame in episodes.groupby(keys, sort=True):
        n = len(frame)
        crashes = int(frame.crashed.sum())
        interval = binomtest(crashes, n).proportion_ci(method="wilson")
        rows.append({
            **dict(zip(keys, group)), "episodes": n, "crashes": crashes,
            "median_surviving_rmse_cm": 100 * frame.loc[~frame.crashed, "rmse"].median(),
            "mean_capped_rmse_cm": 100 * frame.rmse_capped.mean(),
            "crash_pct": 100 * crashes / n,
            "crash_ci95_low_pct": 100 * interval.low, "crash_ci95_high_pct": 100 * interval.high,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("results/defense/episodes.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/defense/summary.csv"))
    args = parser.parse_args()
    result = summarize(pd.read_csv(args.csv))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False, float_format="%.6f")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
