"""Produce the per-controller, per-seed results used in the defense."""
import argparse
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import binomtest


def archived_backend(run):
    """Older CSVs record the checkpoint folder using the producer's path separator."""
    if run.split("__")[0] in ("pid", "pd"):
        return "baseline"
    config = Path(__file__).resolve().parents[1] / run.replace("\\", "/") / "config.yaml"
    if not config.is_file():
        return "unknown"
    return yaml.safe_load(config.read_text(encoding="utf-8")).get("backend", "sb3")


def summarize(episodes):
    rows = []
    episodes = episodes.copy()
    if "backend" not in episodes:
        episodes["backend"] = pd.NA
    missing = episodes.backend.isna()
    if missing.any():
        runs = episodes.loc[missing, "run"]
        episodes.loc[missing, "backend"] = runs.map({run: archived_backend(run) for run in runs.unique()})
    keys = ["namespace", "environment_group", "suite", "method", "backend", "seed_train"]
    if "checkpoint" in episodes:
        episodes["checkpoint"] = episodes.checkpoint.fillna("unrecorded")
        keys.append("checkpoint")
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
    parser.add_argument("--out", type=Path, help="Defaults to summary.csv next to the input episodes.")
    args = parser.parse_args()
    result = summarize(pd.read_csv(args.csv))
    output = args.out or args.csv.with_name("summary.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, float_format="%.6f")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
