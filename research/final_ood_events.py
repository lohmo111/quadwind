"""S5/S4 для финалистов на всех сидах: OOD-наборы и сценарии событий. Кэш на запуск в research/cache/ood_*.csv.

    python research/final_ood_events.py --part 0 --parts 2     # можно делить список между процессами
    python research/final_ood_events.py --summary              # сводка по группам (среднее [мин–макс] по сидам)
"""
import argparse, glob, json, sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quadwind.conditions import OOD_SPACES, make_suite  # noqa: E402
from quadwind.control import PIDGains  # noqa: E402
from quadwind.env import EnvConfig  # noqa: E402
from quadwind.evaluation import evaluate_in_chunks  # noqa: E402
from quadwind.metrics import episode_metrics  # noqa: E402
from quadwind.policies import PIDPolicy, SB3Policy  # noqa: E402
from quadwind.training import load_model  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"
SUITES = ["test_id", *OOD_SPACES, "step_gust", "gust_pulse", "mass_pickup", "mass_drop", "push", "motor_degrade"]
GROUPS = {
    "MLP (asym)": "ppo_mlp__pi__safe__asym",
    "MLP-k16 (asym)": "ppo_stack16__pi__safe__asym",
    "MLP+∫": "ppo_mlp_integral__pi__safe",
    "MLP+∫ (asym)": "ppo_mlp_integral__pi__safe__asym",
    "MLP+∫ (attitude)": "ppo_mlp_integral__pi__safe__att",
    "Residual MLP": "ppo_mlp__pi__safe__res",
    "RPPO": "rppo__pi__safe",
    "RPPO (рецепт)": "rppo__pi__safe__asym__hp_epochs5__hp_lstm128",
    "RPPO (attitude)": "rppo__pi__safe__att__hp_epochs5",
    "RPPO (attitude, рецепт)": "rppo__pi__safe__att__asym__hp_epochs5__hp_lstm128",
    "RPPO+∫ (attitude, рецепт)": "rppo_integral__pi__safe__att__asym__hp_epochs5__hp_lstm128",
    "MLP+∫ (attitude, ∫ ±2)": "ppo_mlp_integral__pi__safe__att__int2",
    "RPPO+∫ (attitude, рецепт, ∫ ±2)": "rppo_integral__pi__safe__att__asym__hp_epochs5__hp_lstm128__int2",
    "MLP+∫ (attitude, 5 эпох, асим.)": "ppo_mlp_integral__pi__safe__att__asym__hp_epochs5",
    "MLP+∫ (attitude, 5 эпох)": "ppo_mlp_integral__pi__safe__att__hp_epochs5",
}
PI = (100.0, 100.0, 50.0)


def policies():
    out = [("PID", "pid_ctbr", lambda: (PIDPolicy(EnvConfig(rate_ki=PI), PIDGains(
               **json.load(open("configs/pid_tuned_pi.json")))), EnvConfig(rate_ki=PI))),
           ("PID (attitude)", "pid_att", lambda: (PIDPolicy(EnvConfig(rate_ki=PI, action_mode="attitude"), PIDGains(
               **json.load(open("configs/pid_tuned_pi_att.json")))), EnvConfig(rate_ki=PI, action_mode="attitude")))]
    for group, name in GROUPS.items():
        for run in sorted(glob.glob(f"runs/e7/{name}/seed_*")):
            if Path(run, "final_model.zip").exists():
                def make(run=run):
                    m, n, c = load_model(Path(run), "best")
                    return SB3Policy(m, n), EnvConfig.from_dict(c.get("env"))
                out.append((group, f"{name}__{Path(run).name}", make))
    return out


def episodes(key, make):
    path = CACHE / f"ood_{key}.csv"
    if path.exists():
        return pd.read_csv(path)
    policy, ec = make()
    frames = []
    for suite in SUITES:
        for ro in evaluate_in_chunks(policy, ec, make_suite(suite)):
            df = episode_metrics(ro)
            df["suite"] = suite
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(path, index=False)
    return df


def summary():
    rows = []
    for group, key, make in policies():
        path = CACHE / f"ood_{key}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for suite, d in df.groupby("suite"):
            ok = d[~d.crashed]
            rows.append({"group": group, "run": key, "suite": suite, "med_cm": 100 * ok.rmse.median(),
                         "crash%": 100 * d.crashed.mean(), "peak_cm": 100 * d.event_peak.median(),
                         "rec_s": d.event_recovery_time.median()})
    t = pd.DataFrame(rows)
    t.to_csv("research/final_ood_events_runs.csv", index=False)
    agg = t.groupby(["group", "suite"], sort=False).agg(
        med_cm=("med_cm", "mean"), crash=("crash%", "mean"), crash_max=("crash%", "max"),
        peak_cm=("peak_cm", "mean"), rec_s=("rec_s", "mean"), seeds=("run", "size")).reset_index()
    agg.to_csv("research/final_ood_events.csv", index=False)
    pd.set_option("display.width", 300)
    ood = agg[agg.suite.str.startswith("ood") | (agg.suite == "test_id")]
    cell = ood.assign(c=lambda d: d.med_cm.round(1).astype(str) + " / " + d.crash.round(1).astype(str)
                      + "% (≤" + d.crash_max.round(0).astype(int).astype(str) + ")")
    print("OOD: медиана выживших, см / падения, % (среднее по сидам, в скобках — худший сид)")
    print(cell.pivot(index="suite", columns="group", values="c").to_string())
    ev = agg[~(agg.suite.str.startswith("ood") | (agg.suite == "test_id"))]
    cell = ev.assign(c=lambda d: d.peak_cm.round(1).astype(str) + " / " + d.rec_s.round(2).astype(str))
    print("\nСобытия: пик, см / восстановление до 5 см, с (среднее по сидам медиан)")
    print(cell.pivot(index="suite", columns="group", values="c").to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        return summary()
    torch.set_num_threads(2)
    CACHE.mkdir(exist_ok=True)
    for i, (group, key, make) in enumerate(policies()):
        if i % args.parts == args.part:
            episodes(key, make)
            print("done", key, flush=True)


if __name__ == "__main__":
    main()
