"""Оценка обученных моделей и классических baseline на фиксированных наборах условий.

Пример:
    python scripts/evaluate.py --runs "runs/main/*/seed_*" --suites test_id ood scenarios \
        --baselines pid pd --pid-gains configs/pid_tuned.json --out results/main.csv
Группы наборов: ood (все ood_*), scenarios (сценарии + wind_sweep), all.
На выходе — CSV «один эпизод = одна строка» (метрики + параметры условия) и сводка в консоли.
Перед финальной оценкой на test-наборах зафиксируйте конфиги: подбор — только на val.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quadwind.conditions import OOD_SPACES, SCENARIO_SPACES, make_suite, suite_names  # noqa: E402
from quadwind.env import EnvConfig  # noqa: E402
from quadwind.evaluation import evaluate_in_chunks  # noqa: E402
from quadwind.metrics import episode_metrics  # noqa: E402
from quadwind.policies import PIDPolicy, SB3Policy, baseline_env_config, tuned_pid_gains  # noqa: E402
from quadwind.stats import summarize  # noqa: E402
from quadwind.training import load_model  # noqa: E402

GROUPS = {
    "ood": list(OOD_SPACES),
    "scenarios": [s for s in SCENARIO_SPACES] + ["wind_sweep"],
    "all": suite_names(),
}


def expand_suites(names: list[str]) -> list[str]:
    out = []
    for n in names:
        out += GROUPS.get(n, [n])
    return list(dict.fromkeys(out))


def evaluate_policy(policy, env_cfg: EnvConfig, suites: list[str], tags: dict,
                    namespace: str = "research") -> pd.DataFrame:
    frames = []
    for suite in suites:
        offset = 0
        for ro in evaluate_in_chunks(policy, env_cfg, make_suite(suite, namespace=namespace)):
            df = episode_metrics(ro, max_dist=env_cfg.max_dist)
            df["episode"] += offset
            offset += len(df)
            df.insert(0, "suite", suite)
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["namespace"] = namespace
    for k, v in reversed(list(tags.items())):
        df.insert(0, k, v)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="папки запусков или glob-шаблоны")
    ap.add_argument("--suites", nargs="+", default=["val"])
    ap.add_argument("--namespace", choices=["research", "final"], default="research",
                    help="independent episode seeds for the final evaluation")
    ap.add_argument("--which", default="best", choices=["initial", "early", "best", "final"])
    ap.add_argument("--baselines", nargs="*", default=[], choices=["pid", "pd"])
    ap.add_argument("--pid-gains", default=None, help="JSON с коэффициентами PID (из tune_pid.py)")
    ap.add_argument("--out", default="results/episodes.csv")
    args = ap.parse_args()

    suites = expand_suites(args.suites)
    frames = []
    baseline_configs = []
    run_dirs = set()
    for pattern in args.runs:
        matches = glob.glob(pattern)
        if not matches:
            ap.error(f"no runs match {pattern!r}")
        for p in matches:
            run = Path(p)
            if not (run / f"{args.which}_model.zip").is_file():
                if not glob.has_magic(pattern):
                    ap.error(f"{run}: missing {args.which}_model.zip")
                print(f"skip {run}: missing {args.which}_model.zip")
                continue
            run_dirs.add(run)
    if args.runs and not run_dirs:
        ap.error("no loadable checkpoints in the requested runs")
    if run_dirs:
        import torch
        torch.set_num_threads(2)
    for run_dir in sorted(run_dirs):
        model, normalizer, cfg = load_model(run_dir, args.which)
        env_cfg = EnvConfig.from_dict(cfg.get("env"))
        matched_cfg = baseline_env_config(env_cfg)
        if matched_cfg not in baseline_configs:
            baseline_configs.append(matched_cfg)
        group = baseline_configs.index(matched_cfg) + 1
        seed = int(run_dir.name.split("_")[-1]) if run_dir.name.startswith("seed_") else 0
        frames.append(evaluate_policy(SB3Policy(model, normalizer), env_cfg, suites,
                                      {"method": cfg["name"], "backend": cfg.get("backend", "sb3"),
                                       "seed_train": seed, "run": str(run_dir), "checkpoint": args.which,
                                       "environment_group": group}, namespace=args.namespace))
        print(f"evaluated {run_dir}")

    for group, env_cfg in enumerate(baseline_configs or [EnvConfig()], 1):
        for name in args.baselines:
            gains = tuned_pid_gains(env_cfg, args.pid_gains)
            g = gains.without_integrators() if name == "pd" else gains
            label = name if len(baseline_configs) <= 1 else f"{name}__group{group}_{env_cfg.action_mode}"
            frames.append(evaluate_policy(PIDPolicy(env_cfg, g), env_cfg, suites,
                                          {"method": label, "backend": "baseline", "seed_train": 0, "run": label,
                                           "checkpoint": "fixed",
                                           "environment_group": group}, namespace=args.namespace))
            print(f"evaluated {label}")
    if not frames:
        raise SystemExit("nothing to evaluate")
    df = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    groups = ["environment_group", "suite", "method", "backend", "checkpoint"]
    crash = df.groupby(groups).crashed.mean().rename("crash_rate")
    table = summarize(df, "rmse_capped", by=groups).set_index(groups)[["value", "ci_low", "ci_high", "n_seeds"]]
    print(table.join(crash).round(4).to_string())
    print(f"\n{len(df)} episodes -> {out}")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
