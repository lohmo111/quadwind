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
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quadwind.conditions import OOD_SPACES, SCENARIO_SPACES, make_suite, suite_names  # noqa: E402
from quadwind.control import PIDGains  # noqa: E402
from quadwind.env import EnvConfig  # noqa: E402
from quadwind.evaluation import evaluate_in_chunks  # noqa: E402
from quadwind.metrics import episode_metrics  # noqa: E402
from quadwind.policies import PIDPolicy, SB3Policy  # noqa: E402
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


def evaluate_policy(policy, env_cfg: EnvConfig, suites: list[str], tags: dict) -> pd.DataFrame:
    frames = []
    for suite in suites:
        for ro in evaluate_in_chunks(policy, env_cfg, make_suite(suite)):
            df = episode_metrics(ro, max_dist=env_cfg.max_dist)
            df.insert(0, "suite", suite)
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for k, v in reversed(list(tags.items())):
        df.insert(0, k, v)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="папки запусков или glob-шаблоны")
    ap.add_argument("--suites", nargs="+", default=["val"])
    ap.add_argument("--which", default="best", choices=["best", "final"])
    ap.add_argument("--baselines", nargs="*", default=[], choices=["pid", "pd"])
    ap.add_argument("--pid-gains", default=None, help="JSON с коэффициентами PID (из tune_pid.py)")
    ap.add_argument("--out", default="results/episodes.csv")
    args = ap.parse_args()

    suites = expand_suites(args.suites)
    frames = []
    gains = PIDGains(**json.loads(Path(args.pid_gains).read_text())) if args.pid_gains else PIDGains()
    for name in args.baselines:
        g = gains if name == "pid" else gains.without_integrators()
        env_cfg = EnvConfig()
        frames.append(evaluate_policy(PIDPolicy(env_cfg, g), env_cfg, suites,
                                      {"method": name, "seed_train": 0, "run": name}))
        print(f"evaluated {name}")

    run_dirs = sorted({Path(p) for pattern in args.runs for p in glob.glob(pattern)})
    for run_dir in run_dirs:
        if not (run_dir / f"{args.which}_model.zip").exists():
            print(f"skip {run_dir}: no {args.which}_model.zip")
            continue
        model, normalizer, cfg = load_model(run_dir, args.which)
        env_cfg = EnvConfig.from_dict(cfg.get("env"))
        seed = int(run_dir.name.split("_")[-1]) if run_dir.name.startswith("seed_") else 0
        frames.append(evaluate_policy(SB3Policy(model, normalizer), env_cfg, suites,
                                      {"method": cfg["name"], "seed_train": seed, "run": str(run_dir)}))
        print(f"evaluated {run_dir}")

    if not frames:
        raise SystemExit("nothing to evaluate")
    df = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    crash = df.groupby(["suite", "method"]).crashed.mean().rename("crash_rate")
    table = summarize(df, "rmse_capped").set_index(["suite", "method"])[["value", "ci_low", "ci_high", "n_seeds"]]
    print(table.join(crash).round(4).to_string())
    print(f"\n{len(df)} episodes -> {out}")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
