"""Сравнение методов по вариантам среды (эксперимент E7: «насколько среда требует памяти»).

Для каждого запуска лучший чекпоинт оценивается на val своего варианта и на общих сценариях.
PID и PD прогоняются на val каждого варианта.
Пример:
    python scripts/compare_variants.py --runs "runs/e7/*/seed_0" runs/pilot2/rppo/seed_0 runs/pilot2/ppo_mlp/seed_0
Выход: results/<out>/episodes.csv и summary.md (таблицы метод × вариант).
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

from quadwind.conditions import make_suite  # noqa: E402
from quadwind.env import EnvConfig  # noqa: E402
from quadwind.evaluation import rollout  # noqa: E402
from quadwind.metrics import episode_metrics  # noqa: E402
from quadwind.policies import PIDPolicy, SB3Policy, baseline_env_config, tuned_pid_gains  # noqa: E402
from quadwind.training import build_space, load_model  # noqa: E402

SCENARIOS = ("wind_sweep", "step_gust", "mass_pickup")
METHOD_ORDER = ["pd", "pid", "ppo_mlp", "ppo_mlp_integral", "ppo_stack16", "rppo", "ppo_oracle"]


def evaluate(policy, env_cfg, space, tags: dict) -> pd.DataFrame:
    frames = []
    for suite in ("val", *SCENARIOS):
        ro = rollout(policy, env_cfg, make_suite(suite, space=space if suite == "val" else None))
        df = episode_metrics(ro, max_dist=env_cfg.max_dist)
        df.insert(0, "suite", suite)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for k, v in reversed(list(tags.items())):
        df.insert(0, k, v)
    return df


def summarize(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    val = df[df.suite == "val"]
    sweep = df[(df.suite == "wind_sweep") & (df.cond_wind_speed.between(2.0, 3.0))]
    gust = df[df.suite == "step_gust"]
    mass = df[df.suite == "mass_pickup"]

    def pivot(frame, col, agg="mean", scale=1.0):
        t = frame.groupby(["variant", "method"])[col].agg(agg).unstack("method") * scale
        return t[[m for m in METHOD_ORDER if m in t.columns] + [m for m in t.columns if m not in METHOD_ORDER]]

    return {
        "val: rmse_capped, см (падение = 200 см)": pivot(val, "rmse_capped", scale=100),
        "val: доля падений, %": pivot(val, "crashed", scale=100),
        "val: медиана RMSE выживших, см": pivot(val[~val.crashed], "rmse", "median", scale=100),
        "val: статическая ошибка (медиана), см": pivot(val, "ss_err", "median", scale=100),
        "wind_sweep 2–3 м/с: статическая ошибка, см": pivot(sweep, "ss_err", scale=100),
        "step_gust: пик отклонения, см": pivot(gust, "event_peak", scale=100),
        "step_gust: время восстановления (медиана), с": pivot(gust, "event_recovery_time", "median"),
        "mass_pickup: пик отклонения, см": pivot(mass, "event_peak", scale=100),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--which", default="best")
    ap.add_argument("--out", default="results/e7")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted({Path(p) for pattern in args.runs for p in glob.glob(pattern)})
    frames, variants = [], {}
    import torch
    torch.set_num_threads(2)
    seen = set()
    for run_dir in run_dirs:
        if not (run_dir / f"{args.which}_model.zip").exists():
            print(f"skip {run_dir}")
            continue
        model, normalizer, cfg = load_model(run_dir, args.which)
        method, _, suffix = cfg["name"].partition("__")
        variant = suffix or "v0_current"
        env_cfg, space = EnvConfig.from_dict(cfg.get("env")), build_space(cfg)
        plant = baseline_env_config(env_cfg)
        variant_setup = (plant, space)
        if variant in variants and variants[variant] != variant_setup:
            ap.error(f"Variant {variant!r} has different plants or validation distributions. Compare them separately.")
        variants[variant] = variant_setup
        seed = int(run_dir.name.removeprefix("seed_"))
        if (method, variant, seed) in seen:
            ap.error(f"Multiple checkpoints for {method}, {variant}, seed {seed}. Select one experiment at a time.")
        seen.add((method, variant, seed))
        frames.append(evaluate(SB3Policy(model, normalizer), env_cfg, space,
                               {"variant": variant, "method": method, "run": str(run_dir), "seed_train": seed}))
        print(f"evaluated {method} / {variant}")

    for variant, (env_cfg, space) in variants.items():
        gains = tuned_pid_gains(env_cfg)
        for method, g in (("pid", gains), ("pd", gains.without_integrators())):
            frames.append(evaluate(PIDPolicy(env_cfg, g), env_cfg, space,
                                   {"variant": variant, "method": method, "run": method, "seed_train": 0}))

    if not frames:
        ap.error("no loadable checkpoints in the requested runs")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / "episodes.csv", index=False)
    tables = summarize(df)
    lines = [f"# Сравнение по вариантам среды ({args.which} чекпоинты)\n"]
    for title, t in tables.items():
        lines += [f"## {title}\n", t.round(2).to_markdown(), ""]
    rc = tables["val: rmse_capped, см (падение = 200 см)"]
    if {"ppo_mlp", "rppo"} <= set(rc.columns):
        gain = pd.DataFrame({"MLP − RPPO, см": rc["ppo_mlp"] - rc["rppo"]})
        if "ppo_mlp_integral" in rc.columns:
            gain["MLP − MLP+∫, см"] = rc["ppo_mlp"] - rc["ppo_mlp_integral"]
            gain["MLP+∫ − RPPO, см"] = rc["ppo_mlp_integral"] - rc["rppo"]
        lines += ["## Разница между методами на val (RMSE с учётом падений), см\n", gain.round(2).to_markdown(), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
