"""Скрипты отчётов на синтетических данных: рисунки и сводные таблицы строятся без ошибок."""
import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parent.parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_episodes():
    rng = np.random.default_rng(0)
    rows = []
    for method in ("rppo", "ppo_mlp", "ppo_stack16", "pid", "pd"):
        for seed in (0, 1):
            for suite in ("val", "test_id", "ood_heavy", "wind_sweep", "step_gust", "mass_pickup"):
                for ep in range(6):
                    crashed = rng.uniform() < 0.05
                    rows.append({"variant": "v0_current", "method": method, "seed_train": seed, "run": f"{method}{seed}",
                                 "suite": suite, "episode": ep, "crashed": crashed,
                                 "rmse": rng.uniform(0.02, 0.1), "rmse_capped": 2.0 if crashed else rng.uniform(0.02, 0.1),
                                 "ss_err": rng.uniform(0, 0.05), "event_peak": rng.uniform(0.05, 0.2),
                                 "event_recovery_time": rng.uniform(0.3, 2.0),
                                 "cond_wind_speed": float(ep)})
    return pd.DataFrame(rows)


def test_make_figures_from_csv(tmp_path):
    mf = load_script("make_figures")
    mf.style.apply()
    df = synthetic_episodes()
    mf.fig_static_error(df, tmp_path)
    mf.fig_val_summary(df, tmp_path, "val")
    mf.fig_suites(df, tmp_path)
    assert {p.name for p in tmp_path.glob("*.png")} == {"static_error_vs_wind.png", "val_summary.png", "suites.png"}


def test_compare_variants_summary_tables():
    cv = load_script("compare_variants")
    tables = cv.summarize(synthetic_episodes())
    rc = tables["val: rmse_capped, см (падение = 200 см)"]
    assert list(rc.columns[:2]) == ["pd", "pid"] and "rppo" in rc.columns
    assert np.isfinite(rc.to_numpy()).all()
