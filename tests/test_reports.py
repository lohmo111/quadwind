"""Скрипты отчётов на синтетических данных: рисунки и сводные таблицы строятся без ошибок."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

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
    mf.fig_val_summary(df, tmp_path, "test_id")
    mf.fig_suites(df, tmp_path)
    assert {p.name for p in tmp_path.glob("*.png")} == {
        "static_error_vs_wind.png", "val_summary.png", "test_id_summary.png", "suites.png"}


def test_compare_variants_summary_tables():
    cv = load_script("compare_variants")
    tables = cv.summarize(synthetic_episodes())
    rc = tables["val: rmse_capped, см (падение = 200 см)"]
    assert list(rc.columns[:2]) == ["pd", "pid"] and "rppo" in rc.columns
    assert np.isfinite(rc.to_numpy()).all()


def test_plot_and_table_keep_trainer_and_checkpoint_separate(tmp_path, monkeypatch):
    mf = load_script("make_figures")
    cv = load_script("compare_variants")
    df = synthetic_episodes().query("method == 'rppo'")
    data = pd.concat([df.assign(backend=backend, checkpoint=checkpoint, rmse_capped=error)
                      for backend, checkpoint, error in (("sb3", "best", .02), ("native", "best", .5),
                                                         ("native", "early", 1.))])
    rc = cv.summarize(data)["val: rmse_capped, см (падение = 200 см)"]
    assert rc.loc["v0_current"].to_dict() == {
        "rppo [sb3, best]": 2., "rppo [native, best]": 50., "rppo [native, early]": 100.}
    with monkeypatch.context() as patch:
        patch.setattr(mf.plt, "close", lambda _: None)
        mf.fig_val_summary(data, tmp_path)
        fig = mf.plt.gcf()
    ax = fig.axes[0]
    per_method = {tick.get_text(): dots.get_offsets()[:, 0].tolist()
                  for tick, dots in zip(ax.get_yticklabels(), ax.collections)}
    assert per_method == {
        "RPPO [sb3, best]": [2., 2.], "RPPO [native, best]": [50., 50.], "RPPO [native, early]": [100., 100.]}
    mf.plt.close(fig)


def test_event_figure_uses_all_seeds_and_matched_pid(tmp_path, monkeypatch):
    mf = load_script("make_figures")
    models = {"seed_0": 1.0, "seed_1": 3.0}

    def load(run, which):
        cfg = {"name": "rppo__att", "env": {"action_mode": "attitude", "rate_ki": [100, 100, 50]}}
        if run.parent.name == "native":
            cfg["backend"] = "native"
        return SimpleNamespace(policy=None, distance=models[run.name]), None, cfg

    plants = []

    def rollout(policy, cfg, conditions):
        plants.append(cfg)
        # At the second step one episode is already dead and its stored state has diverged.
        amplitude = policy.model.distance if isinstance(policy, mf.SB3Policy) else 2.0
        x = np.zeros((2, 2, 3))
        x[:, :, 0] = amplitude
        x[1, 1, 0] = 1000
        return SimpleNamespace(t=np.array([5.0, 5.02]), data={"pos_err": x},
                               alive=np.array([[True, True], [True, False]]))

    monkeypatch.setattr(mf, "load_model", load)
    monkeypatch.setattr(mf, "rollout", rollout)
    with monkeypatch.context() as patch:
        patch.setattr(mf.plt, "close", lambda _: None)
        mf.fig_event_response(["seed_0", "seed_1", "native/seed_0"], tmp_path, suites=["mass_pickup"])
        fig = mf.plt.gcf()
    lines = {line.get_label(): line for line in fig.axes[0].lines}
    np.testing.assert_allclose(lines["RPPO [sb3, best] (n=2 seeds)"].get_ydata(), [200, 200])
    np.testing.assert_allclose(lines["RPPO [native, best] (n=1 seeds)"].get_ydata(), [100, 100])
    assert "PID" in lines
    assert all(cfg.action_mode == "attitude" and cfg.rate_ki == (100, 100, 50) for cfg in plants)
    mf.plt.close(fig)
    with pytest.raises(ValueError, match="Duplicate"):
        mf.fig_event_response(["seed_0", "seed_0"], tmp_path)
