"""Verify the three native hover runs and write results/native_defense/summary.md."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quadwind.conditions import SUITE_SIZES, make_suite
from quadwind.training import load_model
from scripts.summarize_hover import summarize


def require(condition, message):
    if not condition:
        raise ValueError(message)


def training_row(run, seed):
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    history = pd.read_csv(run / "train_history.csv")
    val = pd.read_csv(run / "val_history.csv")
    total = cfg["total_timesteps"]
    require(meta["status"] == "done" and cfg["backend"] == meta["backend"] == "native"
            and meta["seed"] == seed and cfg["algo"] == "rppo", f"{run}: run identity")
    require(total == meta["actual_timesteps"] == 3_000_000 and meta["train_seconds"] > 0,
            f"{run}: incomplete training")
    chunk = cfg["n_envs"] * cfg["algo_kwargs"]["n_steps"]
    steps = np.minimum(np.arange(chunk, total + chunk, chunk), total)
    np.testing.assert_array_equal(history.timesteps, steps)
    require(np.isfinite(history.to_numpy()).all(), f"{run}: nonfinite training history")
    require(cfg["eval"]["suite"] == "val" and (cfg["eval"]["n"] or SUITE_SIZES["val"]) == 256,
            f"{run}: validation set")
    freq = cfg["eval"]["freq"]
    eval_at = np.minimum(np.arange(freq, total + freq, freq), total)
    np.testing.assert_array_equal(val.timesteps, np.unique(steps[np.searchsorted(steps, eval_at)]))
    require(np.isfinite(val[["rmse_capped", "crash_rate"]].to_numpy()).all(), f"{run}: invalid validation")
    best = val.loc[val.rmse_capped.idxmin()]
    np.testing.assert_allclose(best.rmse_capped, meta["best_val_rmse_capped"], atol=1e-12)
    previous = None
    for which, expected in (("initial", 0), ("early", steps[0]), ("best", best.timesteps)):
        model, norm, _ = load_model(run, which)
        require(model.num_timesteps == expected, f"{run}: {which} step mismatch")
        dim = model.policy.obs_dim
        require(norm is not None and norm.mean.shape == norm.var.shape == (dim,)
                and np.isfinite(norm.mean).all() and np.isfinite(norm.var).all()
                and (norm.var >= 0).all() and norm.clip == cfg["normalize"]["clip_obs"] and norm.eps > 0,
                f"{run}: {which} observation statistics")
        weights = model.policy.state_dict()
        require(all(t.isfinite().all() for t in weights.values()), f"{run}: {which} weights")
        if previous is not None and model.num_timesteps != previous[0]:
            for head in ("actor_head.weight", "critic_head.weight"):
                require(not torch.equal(previous[1][head], weights[head]), f"{run}: unchanged {which} {head}")
        previous = model.num_timesteps, weights
        action, _ = model.predict(norm(np.zeros(dim)), episode_start=[True])
        require(action.shape == (4,) and (np.abs(action) <= 1).all(), f"{run}: {which} inference")

    def val_cell(row):
        return f"{int(row.timesteps)} / {100 * row.rmse_capped:.2f} / {round(256 * row.crash_rate)}"

    row = (f"| {seed} | {int(steps[0])} | {val_cell(val.iloc[0])} | {val_cell(best)} "
           f"| {meta['train_seconds'] / 60:.1f} |")
    return row, {"early": int(steps[0]), "best": int(best.timesteps)}


def held_out(root, out):
    episodes = pd.read_csv(out / "episodes.csv")
    keys = ["backend", "method", "seed_train", "checkpoint"]
    controllers = {("native", "rppo_hover", i, "best") for i in range(3)} | {("baseline", "pid", 0, "fixed")}
    require(set(episodes.namespace) == {"final"} and episodes.environment_group.nunique() == 1,
            "Expected matched final evaluations")
    require(set(episodes.suite) == {"test_id", "safety"}, "Expected test_id and safety")
    trained = episodes[episodes.backend == "native"]
    for seed, group in trained.groupby("seed_train"):
        suffix = f"runs/native_defense/rppo_hover/seed_{seed}"
        require(all(str(run).replace("\\", "/").endswith(suffix) for run in group.run.unique()),
                f"Unexpected source checkpoint for seed {seed}")
    archived = root / "results/defense/episodes.csv"
    old = pd.read_csv(archived) if archived.is_file() else None
    for suite, group in episodes.groupby("suite"):
        require(set(group[keys].itertuples(index=False, name=None)) == controllers,
                f"{suite}: expected the three selected native checkpoints and PID")
        expected = pd.DataFrame(c.to_record() for c in make_suite(suite, namespace="final")).add_prefix("cond_")
        expected = expected.set_index("cond_seed").sort_index()
        frames = [g for _, g in group.groupby(keys)]
        if old is not None:
            prior = old[(old.suite == suite) & (old.namespace == "final")]
            require(not prior.empty, f"{suite}: archived comparison unavailable")
            frames.extend(g for _, g in prior.groupby(["method", "seed_train"]))
        for frame in frames:
            observed = frame.set_index("cond_seed").sort_index()
            pd.testing.assert_frame_equal(observed[expected.columns], expected,
                                          check_dtype=False, atol=1e-12, rtol=1e-12)
    scores = summarize(episodes)
    saved = pd.read_csv(out / "summary.csv")
    pd.testing.assert_frame_equal(scores, saved[scores.columns], check_dtype=False, atol=5.1e-7, rtol=1e-12)
    rows = []
    for (backend, method, seed, _), group in scores.groupby(keys, sort=False):
        table = group.set_index("suite")
        test, safety = table.loc["test_id"], table.loc["safety"]
        controller = f"PPO-LSTM, сид {seed}" if backend == "native" else "Настроенный PID"
        median = test.median_surviving_rmse_cm
        result = f"{median:.2f}" if np.isfinite(median) else "нет выживших"
        rows.append(f"| {controller} | {backend} | {result} | {int(test.crashes)} / {int(test.episodes)} "
                    f"| {int(safety.crashes)} / {int(safety.episodes)} |")
    return rows, archived.is_file()


def paired_outcomes(out, checkpoints):
    flights = pd.read_csv(out / "paired_flights.csv").set_index("checkpoint")
    expected_seed = make_suite("probe", n=1, namespace="research")[0].to_record()["seed"]
    require(flights.index.tolist() == ["early", "best"] and set(flights.condition_seed) == {expected_seed}
            and set(flights.suite) == {"research/probe"} and set(flights.episode) == {0}
            and flights.seed_train.nunique() == 1, "Mismatched paired flights")
    seed = int(flights.seed_train.iloc[0])
    require(flights.timesteps.to_dict() == checkpoints[seed], "Video checkpoint steps")
    lines = [f"## Два полёта, сид обучения {seed}", "",
             f"Начальное состояние, ветер и шум совпадают по сиду эпизода `{flights.condition_seed.iloc[0]}`.",
             "", "| Политика | Шаг | Исход | Длительность, с | Пик ошибки, см | Ошибка в конце, см |",
             "|---|---:|---|---:|---:|---:|"]
    for which, row in flights.iterrows():
        label = "Ранняя" if which == "early" else "Лучшая по валидации"
        outcome = f"авария (`{row.crash_reason}`)" if row.crashed else "завершён"
        lines.append(f"| {label} | {row.timesteps} | {outcome} | {row.length_s:.2f} "
                     f"| {100 * row.peak_error_m:.2f} | {100 * row.final_error_m:.2f} |")
    return lines


def write_report(root):
    out = root / "results/native_defense"
    run_dir = root / "runs/native_defense/rppo_hover"
    torch.set_num_threads(1)
    runs, checkpoints = zip(*(training_row(run_dir / f"seed_{s}", s) for s in range(3)))
    scores, matched_archive = held_out(root, out)
    for name in ("before_after.mp4", "validation.png"):
        require((out / name).stat().st_size > 0, f"Missing {name}")
    lines = [
        "# Обучение собственного PPO-LSTM",
        "",
        "Три сида прошли по 3 000 000 переходов. На каждом сиде сохранены начальные веса (шаг 0), "
        "ранние веса после первого сбора и лучший чекпоинт по средней RMSE на 256 фиксированных "
        "валидационных эпизодах. При выборе чекпоинта падению присваивали ошибку 200 см.",
        "",
        "| Сид | Ранний шаг | Первая валидация, шаг / RMSE (см) / падения | Лучшая, шаг / RMSE (см) / падения | Время обучения с валидацией, мин |",
        "|---:|---:|---:|---:|---:|", *runs, "",
        "Каждую пару весов и статистик нормализации загрузили и проверили получение действия. "
        "После обучения изменились веса выходных слоёв актора и критика. "
        "[Кривые валидации](validation.png) · [Чекпоинты и журналы](../../runs/native_defense/rppo_hover)", "",
        *paired_outcomes(out, checkpoints), "",
        "[Запись полётов](before_after.mp4) · [Результаты двух полётов](paired_flights.csv)", "",
        "## Проверка выбранных политик", "",
        "Все четыре контроллера прошли одинаковые 200 тестовых эпизодов и отдельные 1000 эпизодов проверки "
        "падений. Условия и сиды шума сверены для каждого эпизода. Медиана рассчитана по выжившим полётам "
        "из трёхмерной ошибки позиции после первых двух секунд.", "",
        "| Контроллер | Реализация | Тест, медиана RMSE (см) | Тест, падения | Проверка падений |",
        "|---|---|---:|---:|---:|", *scores, "",
        "[Результаты всех эпизодов](episodes.csv) · [Сводная таблица](summary.csv)",
    ]
    if matched_archive:
        lines.extend(["", "Исторические модели SB3-Contrib проверяли на тех же условиях. "
                      "[Отдельная таблица SB3-Contrib и PID](../defense/summary.csv)"])
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Root of the checkout or extracted submission")
    write_report(parser.parse_args().root)
