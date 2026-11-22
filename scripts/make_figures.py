"""Стандартные рисунки для отчёта.

Из CSV «эпизод = строка» (evaluate.py / compare_variants.py):
  static_error_vs_wind.png — статическая ошибка от скорости ветра (wind_sweep), зона обучения выделена;
  val_summary.png          — RMSE с учётом падений по методам (точки — сиды, черта — среднее/IQM);
  suites.png               — тот же показатель по наборам (test_id, OOD), если они есть в CSV.
Из обученных моделей (--runs) и PID:
  event_response.png       — средняя ошибка вокруг события (ступенька ветра, подхват груза), коридор — IQR.
Пример:
    python scripts/make_figures.py --csv results/e7_v0_pilot2/episodes.csv \
        --runs runs/pilot2/rppo/seed_0 runs/pilot2/ppo_mlp/seed_0 runs/pilot2/ppo_stack16/seed_0 --out results/figures
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quadwind import style  # noqa: E402
from quadwind.conditions import TRAIN_SPACE, make_suite  # noqa: E402
from quadwind.env import EnvConfig  # noqa: E402
from quadwind.evaluation import rollout  # noqa: E402
from quadwind.policies import PIDPolicy, SB3Policy, baseline_env_config, tuned_pid_gains  # noqa: E402
from quadwind.training import load_model  # noqa: E402


def ordered(methods) -> list[str]:
    order = list(style.PRIMARY) + [m for m in style.METHODS if m not in style.PRIMARY]
    return sorted(methods, key=lambda m: (order.index(m.split("__")[0]) if m.split("__")[0] in order
                                         else len(order), m))


def plot_style(method: str, methods):
    base, _, variant = method.partition("__")
    name, col, ls, marker = style.style(base)
    related = sorted(m for m in methods if m.partition("__")[0] == base)
    if len(related) > 1:
        index = related.index(method)
        name = f"{name} ({variant or 'base'})"
        ls = ["-", "--", "-.", ":"][index % 4]
    return name, col, ls, marker


def end_labels(ax, items: list[tuple[float, float, str]]) -> None:
    """Подписи у концов линий (нейтральный текст) с раздвижкой, чтобы не налезали."""
    items = sorted(items, key=lambda x: x[1])
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    placed = []
    for x, y, text in items:
        y_txt = max([y] + [p + gap for p in placed[-1:]])
        placed.append(y_txt)
        ax.annotate(text, (x, y), xytext=(x, y_txt), textcoords="data", va="center", ha="left",
                    fontsize=9, color=style.TEXT, xycoords="data",
                    annotation_clip=False, arrowprops=None)


def fig_static_error(df: pd.DataFrame, out: Path) -> None:
    sweep = df[df.suite == "wind_sweep"]
    if sweep.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    hi = TRAIN_SPACE.wind_speed.bounds(1.0)[1]
    ax.axvspan(0, hi, color=style.BAND, zorder=0, lw=0)
    ax.text(hi / 2, 0.97, "диапазон обучения", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=9, color=style.TEXT_SECONDARY)
    labels = []
    methods = ordered(sweep.method.unique())
    for m in methods:
        name, col, ls, mk = plot_style(m, methods)
        g = sweep[sweep.method == m].groupby("cond_wind_speed").ss_err.mean() * 100
        ax.plot(g.index, g.values, color=col, ls=ls, marker=mk, ms=6, label=name)
        labels.append((g.index[-1] + 0.12, g.values[-1], name))
    ax.set_xlabel("скорость постоянного ветра, м/с")
    ax.set_ylabel("статическая ошибка, см")
    ax.set_title("Статическая ошибка при постоянном ветре")
    ax.set_xlim(-0.2, sweep.cond_wind_speed.max() + 1.0)
    ax.set_ylim(0, None)
    end_labels(ax, labels)
    ax.legend(loc="upper left", fontsize=9, bbox_to_anchor=(0.0, 0.92))
    fig.tight_layout()
    fig.savefig(out / "static_error_vs_wind.png", dpi=150)
    plt.close(fig)


def fig_val_summary(df: pd.DataFrame, out: Path, suite: str = "val") -> None:
    val = df[df.suite == suite]
    if val.empty:
        return
    seed_col = "seed_train" if "seed_train" in val.columns else "run"
    methods = ordered(val.method.unique())
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(methods) + 1.3))
    for i, m in enumerate(methods):
        name, col, _, mk = plot_style(m, methods)
        g = val[val.method == m]
        per_seed = g.groupby(seed_col).rmse_capped.mean() * 100
        crash = 100 * g.crashed.mean()
        ring = {} if mk in ("x", "+", "|", "_") else {"edgecolor": style.SURFACE, "linewidth": 1.5}
        ax.scatter(per_seed.values, np.full(len(per_seed), i), color=col, marker=mk, s=40, zorder=3, **ring)
        ax.plot([per_seed.mean()] * 2, [i - 0.3, i + 0.3], color=style.TEXT, lw=2)
        ax.text(per_seed.max() + 0.25, i, f"{per_seed.mean():.1f} см" + (f" · падений {crash:.1f}%" if crash else ""),
                va="center", fontsize=9, color=style.TEXT)
    ax.set_yticks(range(len(methods)), [plot_style(m, methods)[0] for m in methods])
    ax.invert_yaxis()
    ax.set_xlabel("RMSE с учётом падений (падение = 200 см), см")
    ax.set_title(f"Точность удержания на наборе {suite}")
    ax.set_xlim(0, (val.groupby(["method", seed_col]).rmse_capped.mean().max() * 100) * 1.45)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / f"{suite}_summary.png", dpi=150)
    plt.close(fig)


def fig_suites(df: pd.DataFrame, out: Path) -> None:
    suites = [s for s in df.suite.unique() if s == "test_id" or s.startswith("ood_")]
    if len(suites) < 2:
        return
    methods = ordered(df.method.unique())
    fig, ax = plt.subplots(figsize=(7.6, 0.45 * len(suites) + 1.4))
    offsets = np.linspace(-0.24, 0.24, len(methods))
    for k, m in enumerate(methods):
        name, col, _, mk = plot_style(m, methods)
        vals = [100 * df[(df.suite == s) & (df.method == m)].rmse_capped.mean() for s in suites]
        ring = {} if mk in ("x", "+", "|", "_") else {"edgecolor": style.SURFACE, "linewidth": 1.2}
        ax.scatter(vals, np.arange(len(suites)) + offsets[k], color=col, marker=mk, s=36, label=name, zorder=3, **ring)
    ax.set_yticks(range(len(suites)), suites)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("RMSE с учётом падений, см (лог. шкала)")
    ax.set_title("Обобщение: знакомые условия и отложенные (OOD)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "suites.png", dpi=150)
    plt.close(fig)


def fig_event_response(runs: list[str], out: Path, suites=("step_gust", "mass_pickup"), namespace="research") -> None:
    import torch
    torch.set_num_threads(2)
    policies = {}
    plant = None
    for run in runs:
        run_dir = Path(run)
        model, normalizer, cfg = load_model(run_dir, "best")
        env_cfg = EnvConfig.from_dict(cfg.get("env"))
        matched = baseline_env_config(env_cfg)
        if plant is not None and plant != matched:
            raise ValueError("An event figure requires one plant and sensor configuration. Plot the groups separately.")
        plant = matched
        seeds = policies.setdefault(cfg["name"], {})
        if run_dir.name in seeds:
            raise ValueError(f"Duplicate {cfg['name']} / {run_dir.name}. Select one checkpoint per seed.")
        seeds[run_dir.name] = (SB3Policy(model, normalizer), env_cfg)
    plant = plant or EnvConfig()
    policies["pid"] = {"pid": (PIDPolicy(plant, tuned_pid_gains(plant)), plant)}
    titles = {"step_gust": "Ступенька ветра +2,5 м/с", "mass_pickup": "Подхват груза +20%"}
    fig, axes = plt.subplots(1, len(suites), figsize=(5.2 * len(suites), 3.8), sharey=False)
    for ax, suite in zip(np.atleast_1d(axes), suites):
        conds = make_suite(suite, namespace=namespace)
        for m in ordered(policies):
            traces = []
            times = np.empty(0)
            for pol, env_cfg in policies[m].values():
                ro = rollout(pol, env_cfg, conds)
                t = ro.t - conds[0].event_time()
                win = (t >= -1.0) & (t <= 4.0)
                err = 100 * np.linalg.norm(ro.data["pos_err"][win], axis=2)
                traces.append(np.where(ro.alive[win], err, np.nan))
                if len(err) > len(times):
                    times = t[win]
            # Some seeds can lose every episode early. Pad those traces to the shared time grid.
            n_steps = len(times)
            if not n_steps:
                continue
            errors = np.concatenate([np.pad(trace, ((0, n_steps - len(trace)), (0, 0)), constant_values=np.nan)
                                     for trace in traces], axis=1)
            valid = np.isfinite(errors).any(axis=1)
            times = times[valid]
            q1, med, q3 = np.nanquantile(errors[valid], [0.25, 0.5, 0.75], axis=1)
            name, col, ls, _ = plot_style(m, policies)
            if m != "pid":
                name += f" (n={len(traces)} seeds)"
            ax.fill_between(times, q1, q3, color=col, alpha=0.15, lw=0)
            ax.plot(times, med, color=col, ls=ls, label=name)
        ax.axvline(0, color=style.TEXT_SECONDARY, lw=1, ls="--")
        ax.set_title(titles.get(suite, suite))
        ax.set_xlabel("время от события, с")
        ax.set_ylabel("ошибка позиции, см")
        ax.set_xlim(-1.0, 4.9)
        ax.set_ylim(0, None)
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Отклик на событие: медиана и межквартильный размах живых траекторий по эпизодам и сидам",
                 fontsize=9, color=style.TEXT_SECONDARY, x=0.01, ha="left", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "event_response.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--namespace", choices=["research", "final"], default="research", help="набор для --runs")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    style.apply()
    if args.csv:
        df = pd.read_csv(args.csv)
        keys = [k for k in ("namespace", "variant", "environment_group") if k in df and df[k].nunique() > 1]
        groups = df.groupby(keys, dropna=False) if keys else [((), df)]
        for group, data in groups:
            tags = group if isinstance(group, tuple) else (group,)
            suffix = "__".join(re.sub(r"[^\w-]", "_", f"{key}-{value}") for key, value in zip(keys, tags))
            directory = out / suffix
            directory.mkdir(exist_ok=True)
            fig_static_error(data, directory)
            fig_val_summary(data, directory, "val")
            fig_val_summary(data, directory, "test_id")
            fig_suites(data, directory)
    if args.runs:
        fig_event_response(args.runs, out, namespace=args.namespace)
    print(f"-> {out}: {sorted(p.name for p in out.glob('*.png'))}")


if __name__ == "__main__":
    main()
