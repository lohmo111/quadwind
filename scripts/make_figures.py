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
import json
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
from quadwind.control import PIDGains  # noqa: E402
from quadwind.env import EnvConfig  # noqa: E402
from quadwind.evaluation import rollout  # noqa: E402
from quadwind.policies import PIDPolicy, SB3Policy  # noqa: E402
from quadwind.training import load_model  # noqa: E402


def ordered(methods) -> list[str]:
    order = list(style.PRIMARY) + [m for m in style.METHODS if m not in style.PRIMARY]
    return sorted(methods, key=lambda m: order.index(m) if m in order else len(order))


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
    methods = [m for m in ordered(sweep.method.unique()) if m in style.PRIMARY or m in ("pd", "ppo_mlp_integral")]
    for m in methods:
        name, col, ls, mk = style.style(m)
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
        name, col, _, mk = style.style(m)
        g = val[val.method == m]
        per_seed = g.groupby(seed_col).rmse_capped.mean() * 100
        crash = 100 * g.crashed.mean()
        ring = {} if mk in ("x", "+", "|", "_") else {"edgecolor": style.SURFACE, "linewidth": 1.5}
        ax.scatter(per_seed.values, np.full(len(per_seed), i), color=col, marker=mk, s=40, zorder=3, **ring)
        ax.plot([per_seed.mean()] * 2, [i - 0.3, i + 0.3], color=style.TEXT, lw=2)
        ax.text(per_seed.max() + 0.25, i, f"{per_seed.mean():.1f} см" + (f" · падений {crash:.1f}%" if crash else ""),
                va="center", fontsize=9, color=style.TEXT)
    ax.set_yticks(range(len(methods)), [style.label(m) for m in methods])
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
    methods = [m for m in ordered(df.method.unique()) if m in style.PRIMARY]
    fig, ax = plt.subplots(figsize=(7.6, 0.45 * len(suites) + 1.4))
    offsets = np.linspace(-0.24, 0.24, len(methods))
    for k, m in enumerate(methods):
        name, col, _, mk = style.style(m)
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


def fig_event_response(runs: list[str], out: Path, suites=("step_gust", "mass_pickup")) -> None:
    policies = {}
    gains_path = ROOT / "configs" / "pid_tuned.json"
    gains = PIDGains(**json.loads(gains_path.read_text())) if gains_path.exists() else PIDGains()
    policies["pid"] = (PIDPolicy(EnvConfig(), gains), EnvConfig())
    for run in runs:
        model, normalizer, cfg = load_model(Path(run), "best")
        policies[cfg["name"].split("__")[0]] = (SB3Policy(model, normalizer), EnvConfig.from_dict(cfg.get("env")))
    titles = {"step_gust": "Ступенька ветра +2,5 м/с", "mass_pickup": "Подхват груза +20%"}
    fig, axes = plt.subplots(1, len(suites), figsize=(5.2 * len(suites), 3.8), sharey=False)
    for ax, suite in zip(np.atleast_1d(axes), suites):
        conds = make_suite(suite)
        labels = []
        for m in ordered(policies):
            if m not in style.PRIMARY:
                continue
            pol, env_cfg = policies[m]
            ro = rollout(pol, env_cfg, conds)
            err = 100 * np.linalg.norm(ro.data["pos_err"], axis=2)          # (T, N), см
            t = ro.t - 5.0
            win = (t >= -1.0) & (t <= 4.0)
            med = np.median(err[win], axis=1)
            q1, q3 = np.percentile(err[win], [25, 75], axis=1)
            name, col, ls, _ = style.style(m)
            ax.fill_between(t[win], q1, q3, color=col, alpha=0.15, lw=0)
            ax.plot(t[win], med, color=col, ls=ls, label=name)
            labels.append((4.05, med[-1], name))
        ax.axvline(0, color=style.TEXT_SECONDARY, lw=1, ls="--")
        ax.set_title(titles.get(suite, suite))
        ax.set_xlabel("время от события, с")
        ax.set_ylabel("ошибка позиции, см")
        ax.set_xlim(-1.0, 4.9)
        ax.set_ylim(0, None)
        end_labels(ax, labels)
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Отклик на событие в t = 0: медиана по 32 эпизодам, коридор — межквартильный размах",
                 fontsize=9, color=style.TEXT_SECONDARY, x=0.01, ha="left", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "event_response.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    style.apply()
    if args.csv:
        df = pd.read_csv(args.csv)
        fig_static_error(df, out)
        fig_val_summary(df, out, "val")
        if "test_id" in set(df.suite):
            fig_val_summary(df, out, "test_id")
        fig_suites(df, out)
    if args.runs:
        fig_event_response(args.runs, out)
    print(f"-> {out}: {sorted(p.name for p in out.glob('*.png'))}")


if __name__ == "__main__":
    main()
