"""Статистика по сидам: IQM, иерархический бутстрап, вероятность улучшения.

Единица независимости в RL — сид обучения. Эпизоды одного сида не независимы
(одна и та же политика), поэтому интервалы строятся бутстрапом по сидам, а внутри
сида — по эпизодам (иерархически). См. Agarwal et al., 2021 (rliable).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def iqm(x) -> float:
    """Interquartile mean: среднее центральных 50% значений."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(sps.trim_mean(x, 0.25)) if x.size else np.nan


def hierarchical_bootstrap(per_seed: list[np.ndarray], stat=np.mean, n_boot: int = 2000, alpha: float = 0.05,
                           seed: int = 0) -> tuple[float, float, float]:
    """Оценка stat(среднее по эпизодам каждого сида) и (1−alpha)-интервал.

    per_seed: список массивов значений метрики по эпизодам для каждого сида.
    """
    rng = np.random.default_rng(seed)
    per_seed = [np.asarray(v, dtype=float)[np.isfinite(v)] for v in per_seed]
    per_seed = [v for v in per_seed if v.size]
    if not per_seed:
        return np.nan, np.nan, np.nan
    point = stat(np.array([v.mean() for v in per_seed]))
    boots = np.empty(n_boot)
    k = len(per_seed)
    for b in range(n_boot):
        seeds = rng.integers(0, k, k)
        means = [per_seed[s][rng.integers(0, per_seed[s].size, per_seed[s].size)].mean() for s in seeds]
        boots[b] = stat(np.array(means))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def prob_improvement(a, b, lower_is_better: bool = True) -> float:
    """P(X_a лучше X_b) по всем парам значений (статистика Манна–Уитни / (n·m)); ничьи — пополам."""
    a = np.asarray(a, dtype=float)[:, None]
    b = np.asarray(b, dtype=float)[None, :]
    better = (a < b) if lower_is_better else (a > b)
    return float(np.mean(better + 0.5 * (a == b)))


def summarize(df: pd.DataFrame, metric: str, by=("method", "suite"), seed_col: str = "seed_train",
              stat=iqm, n_boot: int = 2000) -> pd.DataFrame:
    """Таблица: для каждой группы — значение stat по сидам и бутстрап-интервал."""
    rows = []
    for key, g in df.groupby(list(by)):
        per_seed = [s[metric].to_numpy() for _, s in g.groupby(seed_col)]
        point, lo, hi = hierarchical_bootstrap(per_seed, stat=stat if len(per_seed) >= 4 else np.mean, n_boot=n_boot)
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), "metric": metric,
                     "value": point, "ci_low": lo, "ci_high": hi, "n_seeds": len(per_seed), "n_episodes": len(g)})
    return pd.DataFrame(rows)
