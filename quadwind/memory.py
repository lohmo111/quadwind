"""Анализ памяти рекуррентной политики.

1. Линейный зонд: ridge-регрессия из скрытого состояния h_t в скрытые параметры среды
   (медленная часть ветра, масса). Разбиение по эпизодам; R² как функция времени с начала эпизода.
   Контроли: тот же зонд по текущему наблюдению и по LSTM со случайными весами.
2. Вмешательство: обнуление или подмена h в момент t* при неизменных условиях → изменение ошибки.
"""
from __future__ import annotations

import numpy as np

from .evaluation import Rollout


def ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ridge с нормировкой признаков. Возвращает (W, b, mean, std); прогноз = ((X − mean)/std)·W + b."""
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
    Z = (X - mu) / sd
    b = Y.mean(axis=0)
    W = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (Y - b))
    return W, b, mu, sd


def ridge_predict(model, X: np.ndarray) -> np.ndarray:
    W, b, mu, sd = model
    return ((X - mu) / sd) @ W + b


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    """R², усреднённый по выходам."""
    ss_res = np.sum((y - pred) ** 2, axis=0)
    ss_tot = np.sum((y - y.mean(axis=0)) ** 2, axis=0) + 1e-12
    return float(np.mean(1.0 - ss_res / ss_tot))


def probe_dataset(ro: Rollout, features: np.ndarray, target: str, every: int = 5):
    """(X, Y, номер эпизода, время) из живых шагов; features: (T, N, F)."""
    T = min(features.shape[0], ro.alive.shape[0])
    t_idx, ep = np.nonzero(ro.alive[:T])
    keep = t_idx % every == 0
    t_idx, ep = t_idx[keep], ep[keep]
    Y = ro.data[target][t_idx, ep]
    if Y.ndim == 1:
        Y = Y[:, None]
    return features[t_idx, ep], Y, ep, (t_idx + 1) * ro.dt


def probe(ro: Rollout, features: np.ndarray, target: str, test_frac: float = 0.3, alpha: float = 1.0,
          time_bins: np.ndarray | None = None, seed: int = 0) -> dict:
    """Обучает зонд на части эпизодов, считает R² на остальных — всего и по интервалам времени."""
    X, Y, ep, t = probe_dataset(ro, features, target)
    rng = np.random.default_rng(seed)
    episodes = np.unique(ep)
    test_eps = rng.choice(episodes, size=max(1, int(len(episodes) * test_frac)), replace=False)
    is_test = np.isin(ep, test_eps)
    model = ridge_fit(X[~is_test], Y[~is_test], alpha)
    pred = ridge_predict(model, X[is_test])
    out = {"r2": r2(Y[is_test], pred), "n_train": int((~is_test).sum()), "n_test": int(is_test.sum())}
    if time_bins is not None:
        tt, yy = t[is_test], Y[is_test]
        out["r2_by_time"] = [r2(yy[(tt >= a) & (tt < b)], pred[(tt >= a) & (tt < b)])
                             if ((tt >= a) & (tt < b)).sum() > 10 else np.nan
                             for a, b in zip(time_bins[:-1], time_bins[1:])]
    return out


def zero_state_hook(step: int, mask: np.ndarray | None = None):
    """Хук для rollout: в момент `step` обнулить LSTM-состояние (всем или по маске)."""
    def hook(k, policy, core):
        if k == step and policy.state is not None:
            h, c = (s.copy() for s in policy.state)
            idx = slice(None) if mask is None else mask
            h[:, idx] = 0.0
            c[:, idx] = 0.0
            policy.state = (h, c)
    return hook


def swap_state_hook(step: int, permutation: np.ndarray):
    """Хук: в момент `step` среда i получает LSTM-состояние среды permutation[i]."""
    def hook(k, policy, core):
        if k == step and policy.state is not None:
            policy.state = tuple(s[:, permutation].copy() for s in policy.state)
    return hook
