"""Метрики качества удержания по истинным траекториям (одна строка на эпизод)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import Rollout

SETTLE_TIME = 2.0        # с: переходный процесс после старта не учитываем
SS_WINDOW = 3.0          # с: окно статической ошибки в конце эпизода
EVENT_WINDOW = 8.0       # с: окно анализа отклика на событие
RECOVER_RADIUS = 0.05    # м
HF_CUTOFF = 5.0          # Гц: «дрожание» = амплитуда составляющих действий выше этой частоты


def _hf_rms(actions: np.ndarray, dt: float) -> float:
    """Среднеквадратичная амплитуда высокочастотной (> HF_CUTOFF) части действий, среднее по каналам."""
    if actions.shape[0] < 16:
        return np.nan
    spec = np.fft.rfft(actions - actions.mean(axis=0), axis=0)
    spec[np.fft.rfftfreq(actions.shape[0], dt) <= HF_CUTOFF] = 0.0
    hf = np.fft.irfft(spec, n=actions.shape[0], axis=0)
    return float(np.sqrt(np.mean(hf**2)))


def _event_metrics(t: np.ndarray, dist: np.ndarray, t_event: float) -> dict:
    out = {"event_peak": np.nan, "event_recovery_time": np.nan, "event_recovered": np.nan, "event_pre_err": np.nan}
    if not np.isfinite(t_event) or t[-1] < t_event + EVENT_WINDOW:
        return out
    pre = (t >= t_event - 1.0) & (t < t_event)
    win = (t >= t_event) & (t <= t_event + EVENT_WINDOW)
    d = dist[win]
    out["event_pre_err"] = float(dist[pre].mean()) if pre.any() else np.nan
    out["event_peak"] = float(d.max())
    outside = np.flatnonzero(d > RECOVER_RADIUS)
    if outside.size == 0:
        rec = 0.0
    elif outside[-1] == d.size - 1:
        rec = np.nan                                   # к концу окна так и не вернулся в радиус
    else:
        rec = float(t[win][outside[-1] + 1] - t_event)
    out["event_recovery_time"] = rec
    out["event_recovered"] = float(np.isfinite(rec))
    return out


def episode_metrics(ro: Rollout, max_dist: float = 2.0) -> pd.DataFrame:
    rows = []
    t_all = ro.t
    for i, cond in enumerate(ro.conditions):
        L = int(ro.length[i])
        t = t_all[:L]
        e = ro.data["pos_err"][:L, i]
        dist = np.linalg.norm(e, axis=1)
        act = ro.data["action"][:L, i]
        settle = t >= SETTLE_TIME
        crashed = bool(ro.crashed[i])
        rec = {
            "episode": i, "crashed": crashed, "crash_reason": ro.crash_reason[i] if ro.crash_reason else "",
            "length_s": L * ro.dt, "return": float(ro.reward[:L, i].sum()),
            "rmse": float(np.sqrt(np.mean(dist[settle] ** 2))) if settle.any() else np.nan,
            "rmse_xy": float(np.sqrt(np.mean(np.sum(e[settle, :2] ** 2, axis=1)))) if settle.any() else np.nan,
            "rmse_z": float(np.sqrt(np.mean(e[settle, 2] ** 2))) if settle.any() else np.nan,
            "max_dev": float(dist[settle].max()) if settle.any() else np.nan,
            "ss_err": np.nan, "action_rate": np.nan,
            "effort": float(np.mean(np.abs(ro.data["thrust_ratio"][:L, i] - 1.0))),
            "action_sq": float(np.mean(np.sum(act**2, axis=1))),
            "hf_rms": _hf_rms(act, ro.dt),
            "max_tilt_deg": float(np.degrees(ro.data["tilt"][:L, i].max())),
        }
        if L > 1:
            rec["action_rate"] = float(np.mean(np.linalg.norm(np.diff(act, axis=0), axis=1)))
        if not crashed:
            tail = t >= t[-1] - SS_WINDOW
            rec["ss_err"] = float(np.linalg.norm(e[tail].mean(axis=0)))
        # «штрафная» RMSE: падение = max_dist, чтобы не было систематической ошибки выживших
        rec["rmse_capped"] = max_dist if crashed else rec["rmse"]
        rec.update(_event_metrics(t, dist, cond.event_time()) if not crashed else
                   {"event_peak": np.nan, "event_recovery_time": np.nan, "event_recovered": 0.0,
                    "event_pre_err": np.nan})
        rec.update({f"cond_{k}": v for k, v in cond.to_record().items()})
        rows.append(rec)
    return pd.DataFrame(rows)
