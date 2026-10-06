"""Прогон политики на фиксированном списке условий с записью истинных траекторий.

Каждый эпизод проигрывается ровно один раз, все эпизоды — синхронно в одном батче.
Условия, сиды и реализации ветра одинаковы для всех методов, поэтому сравнения парные.
Метрики считаются по истинному состоянию, а не по шумным наблюдениям.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .conditions import EpisodeCondition
from .env import TERMINATION_REASONS, EnvConfig, HoverCore

RECORD_KEYS = ("pos_err", "vel", "omega", "tilt", "action", "applied", "thrust_ratio", "wind", "wind_slow",
               "mass_ratio")


@dataclass
class Rollout:
    conditions: list[EpisodeCondition]
    dt: float
    data: dict[str, np.ndarray]     # ключ → (T, N, ...) истинные величины после каждого шага
    reward: np.ndarray              # (T, N)
    alive: np.ndarray               # (T, N): шаг t принадлежит эпизоду (эпизод ещё не закончился до него)
    length: np.ndarray              # (N,) число шагов эпизода
    crashed: np.ndarray             # (N,)
    hidden: np.ndarray | None = None  # (T, N, H) скрытые состояния LSTM актора (если записывались)
    crash_reason: list | None = None  # причина падения ("tilt", "distance", …) или ""

    @property
    def n(self) -> int:
        return len(self.conditions)

    @property
    def t(self) -> np.ndarray:
        return (np.arange(self.reward.shape[0]) + 1) * self.dt


def rollout(policy, cfg: EnvConfig, conditions: list[EpisodeCondition], record_hidden: bool = False,
            hook: Callable[[int, object, HoverCore], None] | None = None, record: bool = True) -> Rollout:
    """policy: объект с reset(n) и act(obs, episode_start); hook(step, policy, core) вызывается перед шагом
    (например, чтобы обнулить или подменить скрытое состояние LSTM в анализе памяти)."""
    n = len(conditions)
    core = HoverCore(cfg, n)
    core.reset_envs(np.arange(n), conditions)
    policy.reset(n)
    T = core.max_steps
    episode_start = np.ones(n, dtype=bool)
    alive_now = np.ones(n, dtype=bool)
    crashed = np.zeros(n, dtype=bool)
    reason = [""] * n
    length = np.zeros(n, dtype=int)
    alive = np.zeros((T, n), dtype=bool)
    reward = np.zeros((T, n))
    data: dict[str, list] = {k: [] for k in RECORD_KEYS} if record else {}
    hidden = [] if record_hidden else None

    for k in range(T):
        actions = policy.act(core.observe(), episode_start)
        episode_start[:] = False
        if hook is not None:
            hook(k, policy, core)
        r, term, trunc = core.step(actions)
        alive[k] = alive_now
        reward[k] = r
        length += alive_now
        for i in np.flatnonzero(term & alive_now):
            reason[i] = TERMINATION_REASONS[core.term_reason[i]]
        crashed |= term & alive_now
        alive_now &= ~(term | trunc)
        if record:
            state = core.true_state()
            for key in RECORD_KEYS:
                data[key].append(state[key])
        if record_hidden:
            h = policy.hidden() if hasattr(policy, "hidden") else None
            hidden.append(np.zeros((n, 0)) if h is None else h.copy())
        if not alive_now.any():
            alive, reward = alive[: k + 1], reward[: k + 1]
            break

    arrays = {key: np.asarray(v) for key, v in data.items()}
    return Rollout(conditions, core.dt, arrays, reward, alive, length, crashed,
                   np.asarray(hidden) if record_hidden else None, reason)


def evaluate_in_chunks(policy, cfg: EnvConfig, conditions: list[EpisodeCondition], chunk: int = 256,
                       **kw) -> list[Rollout]:
    """Большие наборы прогоняются частями (память под траектории ~ T·N)."""
    return [rollout(policy, cfg, conditions[i: i + chunk], **kw) for i in range(0, len(conditions), chunk)]
