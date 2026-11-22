"""Единый интерфейс политик для оценки: reset(n) и act(obs, episode_start) → действия (n, 4).

Все методы получают одни и те же «сырые» наблюдения среды. Нормализацию для
SB3-моделей адаптер делает сам по сохранённым статистикам VecNormalize.
Рекуррентный адаптер передаёт LSTM-состояние между шагами и сбрасывает его по
episode_start. Без этого RPPO при оценке превращается в политику без памяти.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .control import PIDGains, PositionPID
from .env import EnvConfig
from .params import CF2X


@dataclass(frozen=True)
class ObsLayout:
    pos: slice = slice(0, 3)
    vel: slice = slice(3, 6)
    rot: slice = slice(6, 15)
    rate: slice = slice(15, 18)
    prev_action: slice = slice(18, 22)


LAYOUT = ObsLayout()


class ConstantPolicy:
    """Постоянное действие (по умолчанию — тяга висения и нулевые угловые скорости)."""

    def __init__(self, action=(0.0, 0.0, 0.0, 0.0)):
        self.action = np.asarray(action, dtype=np.float32)

    def reset(self, n: int) -> None:
        self.n = n

    def act(self, obs: np.ndarray, episode_start: np.ndarray) -> np.ndarray:
        return np.tile(self.action, (obs.shape[0], 1))


class PIDPolicy:
    """Каскадный PID поверх того же CTBR-интерфейса и тех же наблюдений, что у агента."""

    def __init__(self, cfg: EnvConfig, gains: PIDGains | None = None):
        self.cfg = cfg
        self.gains = gains or PIDGains()
        att = (cfg.tilt_max, cfg.yaw_rate_max) if cfg.action_mode == "attitude" else None
        self.pid = PositionPID(CF2X, self.gains, 1.0 / cfg.ctrl_freq, np.asarray(cfg.rate_max), attitude=att)

    def reset(self, n: int) -> None:
        self.pid.reset(n)

    def act(self, obs: np.ndarray, episode_start: np.ndarray) -> np.ndarray:
        episode_start = np.asarray(episode_start, dtype=bool)
        if episode_start.any():
            self.pid.reset_envs(episode_start)
        frame = np.asarray(obs, dtype=float)[:, : self.cfg.frame_dim]   # самый свежий кадр
        R = frame[:, LAYOUT.rot].reshape(-1, 3, 3)
        return self.pid.act(frame[:, LAYOUT.pos], frame[:, LAYOUT.vel], R).astype(np.float32)


@dataclass
class ObsNormalizer:
    """Замороженная нормализация наблюдений (как VecNormalize при training=False)."""
    mean: np.ndarray
    var: np.ndarray
    clip: float = 10.0
    eps: float = 1e-8

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.clip((obs - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip).astype(np.float32)

    @classmethod
    def from_vecnormalize(cls, vn) -> "ObsNormalizer":
        return cls(vn.obs_rms.mean.copy(), vn.obs_rms.var.copy(), float(vn.clip_obs), float(vn.epsilon))

    def save(self, path) -> None:
        np.savez(path, mean=self.mean, var=self.var, clip=self.clip, eps=self.eps)

    @classmethod
    def load(cls, path) -> "ObsNormalizer":
        z = np.load(path)
        return cls(z["mean"], z["var"], float(z["clip"]), float(z["eps"]))


class SB3Policy:
    """Адаптер для моделей SB3 / sb3-contrib (PPO, RecurrentPPO)."""

    def __init__(self, model, normalizer: ObsNormalizer | None = None, deterministic: bool = True):
        self.model = model
        self.normalizer = normalizer
        self.deterministic = deterministic
        self.recurrent = hasattr(model.policy, "lstm_actor")
        self.state = None

    def reset(self, n: int) -> None:
        self.state = None

    def act(self, obs: np.ndarray, episode_start: np.ndarray) -> np.ndarray:
        x = self.normalizer(obs) if self.normalizer is not None else obs.astype(np.float32)
        if self.recurrent:
            actions, self.state = self.model.predict(
                x, state=self.state, episode_start=np.asarray(episode_start, dtype=bool),
                deterministic=self.deterministic)
        else:
            actions, _ = self.model.predict(x, deterministic=self.deterministic)
        return np.asarray(actions, dtype=np.float32)

    def hidden(self) -> np.ndarray | None:
        """Скрытое состояние h актора после последнего шага: (n, n_layers·hidden)."""
        if not self.recurrent or self.state is None:
            return None
        h = self.state[0]                       # (n_layers, n, hidden)
        return np.transpose(h, (1, 0, 2)).reshape(h.shape[1], -1)

