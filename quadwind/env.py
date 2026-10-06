"""Среда удержания позиции: батчевое ядро + SB3 VecEnv + обёртка gymnasium.Env.

Шаг политики (50 Гц) = 10 шагов физики (500 Гц). На каждом подшаге работают
регулятор угловых скоростей и миксер — как на реальном полётном контроллере.

Наблюдение одного кадра (22 числа): [e_p (3), v (3), R (9), ω/5 (3), a_{t−1} (4)].
Первые 18 — измерения с шумом и, возможно, задержкой; a_{t−1} — последнее
*отправленное* агентом действие (его агент знает точно).
Наблюдение = последние `history` кадров (новый первым) [+ привилегированные признаки для оракула].
Награда и метрики считаются по истинному состоянию, а не по наблюдению.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from .conditions import TRAIN_NAMESPACE, TRAIN_SPACE, ConditionSpace, EpisodeCondition
from . import kernels
from .dynamics import QuadDynamics
from .params import CF2X, GRAVITY, QuadParams
from .rotations import euler_to_quat, quat_to_rotmat, tilt_angle
from .wind import generate_wind

STATE_DIM = 18          # e_p, v, R, ω
TERMINATION_REASONS = ("nonfinite", "distance", "ground", "tilt", "rate")
PRIV_DIM = 13           # ветер (3), масса, η (4), смещение ЦМ (2), τ_мотора, задержки (2)
RATE_SCALE = 5.0


@dataclass
class EnvConfig:
    ctrl_freq: int = 50
    sim_freq: int = 500
    episode_seconds: float = 15.0
    action_mode: str = "ctbr"                 # "ctbr" | "attitude"
    rate_max: tuple = (5.0, 5.0, 2.5)         # рад/с, CTBR
    tilt_max: float = 0.6                     # рад, режим attitude
    yaw_rate_max: float = 2.5                 # рад/с, режим attitude
    rate_gains: tuple = (25.0, 25.0, 20.0)      # ζ ≥ 0.4 при τ_мотора до 0.07 с
    att_gains: tuple = (8.0, 8.0, 3.0)        # внутренний P по ориентации (режим attitude)
    target: tuple = (0.0, 0.0, 1.5)
    history: int = 1
    include_prev_action: bool = True
    obs_integral: bool = False                # «ручная память»: ∫(наблюдаемая e_p)dt, |·| ≤ 1 м·с
    privileged: bool = False
    noise_pos: float = 0.005                  # м
    noise_vel: float = 0.03                   # м/с
    noise_att: float = 0.01                   # рад
    noise_rate: float = 0.05                  # рад/с
    max_delay: int = 4
    # награда: r = r_pos + 1 − tanh(затраты) ∈ [0, 2]; при падении r = −crash_penalty
    # (веса выбраны перебором на val, см. PLAN.md, журнал)
    w_pos_far: float = 0.5
    sigma_far: float = 0.5
    w_pos_near: float = 0.5
    sigma_near: float = 0.1
    w_vel: float = 0.05
    w_rate: float = 0.03
    w_dact: float = 0.5
    w_act: float = 0.02
    w_yaw: float = 0.1
    w_tilt: float = 0.3                       # штраф за наклон tilt² (рад²)
    cost_saturation: str = "tanh"             # "tanh": tanh(c) — градиент есть и при больших c; "clip": min(c, 1)
    crash_penalty: float = 50.0
    # прерывание эпизода
    max_dist: float = 2.0
    min_z: float = 0.1
    max_tilt: float = 1.2
    max_rate: float = 25.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "EnvConfig":
        d = dict(d or {})
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise KeyError(f"unknown EnvConfig keys: {sorted(unknown)}")
        for k in ("rate_max", "rate_gains", "att_gains", "target"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)

    @property
    def frame_dim(self) -> int:
        return STATE_DIM + (4 if self.include_prev_action else 0)

    @property
    def obs_dim(self) -> int:
        return self.history * self.frame_dim + (3 if self.obs_integral else 0) + (PRIV_DIM if self.privileged else 0)


class HoverCore:
    """N независимых эпизодов, шагающих синхронно. Без автосброса: им управляет обёртка."""

    def __init__(self, cfg: EnvConfig, n: int, params: QuadParams = CF2X):
        if cfg.sim_freq % cfg.ctrl_freq:
            raise ValueError("sim_freq must be a multiple of ctrl_freq")
        self.cfg, self.n, self.p = cfg, n, params
        self.dt = 1.0 / cfg.ctrl_freq
        self.substeps = cfg.sim_freq // cfg.ctrl_freq
        self.max_steps = int(round(cfg.episode_seconds * cfg.ctrl_freq))
        self.dyn = QuadDynamics(params, n, 1.0 / cfg.sim_freq)
        self.rate_gains = np.asarray(cfg.rate_gains, dtype=float)
        self.mode = {"ctbr": kernels.MODE_CTBR, "attitude": kernels.MODE_ATTITUDE}[cfg.action_mode]
        self.target = np.asarray(cfg.target, dtype=float)
        self.rate_max = np.asarray(cfg.rate_max, dtype=float)
        self.att_gains = np.asarray(cfg.att_gains, dtype=float)
        self.hover_rotor = params.hover_rotor_speed
        self.noise_std = np.repeat([cfg.noise_pos, cfg.noise_vel, cfg.noise_att, cfg.noise_rate], 3)

        T, D = self.max_steps, cfg.max_delay
        self.step_count = np.zeros(n, dtype=int)
        self.act_delay = np.zeros(n, dtype=int)
        self.obs_delay = np.zeros(n, dtype=int)
        self.noise_scale = np.ones(n)
        self.wind = np.zeros((n, T + 1, 3))
        self.wind_slow = np.zeros((n, T + 1, 3))
        self.noise = np.zeros((n, T + 1, 12))
        self.act_buf = np.zeros((n, D + 1, 4))
        self.meas_buf = np.zeros((n, D + 1, STATE_DIM))
        self.hist = np.zeros((n, cfg.history, cfg.frame_dim))
        self.err_integral = np.zeros((n, 3))
        self.prev_action = np.zeros((n, 4))
        self.applied_action = np.zeros((n, 4))
        self.motor_tau = np.full(n, params.motor_tau)
        # события: время = inf, если события нет
        self.ev_mass_t = np.full(n, np.inf)
        self.ev_mass_val = np.ones(n)
        self.ev_motor_t = np.full(n, np.inf)
        self.ev_motor_idx = np.zeros(n, dtype=int)
        self.ev_motor_eff = np.ones(n)
        self.ev_push_t = np.full(n, np.inf)
        self.ev_push_dur = np.zeros(n)
        self.ev_push_force = np.zeros((n, 3))
        self.conditions: list[EpisodeCondition | None] = [None] * n
        self.term_reason = np.full(n, -1)
        # интерактивный режим: если задан, этот ветер (N,3) или (3,) используется вместо ряда эпизода
        self.wind_override: np.ndarray | None = None
        self.wind_applied = np.zeros((n, 3))
        self._rows = np.arange(n)

    # ------------------------------------------------------------------ reset
    def reset_envs(self, idx, conditions: list[EpisodeCondition]) -> None:
        idx = np.atleast_1d(np.asarray(idx, dtype=int))
        for i, c in zip(idx, conditions):
            self._reset_one(int(i), c)
        self.dyn.R[idx] = quat_to_rotmat(self.dyn.quat[idx])
        meas = self._measure(idx)
        self.meas_buf[idx] = meas[:, None, :]
        frame = self._frame(meas, self.prev_action[idx])
        self.hist[idx] = frame[:, None, :]

    def _reset_one(self, i: int, c: EpisodeCondition) -> None:
        cfg, p, T = self.cfg, self.p, self.max_steps
        rng = np.random.default_rng(list(c.seed))
        # начальное состояние: равномерно в шаре радиуса init_radius
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction) + 1e-12
        offset = direction * c.init_radius * rng.uniform() ** (1.0 / 3.0)
        vel = rng.standard_normal(3) * c.init_vel_std
        rp = np.clip(rng.standard_normal(2) * c.init_att_std, -3 * c.init_att_std, 3 * c.init_att_std)
        yaw = rng.uniform(*c.init_yaw) if c.init_yaw[1] > c.init_yaw[0] else c.init_yaw[0]
        quat = euler_to_quat(np.array(rp[0]), np.array(rp[1]), np.array(yaw))
        omega = rng.standard_normal(3) * c.init_rate_std
        self.dyn.reset_state(i, self.target + offset, vel, quat, omega, self.hover_rotor)

        self.dyn.mass[i] = c.mass_ratio * p.mass
        self.dyn.motor_eff[i] = c.motor_eff
        self.dyn.com_offset[i] = c.com_offset
        self.dyn.ext_force[i] = 0.0
        self.motor_tau[i] = c.motor_tau
        self.dyn.set_motor_tau(c.motor_tau, i)

        w, w_slow = generate_wind(c.wind, T + 1, self.dt, rng, altitude=float(self.target[2]))
        self.wind[i], self.wind_slow[i] = w, w_slow
        self.wind_applied[i] = w[0]
        self.noise[i] = rng.standard_normal((T + 1, 12)) * self.noise_std * c.noise_scale
        self.noise_scale[i] = c.noise_scale

        self.act_delay[i] = min(c.act_delay, cfg.max_delay)
        self.obs_delay[i] = min(c.obs_delay, cfg.max_delay)
        self.act_buf[i] = 0.0
        self.prev_action[i] = 0.0
        self.err_integral[i] = 0.0
        self.applied_action[i] = 0.0
        self.step_count[i] = 0

        self.ev_mass_t[i], self.ev_mass_val[i] = c.mass_event if c.mass_event else (np.inf, 1.0)
        if c.motor_event:
            self.ev_motor_t[i], self.ev_motor_idx[i], self.ev_motor_eff[i] = c.motor_event
        else:
            self.ev_motor_t[i] = np.inf
        if c.push_event:
            self.ev_push_t[i], self.ev_push_dur[i] = c.push_event[0], c.push_event[1]
            self.ev_push_force[i] = c.push_event[2]
        else:
            self.ev_push_t[i] = np.inf
        self.conditions[i] = c

    # ------------------------------------------------------------ observation
    def _measure(self, idx=slice(None)) -> np.ndarray:
        """Измерение истинного состояния с шумом текущего шага: (len(idx), 18)."""
        d = self.dyn
        rows = self._rows[idx]
        noise = self.noise[rows, np.minimum(self.step_count[rows], self.max_steps)]
        out = np.empty((rows.size, STATE_DIM))
        kernels.measure(d.pos[rows], d.vel[rows], d.quat[rows], d.omega[rows], self.target, noise,
                        RATE_SCALE, out)
        return out

    def _frame(self, meas: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
        if self.cfg.include_prev_action:
            return np.concatenate([meas, prev_action], axis=1)
        return meas

    def _wind_now(self) -> np.ndarray:
        if self.wind_override is not None:
            return np.broadcast_to(np.asarray(self.wind_override, dtype=float), (self.n, 3)).copy()
        return self.wind[self._rows, np.minimum(self.step_count, self.max_steps)]

    def privileged(self) -> np.ndarray:
        d = self.dyn
        wind_now = self._wind_now()
        return np.concatenate([
            wind_now / 5.0,
            (d.mass / self.p.mass - 1.0)[:, None],
            d.motor_eff - 1.0,
            d.com_offset / 0.002,
            ((self.motor_tau - 0.02) / 0.02)[:, None],
            self.act_delay[:, None] / 2.0,
            self.obs_delay[:, None] / 2.0,
        ], axis=1)

    def observe(self) -> np.ndarray:
        obs = self.hist.reshape(self.n, -1)
        if self.cfg.obs_integral:
            obs = np.concatenate([obs, self.err_integral], axis=1)
        if self.cfg.privileged:
            obs = np.concatenate([obs, self.privileged()], axis=1)
        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

    # ------------------------------------------------------------------- step
    def _apply_events(self) -> None:
        t = self.step_count * self.dt
        d = self.dyn
        hit = t >= self.ev_mass_t
        if hit.any():
            d.mass[hit] = self.ev_mass_val[hit] * self.p.mass
            self.ev_mass_t[hit] = np.inf
        hit = t >= self.ev_motor_t
        if hit.any():
            rows = np.flatnonzero(hit)
            d.motor_eff[rows, self.ev_motor_idx[rows]] = self.ev_motor_eff[rows]
            self.ev_motor_t[hit] = np.inf
        push_on = (t >= self.ev_push_t) & (t < self.ev_push_t + self.ev_push_dur)
        d.ext_force = np.where(push_on[:, None], self.ev_push_force, 0.0)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg, d, rows = self.cfg, self.dyn, self._rows
        a = np.clip(np.asarray(action, dtype=float).reshape(self.n, 4), -1.0, 1.0)
        size = cfg.max_delay + 1
        self.act_buf[rows, self.step_count % size] = a
        applied = self.act_buf[rows, (self.step_count - self.act_delay) % size]
        early = self.step_count < self.act_delay      # до первой команды — висение
        applied[early] = 0.0
        self.applied_action = applied
        self._apply_events()

        thrust = self.p.weight * (1.0 + applied[:, 0])
        wind_now = self._wind_now()
        self.wind_applied = wind_now.copy()
        if self.mode == kernels.MODE_CTBR:
            setpoint = applied[:, 1:] * self.rate_max
        else:
            setpoint = applied[:, 1:] * np.array([cfg.tilt_max, cfg.tilt_max, cfg.yaw_rate_max])
        d.run(self.substeps, self.mode, thrust, setpoint, wind_now, self.rate_gains, self.att_gains)

        self.step_count += 1
        e = d.pos - self.target
        dist = np.linalg.norm(e, axis=1)
        tilt = tilt_angle(d.R)
        rate = np.linalg.norm(d.omega, axis=1)
        finite = np.isfinite(d.pos).all(axis=1) & np.isfinite(d.quat).all(axis=1) & np.isfinite(d.omega).all(axis=1)
        reasons = np.stack([~finite, dist > cfg.max_dist, d.pos[:, 2] < cfg.min_z, tilt > cfg.max_tilt,
                            rate > cfg.max_rate], axis=1)
        terminated = reasons.any(axis=1)
        # причина прерывания для диагностики: индекс в TERMINATION_REASONS, −1 — эпизод продолжается
        self.term_reason = np.where(terminated, np.argmax(reasons, axis=1), -1)
        truncated = (self.step_count >= self.max_steps) & ~terminated

        r_pos = cfg.w_pos_far * np.exp(-(dist / cfg.sigma_far) ** 2) \
            + cfg.w_pos_near * np.exp(-(dist / cfg.sigma_near) ** 2)
        yaw = np.arctan2(d.R[:, 1, 0], d.R[:, 0, 0])
        cost = cfg.w_vel * np.sum(d.vel**2, axis=1) + cfg.w_rate * rate**2 \
            + cfg.w_dact * np.sum((a - self.prev_action) ** 2, axis=1) + cfg.w_act * np.sum(a**2, axis=1) \
            + cfg.w_yaw * (1.0 - np.cos(yaw)) + cfg.w_tilt * tilt**2
        cost = np.tanh(cost) if cfg.cost_saturation == "tanh" else np.minimum(cost, 1.0)
        reward = r_pos + 1.0 - cost
        reward = np.where(terminated, -cfg.crash_penalty, reward)
        reward = np.nan_to_num(reward, nan=-cfg.crash_penalty)

        self.prev_action = a
        if finite.all():
            meas = self._measure()
        else:
            meas = np.nan_to_num(self._measure(), nan=0.0, posinf=10.0, neginf=-10.0)
        size = cfg.max_delay + 1
        self.meas_buf[rows, self.step_count % size] = meas
        delayed = self.meas_buf[rows, np.maximum(self.step_count - self.obs_delay, 0) % size]
        self.hist[:, 1:] = self.hist[:, :-1]
        self.hist[:, 0] = self._frame(delayed, a)
        if cfg.obs_integral:
            self.err_integral = np.clip(self.err_integral + delayed[:, 0:3] * self.dt, -1.0, 1.0)
        return reward, terminated, truncated

    # ------------------------------------------------------------- truth view
    def true_state(self) -> dict[str, np.ndarray]:
        """Истинное состояние для метрик и анализа (копии)."""
        d, rows = self.dyn, self._rows
        idx = np.minimum(self.step_count, self.max_steps)
        return {
            "pos_err": d.pos - self.target, "vel": d.vel.copy(), "omega": d.omega.copy(),
            "tilt": tilt_angle(d.R), "action": self.prev_action.copy(), "applied": self.applied_action.copy(),
            "thrust_ratio": d.thrusts.sum(axis=1) / (d.mass * GRAVITY),
            "wind": self.wind_applied.copy(),
            "wind_slow": self.wind_slow[rows, np.maximum(idx - 1, 0)].copy(),
            "mass_ratio": d.mass / self.p.mass, "motor_eff": d.motor_eff.copy(),
        }


def _box(dim: int) -> spaces.Box:
    return spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)


ACTION_SPACE = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)


class QuadHoverVecEnv(VecEnv):
    """SB3 VecEnv над HoverCore. Условия берутся из пространства (обучение) или из фиксированного списка.

    Каждой среде i соответствует свой поток случайных чисел [TRAIN_NAMESPACE, seed, i],
    поэтому разные сиды обучения не делят условия между собой.
    """

    def __init__(self, cfg: EnvConfig, num_envs: int, space: ConditionSpace | None = TRAIN_SPACE,
                 conditions: list[EpisodeCondition] | None = None, seed: int = 0, difficulty: float = 1.0):
        self.core = HoverCore(cfg, num_envs)
        self.cfg = cfg
        self.render_mode = None
        self.difficulty = float(difficulty)
        self._space = space
        self._fixed = conditions
        self._init_rngs(seed)
        self._actions = None
        super().__init__(num_envs, _box(cfg.obs_dim), ACTION_SPACE)

    def _init_rngs(self, seed: int) -> None:
        self._env_rngs = [np.random.default_rng([TRAIN_NAMESPACE, int(seed), i]) for i in range(self.core.n)]

    def _next_conditions(self, idx) -> list[EpisodeCondition]:
        if self._fixed is not None:
            return [self._fixed[i % len(self._fixed)] for i in idx]
        return [self._space.sample(self._env_rngs[i], self.difficulty) for i in idx]

    def reset(self):
        if self._seeds[0] is not None:
            self._init_rngs(self._seeds[0])
        self._reset_seeds()
        self._reset_options()
        idx = np.arange(self.num_envs)
        self.core.reset_envs(idx, self._next_conditions(idx))
        self.reset_infos = [{} for _ in range(self.num_envs)]
        return self.core.observe()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        reward, terminated, truncated = self.core.step(self._actions)
        obs = self.core.observe()
        dones = terminated | truncated
        infos = [{} for _ in range(self.num_envs)]
        done_idx = np.flatnonzero(dones)
        if done_idx.size:
            for i in done_idx:
                infos[i]["terminal_observation"] = obs[i].copy()
                infos[i]["TimeLimit.truncated"] = bool(truncated[i])
                infos[i]["crashed"] = bool(terminated[i])
            self.core.reset_envs(done_idx, self._next_conditions(done_idx))
            obs[done_idx] = self.core.observe()[done_idx]
        return obs, reward.astype(np.float32), dones, infos

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices=None) -> list:
        return [getattr(self, attr_name)] * len(self._get_indices(indices))

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list:
        return [getattr(self, method_name)(*method_args, **method_kwargs) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False] * len(self._get_indices(indices))


class QuadHoverEnv(gym.Env):
    """Одна среда с API gymnasium (для проверок, визуализации и внешних библиотек)."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig | None = None, space: ConditionSpace = TRAIN_SPACE,
                 difficulty: float = 1.0):
        self.cfg = cfg or EnvConfig()
        self.core = HoverCore(self.cfg, 1)
        self.space = space
        self.difficulty = difficulty
        self.observation_space = _box(self.cfg.obs_dim)
        self.action_space = ACTION_SPACE

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        cond = (options or {}).get("condition")
        if cond is None:
            cond = self.space.sample(self.np_random, self.difficulty)
        self.core.reset_envs([0], [cond])
        return self.core.observe()[0], {}

    def step(self, action):
        reward, terminated, truncated = self.core.step(np.asarray(action)[None])
        info = {"crashed": bool(terminated[0])}
        return self.core.observe()[0], float(reward[0]), bool(terminated[0]), bool(truncated[0]), info
