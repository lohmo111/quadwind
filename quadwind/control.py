"""Миксер и классический baseline (каскадный PID).

Нижние контуры (регулятор угловых скоростей, миксер) реализованы в `kernels.py` и работают
на частоте физики внутри среды. Они знают только номинальные параметры дрона:
ослабленный мотор, другую массу и смещение центра масс должен компенсировать
верхний уровень (агент или PID). Регулятор угловых скоростей — P или PI (`EnvConfig.rate_ki`);
PI компенсирует постоянный момент сам (см. RESEARCH_RESULTS.md, R2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import kernels
from .params import GRAVITY, QuadParams
from .rotations import vee


class Mixer:
    """Желаемые [T, τ] → тяги/обороты моторов (обёртка над JIT-ядром `kernels.mix_one`).

    Приоритет при насыщении: моменты крена/тангажа > рыскание > суммарная тяга.
    """

    def __init__(self, params: QuadParams):
        self.kf = params.kf
        self.f_max = params.max_thrust_per_motor
        self.a_inv = np.linalg.inv(params.allocation)

    def thrusts(self, thrust: np.ndarray, torque: np.ndarray) -> np.ndarray:
        return kernels.mix_batch(np.ascontiguousarray(thrust, dtype=float),
                                 np.ascontiguousarray(torque, dtype=float), self.a_inv, self.f_max)

    def rotor_cmd(self, thrust: np.ndarray, torque: np.ndarray) -> np.ndarray:
        return np.sqrt(self.thrusts(thrust, torque) / self.kf)


@dataclass
class PIDGains:
    kp_xy: float = 5.0
    kd_xy: float = 3.5
    ki_xy: float = 1.5
    kp_z: float = 8.0
    kd_z: float = 5.0
    ki_z: float = 3.0
    k_att_rp: float = 8.0
    k_att_yaw: float = 3.0
    ki_att_rp: float = 3.0     # интегратор по ошибке ориентации: компенсирует ослабленный мотор и смещение ЦМ
    ki_att_yaw: float = 1.0
    i_max: float = 3.0         # ограничение интеграла позиции (м·с), анти-виндап
    i_att_max: float = 1.0     # ограничение интеграла ориентации (рад·с)
    max_tilt: float = 0.7      # рад

    def without_integrators(self) -> "PIDGains":
        """PD-вариант: все интеграторы выключены (статическая ошибка при постоянном возмущении)."""
        return PIDGains(**{**self.__dict__, "ki_xy": 0.0, "ki_z": 0.0, "ki_att_rp": 0.0, "ki_att_yaw": 0.0})

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class PositionPID:
    """Каскадный геометрический PID удержания позиции, выдающий CTBR-действие среды.

    Позиция (PID) → желаемая сила → тяга + желаемая ориентация → P по ориентации → ω*.
    Работает на частоте политики по тем же (шумным, задержанным) наблюдениям, что и агент.
    Интеграторы есть и по позиции, и по ориентации. `gains.without_integrators()` — PD-вариант
    со статической ошибкой при постоянном возмущении.
    """

    def __init__(self, params: QuadParams, gains: PIDGains, dt: float, rate_max: np.ndarray,
                 attitude: tuple | None = None):
        # attitude = (tilt_max, yaw_rate_max): выдавать действие режима attitude (углы + скорость рыскания)
        self.attitude = attitude
        self.m = params.mass
        self.g = gains
        self.dt = dt
        self.rate_max = np.asarray(rate_max, dtype=float)
        self.kp = np.array([gains.kp_xy, gains.kp_xy, gains.kp_z])
        self.kd = np.array([gains.kd_xy, gains.kd_xy, gains.kd_z])
        self.ki = np.array([gains.ki_xy, gains.ki_xy, gains.ki_z])
        self.k_att = np.array([gains.k_att_rp, gains.k_att_rp, gains.k_att_yaw])
        self.ki_att = np.array([gains.ki_att_rp, gains.ki_att_rp, gains.ki_att_yaw])
        self.integral = np.zeros((0, 3))
        self.integral_att = np.zeros((0, 3))

    def reset(self, n: int) -> None:
        self.integral = np.zeros((n, 3))
        self.integral_att = np.zeros((n, 3))

    def reset_envs(self, mask: np.ndarray) -> None:
        self.integral[mask] = 0.0
        self.integral_att[mask] = 0.0

    def act(self, e_pos: np.ndarray, vel: np.ndarray, R: np.ndarray) -> np.ndarray:
        """e_pos = p − p_target (N,3), vel (N,3), R (N,3,3) → действие CTBR в [-1, 1]⁴."""
        self.integral = np.clip(self.integral + e_pos * self.dt, -self.g.i_max, self.g.i_max)
        acc_des = -self.kp * e_pos - self.kd * vel - self.ki * self.integral
        force = self.m * (acc_des + np.array([0.0, 0.0, GRAVITY]))

        # ограничение наклона: |F_xy| ≤ F_z·tan(max_tilt), F_z > 0
        fz = np.maximum(force[:, 2], 0.1 * self.m * GRAVITY)
        fxy = force[:, :2]
        fxy_norm = np.linalg.norm(fxy, axis=1)
        limit = fz * np.tan(self.g.max_tilt)
        fxy = fxy * np.minimum(1.0, limit / np.maximum(fxy_norm, 1e-12))[:, None]
        force = np.concatenate([fxy, fz[:, None]], axis=1)

        b3 = force / np.linalg.norm(force, axis=1, keepdims=True)
        thrust = np.maximum(np.einsum("ni,ni->n", force, R[:, :, 2]), 0.0)
        b1c = np.array([1.0, 0.0, 0.0])                       # желаемое рыскание = 0
        b2 = np.cross(b3, b1c)
        b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
        b1 = np.cross(b2, b3)
        R_des = np.stack([b1, b2, b3], axis=2)

        e_R = 0.5 * vee(np.einsum("nji,njk->nik", R_des, R) - np.einsum("nji,njk->nik", R, R_des))
        self.integral_att = np.clip(self.integral_att + e_R * self.dt, -self.g.i_att_max, self.g.i_att_max)
        omega_des = -self.k_att * e_R - self.ki_att * self.integral_att

        action = np.empty((e_pos.shape[0], 4))
        action[:, 0] = thrust / (self.m * GRAVITY) - 1.0
        if self.attitude is not None:            # углы отрабатывает внутренний контур среды на 500 Гц
            tilt_max, yaw_rate_max = self.attitude
            roll_d = np.arctan2(R_des[:, 2, 1], R_des[:, 2, 2])
            pitch_d = np.arcsin(np.clip(-R_des[:, 2, 0], -1.0, 1.0))
            yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
            action[:, 1] = roll_d / tilt_max
            action[:, 2] = pitch_d / tilt_max
            action[:, 3] = -self.k_att[2] * yaw / yaw_rate_max
        else:
            action[:, 1:] = omega_des / self.rate_max
        return np.clip(action, -1.0, 1.0)
