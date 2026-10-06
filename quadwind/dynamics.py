"""Состояние N дронов и вызов JIT-ядер физики.

Модель (мировая система: z вверх; связанная: x вперёд, y влево, z вверх):
  моторы:   ω̇_i = (ω_cmd,i − ω_i) / τ_m (точная дискретизация), f_i = η_i · kf · ω_i², ω_i ∈ [0, ω_max]
  силы:     m·a = R·[0,0,Σf] + F_aero + F_ext + m·g
  аэро:     v_rel = v − w (w — скорость ветра); в связанной системе
            F = −(Σω_i·C_rotor + ½ρ·(C_d A)·|v_rel|) ⊙ v_rel
            (линейное сопротивление винтов, как PYB_DRAG в gym-pybullet-drones, + квадратичное корпуса)
  моменты:  τ = Σ r_i × f_i + реактивные моменты роторов − r_cm × (0,0,Σf)  (смещение центра масс)
            J·ω̇ = τ − ω × Jω
Интегрирование — полунеявный Эйлер; ориентация — экспоненциальное отображение кватерниона.
Сами вычисления — в `kernels.py`.
"""
from __future__ import annotations

import numpy as np

from . import kernels
from .params import GRAVITY, QuadParams
from .rotations import quat_to_rotmat


class QuadDynamics:
    def __init__(self, params: QuadParams, n: int, dt: float):
        self.p = params
        self.n = n
        self.dt = dt
        self.kf = params.kf
        self.inertia = params.inertia
        self.max_rotor = params.max_rotor_speed
        self.f_max = params.max_thrust_per_motor
        self.a_inv = np.linalg.inv(params.allocation)
        pos = params.motor_positions
        # τ = f @ torque_map: столбцы — плечо по y (τx), −плечо по x (τy), реактивный момент на единицу тяги (τz)
        self.torque_map = np.ascontiguousarray(
            np.stack([pos[:, 1], -pos[:, 0], -params.motor_spin * params.km / params.kf], axis=1))
        self.rotor_drag = np.array([params.rotor_drag_xy, params.rotor_drag_xy, params.rotor_drag_z])
        self.body_drag = 0.5 * params.air_density * np.array([params.body_cda_xy, params.body_cda_xy,
                                                               params.body_cda_z])

        # состояние
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))
        self.quat = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
        self.omega = np.zeros((n, 3))
        self.rotor = np.full((n, 4), params.hover_rotor_speed)
        self.R = quat_to_rotmat(self.quat)

        # скрытые параметры (задаются условием эпизода)
        self.mass = np.full(n, params.mass)
        self.motor_eff = np.ones((n, 4))
        self.com_offset = np.zeros((n, 2))
        self.ext_force = np.zeros((n, 3))
        self.motor_alpha = np.empty(n)
        self.set_motor_tau(np.full(n, params.motor_tau))

        # последние вычисленные величины (для логов и тестов)
        self.thrusts = np.zeros((n, 4))
        self.acc = np.zeros((n, 3))
        self.aero_force = np.zeros((n, 3))

    def set_motor_tau(self, tau, idx=slice(None)) -> None:
        self.motor_alpha[idx] = 1.0 - np.exp(-self.dt / np.maximum(np.asarray(tau, dtype=float), 1e-6))

    def reset_state(self, idx, pos, vel, quat, omega, rotor) -> None:
        self.pos[idx] = pos
        self.vel[idx] = vel
        self.quat[idx] = quat
        self.omega[idx] = omega
        self.rotor[idx] = rotor
        self.R[idx] = quat_to_rotmat(self.quat[idx])

    def _common(self):
        return (self.pos, self.vel, self.quat, self.omega, self.rotor, self.R, self.thrusts, self.acc,
                self.aero_force, self.mass, self.motor_eff, self.motor_alpha, self.com_offset, self.ext_force)

    def step(self, rotor_cmd: np.ndarray, wind: np.ndarray) -> None:
        """Один шаг физики при заданных командах оборотов (N,4) рад/с и ветре (N,3) м/с."""
        kernels.physics_batch(self.dt, np.ascontiguousarray(rotor_cmd, dtype=float), *self._common(),
                              np.ascontiguousarray(wind, dtype=float), self.kf, self.torque_map,
                              self.max_rotor, self.inertia, self.rotor_drag, self.body_drag, GRAVITY)

    def run(self, n_sub: int, mode: int, thrust_cmd: np.ndarray, setpoint: np.ndarray, wind: np.ndarray,
            rate_gains: np.ndarray, att_gains: np.ndarray) -> None:
        """n_sub шагов физики с регулятором угловых скоростей и миксером на каждом шаге."""
        kernels.inner_loop(n_sub, self.dt, mode, np.ascontiguousarray(thrust_cmd, dtype=float),
                           np.ascontiguousarray(setpoint, dtype=float), np.ascontiguousarray(wind, dtype=float),
                           *self._common(), self.kf, self.torque_map, self.a_inv, self.f_max, self.max_rotor,
                           self.inertia, rate_gains, att_gains, self.rotor_drag, self.body_drag, GRAVITY)
