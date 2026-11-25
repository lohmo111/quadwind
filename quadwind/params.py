"""Физические параметры Crazyflie 2.x в X-схеме.

Масса, инерция, плечо, коэффициенты тяги/момента и сопротивления винтов взяты из
`cf2x.urdf` в gym-pybullet-drones (utiasDSL), чтобы модель была сопоставима с этой средой.
Там kf и km заданы для оборотов в RPM, здесь они пересчитаны на рад/с.

Аэродинамика корпуса (C_d·A) и постоянная времени моторов — наши допущения
(см. PROJECT.md, Б1): точные значения для Crazyflie известны лишь приблизительно.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81
RPM_TO_RADS = 2.0 * np.pi / 60.0


@dataclass(frozen=True)
class QuadParams:
    mass: float = 0.027                 # кг
    arm: float = 0.0397                 # м, от центра до оси мотора
    kf_rpm: float = 3.16e-10            # Н / RPM²
    km_rpm: float = 7.94e-12            # Н·м / RPM²
    thrust2weight: float = 2.25
    ixx: float = 1.4e-5                 # кг·м²
    iyy: float = 1.4e-5
    izz: float = 2.17e-5
    rotor_drag_xy: float = 9.1785e-7    # Н / (м/с) / (рад/с), линейное сопротивление винтов
    rotor_drag_z: float = 10.311e-7
    body_cda_xy: float = 4.0e-3         # м², C_d·A корпуса вбок (допущение)
    body_cda_z: float = 8.0e-3          # м², C_d·A корпуса вертикально (допущение)
    air_density: float = 1.225          # кг/м³
    motor_tau: float = 0.02             # с, номинальная постоянная времени мотора (допущение)

    # ---- производные величины ----
    @property
    def kf(self) -> float:
        """Коэффициент тяги, Н / (рад/с)²."""
        return self.kf_rpm / RPM_TO_RADS**2

    @property
    def km(self) -> float:
        """Коэффициент реактивного момента, Н·м / (рад/с)²."""
        return self.km_rpm / RPM_TO_RADS**2

    @property
    def weight(self) -> float:
        return self.mass * GRAVITY

    @property
    def max_thrust_per_motor(self) -> float:
        return self.thrust2weight * self.weight / 4.0

    @property
    def max_rotor_speed(self) -> float:
        return float(np.sqrt(self.max_thrust_per_motor / self.kf))

    @property
    def hover_rotor_speed(self) -> float:
        return float(np.sqrt(self.weight / (4.0 * self.kf)))

    @property
    def inertia(self) -> np.ndarray:
        return np.array([self.ixx, self.iyy, self.izz])

    @property
    def motor_positions(self) -> np.ndarray:
        """Координаты моторов (x, y) в связанной системе: x вперёд, y влево, z вверх."""
        d = self.arm / np.sqrt(2.0)
        return np.array([[d, -d], [-d, -d], [-d, d], [d, d]])

    @property
    def motor_spin(self) -> np.ndarray:
        """+1: ротор вращается против часовой стрелки (вид сверху); реактивный момент тогда −z."""
        return np.array([1.0, -1.0, 1.0, -1.0])

    @property
    def allocation(self) -> np.ndarray:
        """Матрица A: [T, τx, τy, τz]ᵀ = A · f, где f — тяги четырёх моторов."""
        pos = self.motor_positions
        yaw_per_thrust = self.km / self.kf
        return np.stack([
            np.ones(4),
            pos[:, 1],                          # τx = Σ y_i f_i
            -pos[:, 0],                         # τy = −Σ x_i f_i
            -self.motor_spin * yaw_per_thrust,  # τz = −Σ s_i (km/kf) f_i
        ])


CF2X = QuadParams()
