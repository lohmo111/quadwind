"""Батчевые операции с поворотами.

Кватернион хранится как (w, x, y, z) и задаёт поворот из связанной системы в мировую:
v_world = R(q) · v_body. Все функции принимают массивы формы (..., 4) / (..., 3).
"""
from __future__ import annotations

import numpy as np


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[..., 0, 1] = 2.0 * (xy - wz)
    R[..., 0, 2] = 2.0 * (xz + wy)
    R[..., 1, 0] = 2.0 * (xy + wz)
    R[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[..., 1, 2] = 2.0 * (yz - wx)
    R[..., 2, 0] = 2.0 * (xz - wy)
    R[..., 2, 1] = 2.0 * (yz + wx)
    R[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return R


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Произведение Гамильтона a ⊗ b."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def rotvec_to_quat(v: np.ndarray) -> np.ndarray:
    """Экспоненциальное отображение: вектор поворота (ось·угол) → единичный кватернион."""
    angle = np.linalg.norm(v, axis=-1)
    half = 0.5 * angle
    # sin(θ/2)/θ с безопасным пределом 1/2 при θ → 0
    k = np.where(angle > 1e-12, np.sin(half) / np.maximum(angle, 1e-12), 0.5 - angle**2 / 48.0)
    return np.concatenate([np.cos(half)[..., None], k[..., None] * v], axis=-1)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Логарифм: единичный кватернион → вектор поворота с углом в [0, π]."""
    q = np.where(q[..., :1] < 0.0, -q, q)  # кратчайший поворот
    vec = q[..., 1:]
    s = np.linalg.norm(vec, axis=-1)
    angle = 2.0 * np.arctan2(s, q[..., 0])
    k = np.where(s > 1e-12, angle / np.maximum(s, 1e-12), 2.0)
    return k[..., None] * vec


def integrate_body_rates(q: np.ndarray, omega_body: np.ndarray, dt: float) -> np.ndarray:
    """q(t+dt) = q ⊗ exp(ω·dt) для угловой скорости ω в связанной системе, с нормировкой."""
    half = 0.5 * dt * omega_body
    hx, hy, hz = half[..., 0], half[..., 1], half[..., 2]
    a = np.sqrt(hx * hx + hy * hy + hz * hz)            # половина угла поворота
    k = np.where(a > 1e-9, np.sin(a) / np.maximum(a, 1e-9), 1.0 - a * a / 6.0)
    dw, dx, dy, dz = np.cos(a), k * hx, k * hy, k * hz
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    out = np.empty_like(q)
    out[..., 0] = w * dw - x * dx - y * dy - z * dz
    out[..., 1] = w * dx + x * dw + y * dz - z * dy
    out[..., 2] = w * dy - x * dz + y * dw + z * dx
    out[..., 3] = w * dz + x * dy - y * dx + z * dw
    out /= np.sqrt(np.sum(out * out, axis=-1, keepdims=True))
    return out


def euler_to_quat(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Углы Эйлера ZYX (рыскание → тангаж → крен) → кватернион."""
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    return np.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], axis=-1)


def rotmat_to_euler(R: np.ndarray) -> np.ndarray:
    """Матрица поворота → (крен, тангаж, рыскание) в конвенции ZYX."""
    roll = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    pitch = np.arcsin(np.clip(-R[..., 2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    return np.stack([roll, pitch, yaw], axis=-1)


def tilt_angle(R: np.ndarray) -> np.ndarray:
    """Угол между осью z корпуса и вертикалью."""
    return np.arccos(np.clip(R[..., 2, 2], -1.0, 1.0))


def vee(M: np.ndarray) -> np.ndarray:
    """Обратная к кососимметричной матрице операция: [ω]× → ω."""
    return np.stack([M[..., 2, 1], M[..., 0, 2], M[..., 1, 0]], axis=-1)


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Векторное произведение по последней оси (быстрее np.cross на малых батчах)."""
    out = np.empty(np.broadcast_shapes(a.shape, b.shape))
    out[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    out[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    out[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return out


def norm(v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(v * v, axis=-1))
