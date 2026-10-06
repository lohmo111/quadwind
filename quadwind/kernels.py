"""JIT-ядра (numba) внутреннего цикла: регулятор угловых скоростей → миксер → моторы → физика.

Это единственная реализация физики и нижних контуров: и среда, и тесты вызывают эти функции.
Уравнения описаны в `dynamics.py`. Для отладки JIT можно отключить: переменная окружения
NUMBA_DISABLE_JIT=1, тогда тот же код выполнится как обычный Python.
"""
from __future__ import annotations

from math import atan2, cos, sin, sqrt

import numpy as np
from numba import njit

MODE_CTBR = 0
MODE_ATTITUDE = 1


@njit(cache=True)
def quat_to_R(q, R):
    w, x, y, z = q[0], q[1], q[2], q[3]
    R[0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[0, 1] = 2.0 * (x * y - w * z)
    R[0, 2] = 2.0 * (x * z + w * y)
    R[1, 0] = 2.0 * (x * y + w * z)
    R[1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[1, 2] = 2.0 * (y * z - w * x)
    R[2, 0] = 2.0 * (x * z - w * y)
    R[2, 1] = 2.0 * (y * z + w * x)
    R[2, 2] = 1.0 - 2.0 * (x * x + y * y)


@njit(cache=True)
def quat_integrate(q, wx, wy, wz, dt):
    """q ← q ⊗ exp(ω·dt), нормировка."""
    hx, hy, hz = 0.5 * dt * wx, 0.5 * dt * wy, 0.5 * dt * wz
    a = sqrt(hx * hx + hy * hy + hz * hz)
    k = sin(a) / a if a > 1e-9 else 1.0 - a * a / 6.0
    dw, dx, dy, dz = cos(a), k * hx, k * hy, k * hz
    w, x, y, z = q[0], q[1], q[2], q[3]
    nw = w * dw - x * dx - y * dy - z * dz
    nx = w * dx + x * dw + y * dz - z * dy
    ny = w * dy - x * dz + y * dw + z * dx
    nz = w * dz + x * dy - y * dx + z * dw
    inv = 1.0 / sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
    q[0], q[1], q[2], q[3] = nw * inv, nx * inv, ny * inv, nz * inv


@njit(cache=True)
def mix_one(thrust, tx, ty, tz, a_inv, f_max, out):
    """[T, τ] → тяги моторов с приоритетом: крен/тангаж > рыскание > суммарная тяга."""
    rp = np.empty(4)
    yw = np.empty(4)
    for m in range(4):
        rp[m] = a_inv[m, 1] * tx + a_inv[m, 2] * ty
        yw[m] = a_inv[m, 3] * tz
    spread_rp = rp.max() - rp.min()
    if spread_rp > f_max:
        s = f_max / spread_rp
        for m in range(4):
            rp[m] *= s
        spread_rp = f_max
    spread_y = yw.max() - yw.min()
    k = 1.0
    if spread_y > 1e-12:
        k = min(max((f_max - spread_rp) / spread_y, 0.0), 1.0)
    lo, hi = 1e300, -1e300
    for m in range(4):
        rp[m] += k * yw[m]
        lo = min(lo, rp[m])
        hi = max(hi, rp[m])
    lo, hi = -lo, f_max - hi
    hi = max(hi, lo)
    base = min(max(0.25 * thrust, lo), hi)
    for m in range(4):
        out[m] = min(max(base + rp[m], 0.0), f_max)


@njit(cache=True)
def mix_batch(thrust, torque, a_inv, f_max):
    n = thrust.shape[0]
    out = np.empty((n, 4))
    for i in range(n):
        mix_one(thrust[i], torque[i, 0], torque[i, 1], torque[i, 2], a_inv, f_max, out[i])
    return out


@njit(cache=True)
def physics_one(i, dt, cmd, pos, vel, quat, omega, rotor, R, thrusts, acc, aero,
                mass, eff, alpha, com, ext, wind, kf, torque_map, max_rotor, inertia,
                rotor_drag, body_drag, g):
    """Один шаг физики дрона i при командах оборотов cmd (4,) рад/с."""
    thrust, tx, ty, tz, rsum = 0.0, 0.0, 0.0, 0.0, 0.0
    for m in range(4):
        c = min(max(cmd[m], 0.0), max_rotor)
        rotor[i, m] += alpha[i] * (c - rotor[i, m])
        r = rotor[i, m]
        f = kf * eff[i, m] * r * r
        thrusts[i, m] = f
        thrust += f
        rsum += r
        tx += f * torque_map[m, 0]
        ty += f * torque_map[m, 1]
        tz += f * torque_map[m, 2]
    tx -= com[i, 1] * thrust
    ty += com[i, 0] * thrust

    Ri = R[i]
    vrx, vry, vrz = vel[i, 0] - wind[i, 0], vel[i, 1] - wind[i, 1], vel[i, 2] - wind[i, 2]
    bx = Ri[0, 0] * vrx + Ri[1, 0] * vry + Ri[2, 0] * vrz        # Rᵀ·v_rel
    by = Ri[0, 1] * vrx + Ri[1, 1] * vry + Ri[2, 1] * vrz
    bz = Ri[0, 2] * vrx + Ri[1, 2] * vry + Ri[2, 2] * vrz
    sp = sqrt(bx * bx + by * by + bz * bz)
    fx = -(rsum * rotor_drag[0] + body_drag[0] * sp) * bx
    fy = -(rsum * rotor_drag[1] + body_drag[1] * sp) * by
    fz = -(rsum * rotor_drag[2] + body_drag[2] * sp) * bz
    for k in range(3):
        aero[i, k] = Ri[k, 0] * fx + Ri[k, 1] * fy + Ri[k, 2] * fz

    inv_m = 1.0 / mass[i]
    for k in range(3):
        acc[i, k] = (Ri[k, 2] * thrust + aero[i, k] + ext[i, k]) * inv_m
    acc[i, 2] -= g

    jx, jy, jz = inertia[0], inertia[1], inertia[2]
    wx, wy, wz = omega[i, 0], omega[i, 1], omega[i, 2]
    dwx = (tx - (wy * jz * wz - wz * jy * wy)) / jx                 # J·ω̇ = τ − ω × Jω
    dwy = (ty - (wz * jx * wx - wx * jz * wz)) / jy
    dwz = (tz - (wx * jy * wy - wy * jx * wx)) / jz

    for k in range(3):
        vel[i, k] += acc[i, k] * dt
        pos[i, k] += vel[i, k] * dt
    omega[i, 0] = wx + dwx * dt
    omega[i, 1] = wy + dwy * dt
    omega[i, 2] = wz + dwz * dt
    quat_integrate(quat[i], omega[i, 0], omega[i, 1], omega[i, 2], dt)
    quat_to_R(quat[i], Ri)


@njit(cache=True)
def physics_batch(dt, cmd, pos, vel, quat, omega, rotor, R, thrusts, acc, aero,
                  mass, eff, alpha, com, ext, wind, kf, torque_map, max_rotor, inertia,
                  rotor_drag, body_drag, g):
    for i in range(pos.shape[0]):
        physics_one(i, dt, cmd[i], pos, vel, quat, omega, rotor, R, thrusts, acc, aero,
                    mass, eff, alpha, com, ext, wind, kf, torque_map, max_rotor, inertia,
                    rotor_drag, body_drag, g)


@njit(cache=True)
def attitude_rates(q, roll_d, pitch_d, gains, out):
    """P по ориентации: ω* = K ⊙ log(q⁻¹ ⊗ q_des), q_des = (roll_d, pitch_d, текущее рыскание)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    yaw = atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))
    cr, sr = cos(0.5 * roll_d), sin(0.5 * roll_d)
    cp, sp = cos(0.5 * pitch_d), sin(0.5 * pitch_d)
    cy, sy = cos(0.5 * yaw), sin(0.5 * yaw)
    dw = cr * cp * cy + sr * sp * sy
    dx = sr * cp * cy - cr * sp * sy
    dy = cr * sp * cy + sr * cp * sy
    dz = cr * cp * sy - sr * sp * cy
    # e = q⁻¹ ⊗ q_des
    ew = w * dw + x * dx + y * dy + z * dz
    ex = w * dx - x * dw - y * dz + z * dy
    ey = w * dy + x * dz - y * dw - z * dx
    ez = w * dz - x * dy + y * dx - z * dw
    if ew < 0.0:
        ew, ex, ey, ez = -ew, -ex, -ey, -ez
    s = sqrt(ex * ex + ey * ey + ez * ez)
    k = 2.0 * atan2(s, ew) / s if s > 1e-12 else 2.0
    out[0] = gains[0] * k * ex
    out[1] = gains[1] * k * ey
    out[2] = gains[2] * k * ez


@njit(cache=True)
def inner_loop(n_sub, dt, mode, thrust_cmd, setpoint, wind,
               pos, vel, quat, omega, rotor, R, thrusts, acc, aero,
               mass, eff, alpha, com, ext,
               kf, torque_map, a_inv, f_max, max_rotor, inertia, rate_gains, att_gains,
               rotor_drag, body_drag, g, rate_ki, rate_int, rate_int_max):
    """n_sub шагов физики с нижними контурами на каждом шаге.

    mode = MODE_CTBR:     setpoint = желаемые угловые скорости (рад/с)
    mode = MODE_ATTITUDE: setpoint = (крен, тангаж, скорость рыскания)
    rate_ki > 0 включает интегратор в контуре угловых скоростей (как I-член реального полётного
    контроллера): он компенсирует постоянные моменты — ослабленный мотор, смещение центра масс.
    """
    n = pos.shape[0]
    wd = np.empty(3)
    f_cmd = np.empty(4)
    cmd = np.empty(4)
    for i in range(n):
        for _ in range(n_sub):
            if mode == MODE_ATTITUDE:
                attitude_rates(quat[i], setpoint[i, 0], setpoint[i, 1], att_gains, wd)
                wd[2] = setpoint[i, 2]
            else:
                wd[0], wd[1], wd[2] = setpoint[i, 0], setpoint[i, 1], setpoint[i, 2]
            wx, wy, wz = omega[i, 0], omega[i, 1], omega[i, 2]
            jx, jy, jz = inertia[0], inertia[1], inertia[2]
            for k in range(3):
                if rate_ki[k] > 0.0:
                    v = rate_int[i, k] + (wd[k] - omega[i, k]) * dt
                    rate_int[i, k] = min(max(v, -rate_int_max), rate_int_max)
            tx = jx * (rate_gains[0] * (wd[0] - wx) + rate_ki[0] * rate_int[i, 0]) + (wy * jz * wz - wz * jy * wy)
            ty = jy * (rate_gains[1] * (wd[1] - wy) + rate_ki[1] * rate_int[i, 1]) + (wz * jx * wx - wx * jz * wz)
            tz = jz * (rate_gains[2] * (wd[2] - wz) + rate_ki[2] * rate_int[i, 2]) + (wx * jy * wy - wy * jx * wx)
            mix_one(thrust_cmd[i], tx, ty, tz, a_inv, f_max, f_cmd)
            for m in range(4):
                cmd[m] = sqrt(f_cmd[m] / kf)
            physics_one(i, dt, cmd, pos, vel, quat, omega, rotor, R, thrusts, acc, aero,
                        mass, eff, alpha, com, ext, wind, kf, torque_map, max_rotor, inertia,
                        rotor_drag, body_drag, g)



@njit(cache=True)
def measure(pos, vel, quat, omega, target, noise, rate_scale, out):
    """Измерения с шумом: [p − p*, v, R(q ⊗ exp(n_att)) (9), ω/rate_scale] → out (N, 18)."""
    q_noisy = np.empty(4)
    Rm = np.empty((3, 3))
    for i in range(pos.shape[0]):
        nz = noise[i]
        for k in range(3):
            out[i, k] = pos[i, k] - target[k] + nz[k]
            out[i, 3 + k] = vel[i, k] + nz[3 + k]
            out[i, 15 + k] = (omega[i, k] + nz[9 + k]) / rate_scale
        for k in range(4):
            q_noisy[k] = quat[i, k]
        quat_integrate(q_noisy, nz[6], nz[7], nz[8], 1.0)   # q ⊗ exp(n_att): поворот на малый угол
        quat_to_R(q_noisy, Rm)
        for r in range(3):
            for c in range(3):
                out[i, 6 + 3 * r + c] = Rm[r, c]
