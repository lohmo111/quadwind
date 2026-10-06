"""Физика, моторы, миксер и нижние контуры: знаки, равновесия, точность интегрирования."""
import numpy as np
import pytest

from quadwind import kernels
from quadwind.control import Mixer
from quadwind.dynamics import QuadDynamics
from quadwind.params import CF2X, GRAVITY
from quadwind.rotations import (euler_to_quat, integrate_body_rates, quat_to_rotmat, quat_to_rotvec,
                                rotmat_to_euler, rotvec_to_quat)

DT = 1.0 / 500.0
NO_WIND = np.zeros((1, 3))


def make(n=1, dt=DT):
    return QuadDynamics(CF2X, n, dt)


def rotor_for_thrust(per_motor_thrust):
    return np.sqrt(np.asarray(per_motor_thrust) / CF2X.kf)


def test_params_match_gym_pybullet_drones():
    # обороты висения CF2X в gym-pybullet-drones ≈ 14 468 RPM при g = 9.8
    hover_rpm = CF2X.hover_rotor_speed * 60 / (2 * np.pi)
    assert hover_rpm == pytest.approx(14468 * np.sqrt(9.81 / 9.8), rel=1e-3)
    assert CF2X.max_thrust_per_motor * 4 == pytest.approx(2.25 * CF2X.weight)


def test_hover_is_equilibrium():
    d = make()
    cmd = np.full((1, 4), CF2X.hover_rotor_speed)
    for _ in range(500):
        d.step(cmd, NO_WIND)
    assert np.abs(d.pos).max() < 1e-9
    assert np.abs(d.omega).max() < 1e-9


def test_free_fall_and_thrust_step():
    d = make(2)
    d.rotor[0] = 0.0
    d.rotor[1] = rotor_for_thrust(1.5 * CF2X.weight / 4)
    cmd = d.rotor.copy()
    d.step(cmd, np.zeros((2, 3)))
    assert d.acc[0, 2] == pytest.approx(-GRAVITY, rel=1e-6)
    assert d.acc[1, 2] == pytest.approx(0.5 * GRAVITY, rel=1e-6)


@pytest.mark.parametrize("motors, axis", [((2, 3), 0), ((1, 2), 1), ((1, 3), 2)])
def test_torque_signs(motors, axis):
    """Больше тяги на моторах с y>0 → +τx; с x<0 → +τy; на роторах по часовой → +τz."""
    d = make()
    f = np.full(4, CF2X.weight / 4)
    f[list(motors)] *= 1.1
    d.rotor[:] = rotor_for_thrust(f)
    d.step(d.rotor.copy(), NO_WIND)
    assert d.omega[0, axis] > 0
    others = [k for k in range(3) if k != axis]
    assert np.abs(d.omega[0, others]).max() < 1e-9 + 1e-3 * abs(d.omega[0, axis])


def test_mixer_inverts_allocation_and_respects_limits():
    mixer = Mixer(CF2X)
    rng = np.random.default_rng(0)
    T = rng.uniform(0.5, 1.5, 100) * CF2X.weight
    tau = rng.normal(0, [2e-4, 2e-4, 2e-5], (100, 3))
    f = mixer.thrusts(T, tau)
    back = f @ CF2X.allocation.T
    assert np.allclose(back[:, 0], T, rtol=1e-9)
    assert np.allclose(back[:, 1:], tau, atol=1e-12)
    # насыщение: огромный момент крена — тяги в пределах, знак момента сохранён
    f = mixer.thrusts(np.array([CF2X.weight]), np.array([[1.0, 0.0, 0.0]]))
    assert f.min() >= 0 and f.max() <= CF2X.max_thrust_per_motor + 1e-12
    assert (f @ CF2X.allocation.T)[0, 1] > 0


def test_motor_time_constant():
    d = make()
    d.set_motor_tau(0.03)
    d.rotor[:] = 1000.0
    cmd = np.full((1, 4), 2000.0)
    for _ in range(int(round(0.03 / DT))):
        d.step(cmd, NO_WIND)
    assert d.rotor[0, 0] == pytest.approx(1000 + 1000 * (1 - np.exp(-1)), rel=1e-6)


def test_rotation_integration_matches_analytic():
    d = make()
    d.rotor[:] = 0.0
    d.omega[0] = [0.0, 0.0, 2.0]
    for _ in range(500):
        d.step(np.zeros((1, 4)), NO_WIND)
    yaw = rotmat_to_euler(d.R)[0, 2]
    assert yaw == pytest.approx(2.0, abs=1e-6)
    assert np.linalg.norm(d.quat[0]) == pytest.approx(1.0, abs=1e-12)


def test_quaternion_stays_normalized_while_tumbling():
    d = make()
    d.omega[0] = [7.0, -3.0, 11.0]
    for _ in range(20_000):
        d.step(d.rotor.copy(), NO_WIND)
    assert np.linalg.norm(d.quat[0]) == pytest.approx(1.0, abs=1e-9)
    R = d.R[0]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_rotation_helpers_roundtrip():
    rng = np.random.default_rng(1)
    small = rng.normal(0, 0.5, (50, 3))
    assert np.allclose(quat_to_rotvec(rotvec_to_quat(small)), small, atol=1e-9)
    rpy = rng.uniform(-1, 1, (50, 3))
    assert np.allclose(rotmat_to_euler(quat_to_rotmat(euler_to_quat(*rpy.T))), rpy, atol=1e-9)
    q = rotvec_to_quat(small)
    assert np.allclose(integrate_body_rates(np.tile([1.0, 0, 0, 0], (50, 1)), small, 1.0), q, atol=1e-12)


def test_wind_force_direction_and_magnitude():
    d = make()
    wind = np.array([[3.0, 0.0, 0.0]])
    d.step(np.full((1, 4), CF2X.hover_rotor_speed), wind)
    f = d.aero_force[0]
    rsum = 4 * CF2X.hover_rotor_speed
    expected = (rsum * CF2X.rotor_drag_xy + 0.5 * CF2X.air_density * CF2X.body_cda_xy * 3.0) * 3.0
    assert f[0] == pytest.approx(expected, rel=1e-9)
    assert abs(f[1]) < 1e-12 and abs(f[2]) < 1e-12
    # порядок величины: ветер 3 м/с ≈ 0.1–0.2 веса
    assert 0.08 < expected / CF2X.weight < 0.25


def test_com_offset_creates_pitch_torque():
    d = make()
    d.com_offset[0] = [0.002, 0.0]
    d.step(np.full((1, 4), CF2X.hover_rotor_speed), NO_WIND)
    assert d.omega[0, 1] > 0 and abs(d.omega[0, 0]) < 1e-12


def test_integration_converges_with_timestep():
    def simulate(dt):
        d = make(dt=dt)
        d.omega[0] = [1.0, -0.5, 0.3]
        d.vel[0] = [0.3, 0.0, -0.2]
        cmd = np.full((1, 4), CF2X.hover_rotor_speed) * np.array([1.02, 0.99, 1.0, 1.01])
        for _ in range(int(round(1.0 / dt))):
            d.step(cmd, np.array([[2.0, 1.0, 0.0]]))
        return d.pos[0].copy()
    p_ref, p_500, p_250 = simulate(1 / 4000), simulate(1 / 500), simulate(1 / 250)
    err_500, err_250 = np.linalg.norm(p_500 - p_ref), np.linalg.norm(p_250 - p_ref)
    # разомкнутое кувыркание за 1 с: смещение ~2 м, ошибка шага 2 мс < 1% и убывает ~ как dt (метод 1-го порядка)
    assert err_500 < 0.01 * np.linalg.norm(p_ref)
    assert err_250 / err_500 > 1.5


def _inner(d, n_sub, mode, thrust, setpoint):
    d.run(n_sub, mode, np.atleast_1d(thrust), np.atleast_2d(setpoint), NO_WIND,
          np.array([25.0, 25.0, 10.0]), np.array([8.0, 8.0, 3.0]))


def test_rate_loop_tracks_command():
    d = make()
    _inner(d, 150, kernels.MODE_CTBR, CF2X.weight, [1.0, -0.5, 0.3])
    assert np.allclose(d.omega[0], [1.0, -0.5, 0.3], atol=0.02)


def test_attitude_mode_reaches_tilt():
    d = make()
    _inner(d, 500, kernels.MODE_ATTITUDE, CF2X.weight, [0.3, -0.2, 0.0])
    roll, pitch, _ = rotmat_to_euler(d.R)[0]
    assert roll == pytest.approx(0.3, abs=0.01) and pitch == pytest.approx(-0.2, abs=0.01)


def test_weak_motor_is_not_compensated_by_inner_loop():
    """P-регулятор угловых скоростей без интегратора: ослабленный мотор даёт постоянную ошибку по ω."""
    d = make()
    d.motor_eff[0, 0] = 0.8
    _inner(d, 250, kernels.MODE_CTBR, CF2X.weight, [0.0, 0.0, 0.0])
    assert np.linalg.norm(d.omega[0]) > 0.1
