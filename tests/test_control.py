"""Классический baseline: PID держит точку, PD — нет при постоянном ветре; train-распределение выполнимо."""
from dataclasses import replace

import numpy as np
import pytest

from quadwind.conditions import make_suite
from quadwind.control import PIDGains
from quadwind.env import EnvConfig
from quadwind.evaluation import rollout
from quadwind.metrics import episode_metrics
from quadwind.policies import PIDPolicy
from quadwind.wind import direction_to_from_deg

CFG = EnvConfig()


def run(gains, suite, n=None):
    return episode_metrics(rollout(PIDPolicy(CFG, gains), CFG, make_suite(suite, n)))


def test_pid_holds_position_in_calm():
    df = run(PIDGains(), "calm", 16)
    assert not df.crashed.any()
    assert df.rmse.median() < 0.03 and df.ss_err.max() < 0.01


def test_integral_action_removes_static_error_pd_does_not():
    pid, pd_ = run(PIDGains(), "wind_sweep"), run(PIDGains().without_integrators(), "wind_sweep")
    speed = pid.cond_wind_speed
    assert pid[speed <= 4.0].ss_err.max() < 0.02
    assert pd_[speed == 3.0].ss_err.min() > 0.1
    # у PD статическая ошибка растёт с силой ветра
    by_speed = pd_.groupby("cond_wind_speed").ss_err.mean()
    assert np.all(np.diff(by_speed.to_numpy()) > 0)


def test_train_distribution_is_feasible_for_pid():
    """Если хорошо настроенный PID падает в train-условиях, диапазоны физически невыполнимы."""
    df = run(PIDGains(), "val")
    assert df.crashed.mean() == 0.0


def test_without_integrators_zeroes_all_integral_gains():
    g = PIDGains().without_integrators()
    assert g.ki_xy == g.ki_z == g.ki_att_rp == g.ki_att_yaw == 0.0
    assert g.kp_xy == PIDGains().kp_xy


@pytest.mark.parametrize("suite", ["mass_pickup", "step_gust"])
def test_pid_recovers_from_events(suite):
    df = run(PIDGains(), suite, 8)
    assert not df.crashed.any()
    assert df.event_recovered.mean() == 1.0


def test_wind_direction_symmetry_and_downwind_drift():
    """X-квадрокоптер симметричен: С и Ю — точное зеркало (поворот на 180° сохраняет схему винтов),
    С и В — почти (отличается только направление вращения винтов). PD сносит по ветру."""
    conds = [replace(c, noise_scale=0.0) for c in make_suite("wind_compass")]
    pid = episode_metrics(rollout(PIDPolicy(CFG, PIDGains()), CFG, conds)).set_index("cond_wind_from_deg")
    assert pid.loc[0.0, "rmse"] == pytest.approx(pid.loc[180.0, "rmse"], rel=1e-6)
    assert pid.loc[90.0, "rmse"] == pytest.approx(pid.loc[270.0, "rmse"], rel=1e-6)
    assert pid.loc[0.0, "rmse"] == pytest.approx(pid.loc[90.0, "rmse"], rel=0.05)
    ro = rollout(PIDPolicy(CFG, PIDGains().without_integrators()), CFG, conds)
    e_end = ro.data["pos_err"][-150:].mean(axis=0)                         # средняя ошибка за 3 с, (N, 3)
    froms = [round(direction_to_from_deg(c.wind.direction)) for c in conds]
    north, west = froms.index(0), froms.index(270)
    assert e_end[north, 1] < -0.05 and abs(e_end[north, 0]) < 0.01        # северный ветер — сносит на юг
    assert e_end[west, 0] > 0.05 and abs(e_end[west, 1]) < 0.01            # западный — на восток
