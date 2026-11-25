"""Пространства условий и наборы для оценки: диапазоны, curriculum, детерминизм, непересечение с train."""
from dataclasses import fields

import numpy as np
import pytest

from quadwind.conditions import (OOD_SPACES, SCENARIO_SPACES, SUITE_SIZES, TRAIN_SPACE, IntU, U, compass_sectors,
                                 make_suite, suite_names)
from quadwind.wind import direction_to_from_deg


def test_samples_within_train_ranges():
    rng = np.random.default_rng(0)
    for _ in range(500):
        c = TRAIN_SPACE.sample(rng)
        assert 0.8 <= c.mass_ratio <= 1.25
        assert 0.0 <= c.wind.speed <= 3.5
        assert 0.015 <= c.motor_tau <= 0.04
        assert 0 <= c.act_delay <= 1 and 0 <= c.obs_delay <= 1
        assert min(c.motor_eff) >= 0.8 - 1e-12 and max(c.motor_eff) <= 1.0
        assert np.hypot(*c.com_offset) <= 0.002 + 1e-12
        assert c.wind.jump_speed == (0.0, 3.5)


def test_difficulty_zero_is_easy():
    rng = np.random.default_rng(1)
    for _ in range(100):
        c = TRAIN_SPACE.sample(rng, difficulty=0.0)
        assert c.mass_ratio == 1.0 and c.wind.speed == 0.0 and c.wind.turb_sigma == 0.0
        assert c.act_delay == 0 and c.motor_eff == (1.0, 1.0, 1.0, 1.0) and c.init_radius <= 0.2
        assert c.wind.jump_rate == 0.0 and c.wind.gust_rate == 0.0


def test_difficulty_widens_ranges_monotonically():
    spec = U(0.8, 1.25, easy=(1.0, 1.0))
    widths = [np.subtract(*spec.bounds(d)[::-1]) for d in np.linspace(0, 1, 11)]
    assert np.all(np.diff(widths) >= 0)
    assert IntU(0, 2, easy=(0, 0)).bounds(0.5) == (0, 1)


def test_holes_are_respected():
    spec = U(0.0, 1.0, holes=((0.4, 0.6),))
    rng = np.random.default_rng(2)
    xs = np.array([spec.sample(rng) for _ in range(2000)])
    assert not ((xs >= 0.4) & (xs <= 0.6)).any()


def test_suites_are_deterministic_and_sized():
    for name in suite_names():
        a, b = make_suite(name), make_suite(name)
        assert len(a) == {"wind_sweep": 28, "wind_compass": 8}.get(name, SUITE_SIZES.get(name))
        assert [c.to_record() for c in a] == [c.to_record() for c in b]


def test_episode_seeds_are_unique_across_suites():
    seeds = [c.seed for name in suite_names() for c in make_suite(name)]
    assert len(seeds) == len(set(seeds))


@pytest.mark.parametrize("suite", ["test_id", "safety", "wind_sweep", "wind_compass", "ood_heavy", "mass_pickup"])
def test_final_suite_is_disjoint_and_repeatable(suite):
    historical = make_suite(suite, n=3)
    final = make_suite(suite, n=3, namespace="final")
    assert {c.seed for c in final}.isdisjoint(c.seed for c in historical)
    assert final == make_suite(suite, n=3, namespace="final")
    if suite == "test_id":
        assert final[0].mass_ratio != historical[0].mass_ratio


def test_val_and_test_differ():
    val = {c.to_record()["seed"] for c in make_suite("val")}
    test = {c.to_record()["seed"] for c in make_suite("test_id")}
    assert not val & test
    v0, t0 = make_suite("val")[0], make_suite("test_id")[0]
    assert v0.mass_ratio != t0.mass_ratio


def _changed_fields(space):
    return [f.name for f in fields(space) if getattr(space, f.name) != getattr(TRAIN_SPACE, f.name)]


@pytest.mark.parametrize("name", list(OOD_SPACES))
def test_ood_does_not_overlap_train(name):
    """Хотя бы один изменённый параметр каждого OOD-набора лежит целиком вне train-диапазона."""
    space = OOD_SPACES[name]
    changed = _changed_fields(space)
    assert changed, name
    if name == "ood_combined":   # все факторы на краю train-диапазона или за ним
        for f in changed:
            assert getattr(space, f).bounds(1.0)[0] >= getattr(TRAIN_SPACE, f).bounds(1.0)[1], f
        return
    outside = False
    for f in changed:
        ood, train = getattr(space, f), getattr(TRAIN_SPACE, f)
        if isinstance(ood, (U, IntU)):
            lo, hi = ood.bounds(1.0)
            tlo, thi = train.bounds(1.0)
            outside |= hi < tlo or lo > thi or (f == "weak_motor_prob" and lo > thi)
        elif isinstance(ood, str):
            outside |= ood != train
    assert outside, f"{name}: изменённые параметры {changed} пересекаются с train"


def test_scenarios_have_events_at_5s():
    for name in SCENARIO_SPACES:
        if name == "calm":
            continue
        for c in make_suite(name, 4):
            assert c.event_time() == pytest.approx(5.0)


def test_ramp_gust_scenario():
    cond = make_suite("ramp_gust", n=1)[0]
    ev = cond.wind.events[0]
    assert ev.kind == "ramp" and ev.time == pytest.approx(5.0) and ev.dur == pytest.approx(3.0)
    assert np.linalg.norm(ev.vec) == pytest.approx(3.0, abs=1e-9)
    assert cond.event_time() == pytest.approx(5.0)


def test_wind_sweep_grid():
    sweep = make_suite("wind_sweep")
    speeds = sorted({c.wind.speed for c in sweep})
    assert speeds == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert all(c.wind.turb_sigma == 0 and c.wind.gust_rate == 0 for c in sweep)


@pytest.mark.parametrize("sectors, allowed", [
    (((315, 45),), [(0, 45), (315, 360)]),                       # через север
    (((240, 300), (60, 120)), [(60, 120), (240, 300)]),          # запад + восток
])
def test_compass_sectors(sectors, allowed):
    rng = np.random.default_rng(0)
    spec = compass_sectors(*sectors)
    xs = np.array([direction_to_from_deg(spec.sample(rng)) for _ in range(3000)])
    inside = np.zeros_like(xs, dtype=bool)
    for lo, hi in allowed:
        inside |= (xs >= lo - 1e-9) & (xs <= hi + 1e-9)
    assert inside.all()
    for lo, hi in allowed:                                       # каждый сектор реально встречается
        assert ((xs >= lo) & (xs <= hi)).mean() > 0.3


def test_wind_compass_suite_directions():
    froms = [round(direction_to_from_deg(c.wind.direction)) for c in make_suite("wind_compass")]
    assert froms == [0, 45, 90, 135, 180, 225, 270, 315]
    assert make_suite("wind_compass")[0].to_record()["wind_from_deg"] == pytest.approx(0.0, abs=1e-9)
