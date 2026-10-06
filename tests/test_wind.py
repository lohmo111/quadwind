"""Модели ветра: статистики процессов, форма порывов, сценарные события, детерминизм."""
import numpy as np
import pytest

from quadwind.wind import (COMPASS, LiveWind, WindEvent, WindParams, ar1, direction_to_from_deg, dryden,
                           dryden_scales, from_deg_to_direction, generate_wind, gust_shape, wind_from)

DT = 0.02


def autocorr(x, lag):
    x = x - x.mean()
    return np.dot(x[:-lag], x[lag:]) / np.dot(x, x)


def test_zero_config_gives_zero_wind():
    w, slow = generate_wind(WindParams(), 500, DT, np.random.default_rng(0))
    assert not w.any() and not slow.any()


def test_ou_statistics():
    rng = np.random.default_rng(0)
    tau, sigma = 0.5, 0.8
    x = ar1(400_000, tau, sigma, DT, rng, size=1)[:, 0]
    assert x.std() == pytest.approx(sigma, rel=0.03)
    assert autocorr(x, int(tau / DT)) == pytest.approx(np.exp(-1), abs=0.03)


def test_ou_starts_stationary():
    rng = np.random.default_rng(1)
    starts = np.array([ar1(5, 5.0, 1.0, DT, rng)[0, 0] for _ in range(4000)])
    assert starts.std() == pytest.approx(1.0, rel=0.05)


def test_dryden_variances_and_time_scale():
    rng = np.random.default_rng(2)
    airspeed, altitude, sigma = 3.0, 1.5, 0.5
    uvw = dryden(300_000, DT, airspeed, altitude, sigma, rng)
    l_u, l_w, s_u, s_w = dryden_scales(altitude, sigma)
    assert uvw[:, 0].std() == pytest.approx(s_u, rel=0.1)
    assert uvw[:, 1].std() == pytest.approx(s_u, rel=0.1)
    assert uvw[:, 2].std() == pytest.approx(s_w, rel=0.1)
    # продольная компонента — фильтр 1-го порядка с T = L_u / V
    lag = int(round(l_u / airspeed / DT))
    assert autocorr(uvw[:, 0], lag) == pytest.approx(np.exp(-1), abs=0.08)
    assert s_w < s_u


def test_gust_shape():
    t = np.linspace(0, 10, 1001)
    g = gust_shape(t, 4.0, 2.0)
    assert g.max() == pytest.approx(1.0, abs=1e-6)
    assert t[np.argmax(g)] == pytest.approx(5.0, abs=0.01)
    assert not g[t < 4.0].any() and not g[t > 6.0].any()


def test_scenario_events():
    wp = WindParams(events=(WindEvent("step", 5.0, (2.0, 0.0, 0.0)), WindEvent("pulse", 8.0, (0.0, 3.0, 0.0), 1.0)))
    w, slow = generate_wind(wp, 750, DT, np.random.default_rng(0))
    t = np.arange(750) * DT
    assert np.allclose(w[t < 5.0], 0.0)
    assert np.allclose(w[(t >= 5.0) & (t < 8.0)], [2.0, 0.0, 0.0])
    assert w[:, 1].max() == pytest.approx(3.0, abs=1e-6)
    assert np.allclose(slow[t >= 5.0, 0], 2.0) and not slow[:, 1].any()   # порыв не входит в медленную часть


def test_determinism_and_slow_part():
    wp = WindParams(speed=2.0, direction=1.0, jump_rate=0.3, jump_speed=(0.0, 3.0), slow_sigma=0.3,
                    turb_sigma=0.4, gust_rate=0.2)
    a = generate_wind(wp, 750, DT, np.random.default_rng(5))
    b = generate_wind(wp, 750, DT, np.random.default_rng(5))
    c = generate_wind(wp, 750, DT, np.random.default_rng(6))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.allclose(a[0], c[0])
    calm = WindParams(speed=2.0, jump_rate=0.5, jump_speed=(0.0, 3.0), slow_sigma=0.3)
    w, slow = generate_wind(calm, 750, DT, np.random.default_rng(7))
    assert np.array_equal(w, slow)          # без турбулентности и порывов ветер = медленная часть


def test_base_wind_direction():
    w, _ = generate_wind(WindParams(speed=2.0, direction=np.pi / 2, vertical=0.3), 10, DT, np.random.default_rng(0))
    assert np.allclose(w, [0.0, 2.0, 0.3], atol=1e-12)


def test_compass_convention():
    """Метеорологическая конвенция: северный ветер дует на юг (−y), западный — на восток (+x)."""
    assert np.allclose(wind_from(COMPASS["С"], 3.0), [0.0, -3.0, 0.0], atol=1e-12)
    assert np.allclose(wind_from(COMPASS["В"], 3.0), [-3.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(wind_from(COMPASS["Ю"], 3.0), [0.0, 3.0, 0.0], atol=1e-12)
    assert np.allclose(wind_from(COMPASS["З"], 3.0), [3.0, 0.0, 0.0], atol=1e-12)
    for deg in np.linspace(0, 359, 37):
        assert direction_to_from_deg(from_deg_to_direction(deg)) == pytest.approx(deg, abs=1e-9)
        w, _ = generate_wind(WindParams(speed=2.0, direction=from_deg_to_direction(deg)), 3, DT, np.random.default_rng(0))
        assert np.allclose(w[0], wind_from(deg, 2.0), atol=1e-12)


def test_live_wind():
    lw = LiveWind(DT, seed=3)
    lw.speed, lw.from_deg = 2.0, COMPASS["З"]
    assert np.allclose(lw.step(), [2.0, 0.0, 0.0])                       # без турбулентности — ровно база
    lw.turb_sigma, lw.turb_tau = 0.5, 0.3
    xs = np.array([lw.step() for _ in range(100_000)]) - lw.base()
    assert xs[:, 0].std() == pytest.approx(0.5, rel=0.05) and xs[:, 2].std() == pytest.approx(0.25, rel=0.05)
    lw.turb_sigma = 0.0
    for _ in range(200):                                                  # турбулентность затухла
        lw.step()
    lw.gust(3.0, duration=1.0, from_deg=COMPASS["С"])
    peak = np.array([lw.step() for _ in range(60)])
    assert peak[:, 1].min() == pytest.approx(-3.0, abs=0.01)             # порыв с севера толкает на юг
    assert np.allclose(lw.step(), [2.0, 0.0, 0.0], atol=1e-6) and not lw._gusts
    a, b = LiveWind(DT, seed=7), LiveWind(DT, seed=7)
    a.turb_sigma = b.turb_sigma = 0.4
    assert np.array_equal(np.array([a.step() for _ in range(50)]), np.array([b.step() for _ in range(50)]))
